"""
Prediction model — Elo + recent form + head-to-head + venue + rest (Phase 2).

Retrains from scratch every time it runs, using every game with a final
score, then writes fresh predictions for every scheduled game. Safe to call
as often as you like — ingest.py calls this automatically after every live
ingestion run so predictions never go stale.

Phase 2 changes vs. the original elo_h2h_v1 model:
  - Added recent-form features (last-5-game points-per-game and goal
    difference for each team) and a rest-days-difference feature, instead of
    relying on Elo + head-to-head + venue alone.
  - K_FACTOR and HOME_ADV are no longer hardcoded guesses — every retrain
    grid-searches them (plus a choice of classifier) using walk-forward
    (time-series) cross-validation, so the constants are actually fit to the
    league's real data instead of eyeballed.
  - Accuracy is now reported as a cross-validated average across several
    chronological folds, not a single 15%-holdout split, so the number in
    the dashboard is a much more stable estimate of real predictive power.
"""
import datetime

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import fetch_all

MODEL_VERSION = "elo_form_v2"
BASE_ELO = 1500
FORM_WINDOW = 5          # games of recent form to average over
REST_CAP_DAYS = 14       # clip rest-days-difference so one huge gap doesn't dominate

K_FACTOR_GRID = [15, 20, 25, 30]
HOME_ADV_GRID = [40, 65, 90]

FEATURE_NAMES = [
    "elo_diff", "h2h_home_ppg", "venue_ppg",
    "home_form_ppg", "away_form_ppg",
    "home_form_gd", "away_form_gd",
    "rest_diff",
]


def _build_features(finished, k_factor, home_adv):
    """One causal (no-lookahead) pass through finished games, already sorted by date.

    Returns (rows, elo, h2h, venue_form, recent, last_played, goals_for, goals_against).
    The trailing dicts hold FINAL state after every finished game, ready to be
    reused as-is to build features for upcoming fixtures.
    """
    team_ids = sorted({g["home_team_id"] for g in finished} | {g["away_team_id"] for g in finished})
    elo = {t: float(BASE_ELO) for t in team_ids}
    h2h = {}
    venue_form = {}
    recent = {t: [] for t in team_ids}     # list of (points, goal_diff), oldest -> newest
    last_played = {}
    goals_for = {t: [] for t in team_ids}
    goals_against = {t: [] for t in team_ids}

    rows = []
    for g in finished:
        home, away, venue, date = g["home_team_id"], g["away_team_id"], g["venue_id"], g["date_time_utc"]

        elo_diff = (elo[home] + home_adv) - elo[away]

        pair = tuple(sorted([home, away]))
        h2h_list = h2h.get(pair, [])
        h2h_home_ppg = (
            sum(3 if w == home else (1 if w is None else 0) for w in h2h_list) / len(h2h_list)
            if h2h_list else 1.3
        )

        vf_list = venue_form.get((home, venue), [])
        venue_ppg = (sum(vf_list) / len(vf_list)) if vf_list else 1.5

        home_recent = recent[home][-FORM_WINDOW:]
        away_recent = recent[away][-FORM_WINDOW:]
        home_form_ppg = float(np.mean([p for p, _ in home_recent])) if home_recent else 1.3
        away_form_ppg = float(np.mean([p for p, _ in away_recent])) if away_recent else 1.3
        home_form_gd = float(np.mean([gd for _, gd in home_recent])) if home_recent else 0.0
        away_form_gd = float(np.mean([gd for _, gd in away_recent])) if away_recent else 0.0

        home_rest = min((date - last_played[home]).days, REST_CAP_DAYS) if home in last_played else REST_CAP_DAYS
        away_rest = min((date - last_played[away]).days, REST_CAP_DAYS) if away in last_played else REST_CAP_DAYS
        rest_diff = max(-REST_CAP_DAYS, min(REST_CAP_DAYS, home_rest - away_rest))

        if g["home_score"] > g["away_score"]:
            label = 2
        elif g["home_score"] < g["away_score"]:
            label = 0
        else:
            label = 1

        rows.append({
            "elo_diff": elo_diff, "h2h_home_ppg": h2h_home_ppg, "venue_ppg": venue_ppg,
            "home_form_ppg": home_form_ppg, "away_form_ppg": away_form_ppg,
            "home_form_gd": home_form_gd, "away_form_gd": away_form_gd,
            "rest_diff": rest_diff, "label": label,
        })

        exp_home = 1 / (1 + 10 ** (-elo_diff / 400))
        actual_home = 1.0 if label == 2 else (0.5 if label == 1 else 0.0)
        elo[home] += k_factor * (actual_home - exp_home)
        elo[away] += k_factor * ((1 - actual_home) - (1 - exp_home))

        winner = home if label == 2 else (away if label == 0 else None)
        h2h.setdefault(pair, []).append(winner)
        pts_home = 3 if label == 2 else (1 if label == 1 else 0)
        pts_away = 3 if label == 0 else (1 if label == 1 else 0)
        venue_form.setdefault((home, venue), []).append(pts_home)

        recent[home].append((pts_home, g["home_score"] - g["away_score"]))
        recent[away].append((pts_away, g["away_score"] - g["home_score"]))
        last_played[home] = date
        last_played[away] = date

        goals_for[home].append(g["home_score"]); goals_against[home].append(g["away_score"])
        goals_for[away].append(g["away_score"]); goals_against[away].append(g["home_score"])

    return rows, elo, h2h, venue_form, recent, last_played, goals_for, goals_against


def _candidate_models():
    """A small model zoo — grid search picks whichever cross-validates best.

    Logistic regression (plain and class-balanced, to give draws a fairer
    shot) plus a gradient-boosted tree model, which can pick up feature
    interactions (e.g. "big Elo edge AND well-rested") that logistic
    regression can't.
    """
    return {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "logreg_balanced": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        ),
        "hist_gb": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=150, random_state=42),
    }


def _cv_score(X, y, model_fn, n_splits):
    """Average accuracy across walk-forward (chronological) folds."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs = []
    for train_idx, test_idx in tscv.split(X):
        if len(set(y[train_idx])) < 2:
            continue
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        accs.append(accuracy_score(y[test_idx], model.predict(X[test_idx])))
    return float(np.mean(accs)) if accs else None


def _select_best_config(finished):
    """Grid-search (k_factor, home_adv, model) by walk-forward CV accuracy.

    Falls back to the old hardcoded defaults when there isn't enough history
    yet to cross-validate reliably (early in a season / fresh database).
    """
    if len(finished) < 40:
        return 20, 65, "logreg", None

    n_splits = min(5, max(2, len(finished) // 30))
    best = None  # (cv_acc, k, home_adv, model_name)
    models = _candidate_models()

    for k in K_FACTOR_GRID:
        for home_adv in HOME_ADV_GRID:
            rows, *_ = _build_features(finished, k, home_adv)
            X = np.array([[r[f] for f in FEATURE_NAMES] for r in rows])
            y = np.array([r["label"] for r in rows])
            for name, model_fn in models.items():
                acc = _cv_score(X, y, model_fn, n_splits)
                if acc is None:
                    continue
                if best is None or acc > best[0]:
                    best = (acc, k, home_adv, name)

    if best is None:
        return 20, 65, "logreg", None
    return best[1], best[2], best[3], best[0]


def train_and_predict(supabase):
    print("Loading games...")
    games = fetch_all(
        supabase, "games",
        "id,date_time_utc,home_team_id,away_team_id,venue_id,home_score,away_score,status",
        order_col="date_time_utc",
    )
    for g in games:
        g["date_time_utc"] = datetime.datetime.fromisoformat(g["date_time_utc"].replace("Z", "+00:00"))

    finished = [g for g in games if g["status"] == "final" and g["home_score"] is not None and g["away_score"] is not None]
    upcoming = [g for g in games if g["status"] == "scheduled"]
    print(f"  {len(finished)} finished games, {len(upcoming)} scheduled games.")

    team_ids = sorted({g["home_team_id"] for g in games} | {g["away_team_id"] for g in games})

    if len(finished) < 20:
        print("Not enough finished games yet to train a model — skipping.")
        return None, {t: float(BASE_ELO) for t in team_ids}

    print("Grid-searching Elo/model hyperparameters via walk-forward CV...")
    k_factor, home_adv, model_name, cv_acc = _select_best_config(finished)
    print(
        f"  Best config: K={k_factor}, HOME_ADV={home_adv}, model={model_name}"
        + (f", CV accuracy={cv_acc:.3f}" if cv_acc is not None else " (default — not enough history to grid-search yet)")
    )

    rows, elo, h2h, venue_form, recent, last_played, goals_for, goals_against = _build_features(
        finished, k_factor, home_adv
    )
    X = np.array([[r[f] for f in FEATURE_NAMES] for r in rows])
    y = np.array([r["label"] for r in rows])

    n_splits = min(5, max(2, len(rows) // 30))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model_fn = _candidate_models()[model_name]

    # Headline "test accuracy" = the LAST (largest, most recent) walk-forward
    # fold — i.e. how well the model predicts the most recent stretch of
    # games after training on everything before it.
    train_idx, test_idx = list(tscv.split(X))[-1]
    clf = model_fn()
    clf.fit(X[train_idx], y[train_idx])
    test_acc = float(accuracy_score(y[test_idx], clf.predict(X[test_idx])))
    baseline_acc = float(accuracy_score(y[test_idx], np.full_like(y[test_idx], 2)))
    print(f"Held-out fold accuracy: {test_acc}  (baseline 'always home win': {baseline_acc})")

    clf_full = model_fn()
    clf_full.fit(X, y)

    league_avg_goals = float(np.mean([g["home_score"] + g["away_score"] for g in finished]) / 2) if finished else 1.3

    def team_attack(t):
        return float(np.mean(goals_for[t][-20:])) if goals_for.get(t) else league_avg_goals

    def team_defense(t):
        return float(np.mean(goals_against[t][-20:])) if goals_against.get(t) else league_avg_goals

    now_iso = datetime.datetime.utcnow().isoformat()
    for t in team_ids:
        supabase.table("teams").update({
            "elo_rating": round(elo.get(t, BASE_ELO), 1),
            "attack_rating": round(team_attack(t), 3),
            "defense_rating": round(team_defense(t), 3),
            "ratings_updated_at": now_iso,
        }).eq("id", t).execute()
    print(f"Updated ratings for {len(team_ids)} teams.")

    supabase.table("model_runs").insert({
        "training_row_count": len(rows),
        "notes": f"{MODEL_VERSION} (model={model_name}, k={k_factor}, home_adv={home_adv})",
        "accuracy_metrics": {
            "test_accuracy": test_acc,
            "baseline_accuracy": baseline_acc,
            "test_size": int(len(test_idx)),
            "cv_accuracy": cv_acc,
        },
    }).execute()

    print(f"Generating predictions for {len(upcoming)} upcoming games...")
    pred_count = 0
    for g in upcoming:
        home, away, venue, date = g["home_team_id"], g["away_team_id"], g["venue_id"], g["date_time_utc"]
        if home not in elo or away not in elo:
            continue
        elo_diff = (elo[home] + home_adv) - elo[away]
        pair = tuple(sorted([home, away]))
        h2h_list = h2h.get(pair, [])
        h2h_home_ppg = (
            sum(3 if w == home else (1 if w is None else 0) for w in h2h_list) / len(h2h_list)
            if h2h_list else 1.3
        )
        vf_list = venue_form.get((home, venue), [])
        venue_ppg = (sum(vf_list) / len(vf_list)) if vf_list else 1.5

        home_recent = recent.get(home, [])[-FORM_WINDOW:]
        away_recent = recent.get(away, [])[-FORM_WINDOW:]
        home_form_ppg = float(np.mean([p for p, _ in home_recent])) if home_recent else 1.3
        away_form_ppg = float(np.mean([p for p, _ in away_recent])) if away_recent else 1.3
        home_form_gd = float(np.mean([gd for _, gd in home_recent])) if home_recent else 0.0
        away_form_gd = float(np.mean([gd for _, gd in away_recent])) if away_recent else 0.0

        home_rest = min((date - last_played[home]).days, REST_CAP_DAYS) if home in last_played else REST_CAP_DAYS
        away_rest = min((date - last_played[away]).days, REST_CAP_DAYS) if away in last_played else REST_CAP_DAYS
        rest_diff = max(-REST_CAP_DAYS, min(REST_CAP_DAYS, home_rest - away_rest))

        feat = [[elo_diff, h2h_home_ppg, venue_ppg, home_form_ppg, away_form_ppg,
                 home_form_gd, away_form_gd, rest_diff]]

        probs = clf_full.predict_proba(feat)[0]
        prob_by_class = dict(zip(clf_full.classes_, probs))
        p_away = float(prob_by_class.get(0, 0))
        p_draw = float(prob_by_class.get(1, 0))
        p_home = float(prob_by_class.get(2, 0))

        home_xg = (team_attack(home) + team_defense(away)) / 2 * 1.1
        away_xg = (team_attack(away) + team_defense(home)) / 2 * 0.9

        supabase.table("predictions").upsert({
            "game_id": g["id"], "model_version": MODEL_VERSION,
            "predicted_home_win_pct": round(p_home * 100, 1),
            "predicted_draw_pct": round(p_draw * 100, 1),
            "predicted_away_win_pct": round(p_away * 100, 1),
            "predicted_home_score": round(home_xg, 2),
            "predicted_away_score": round(away_xg, 2),
        }, on_conflict="game_id").execute()
        pred_count += 1
    print(f"Wrote {pred_count} predictions.")
    return clf_full, elo


if __name__ == "__main__":
    from common import get_supabase
    train_and_predict(get_supabase())
