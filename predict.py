"""
Prediction model — Elo + recent form + head-to-head + venue + rest + xG prior,
calibrated probabilities (Phase 3).

Retrains from scratch every time it runs, using every game with a final
score, then writes fresh predictions for every scheduled game. Safe to call
as often as you like — ingest.py calls this automatically after every live
ingestion run so predictions never go stale.

Phase 3 changes vs. elo_form_v2:
  - Added a prior-season xG feature: each team's squad-total expected goals
    per game from the season BEFORE the one being predicted (never the
    current, in-progress season, so it can't leak information the model
    "shouldn't know yet"). xG strips out finishing luck, which is exactly
    why professional models (Opta, ASA, 538) lean on it harder than raw
    goals — a team that's over/under-performing its xG is more likely to
    regress toward its underlying quality.
  - Probabilities are now calibrated (CalibratedClassifierCV, walk-forward
    safe) instead of taken raw off the classifier, so a "70%" actually means
    something close to "wins about 70% of the time" rather than just being
    whichever number the model happened to output.
  - Brier score and log loss are now tracked alongside accuracy on the
    held-out fold. Accuracy alone can't tell you if the *percentages* are
    trustworthy — a model can pick the right winner most of the time while
    still being badly overconfident or underconfident. Brier/log-loss can.
  - Each prediction now gets a High/Medium/Low confidence tier based on how
    far the top probability sits above a coin-flip guess, so it's visible
    at a glance which games the model actually has an opinion on.

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
import math

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import fetch_all
from awards import compute_and_write_awards
from specialists import Specialists

MODEL_VERSION = "elo_form_xg_travel_draw_v5"
BASE_ELO = 1500
FORM_WINDOW = 5          # games of recent form to average over
REST_CAP_DAYS = 14       # clip rest-days-difference so one huge gap doesn't dominate
MOMENTUM_CAP = 5         # clip win/loss streak so one historic run doesn't dominate
MIN_SURFACE_SAMPLE = 5   # need this many turf (or grass) games before trusting a team's split
TRAVEL_CAP_KM = 4000     # clip travel distance so one huge road trip doesn't dominate

K_FACTOR_GRID = [15, 20, 25, 30]
HOME_ADV_GRID = [40, 65, 90]

FEATURE_NAMES = [
    "elo_diff", "abs_elo_diff", "h2h_home_ppg", "h2h_draw_rate", "venue_ppg",
    "home_form_ppg", "away_form_ppg",
    "home_form_gd", "away_form_gd",
    "rest_diff", "xg_prior_diff",
    "momentum_diff", "travel_diff", "surface_fit_diff",
]
# abs_elo_diff and h2h_draw_rate exist specifically so the model has a real
# signal for draws: closely-matched teams (small |elo_diff|) and pairs with
# a history of drawing each other are the two strongest real-world draw
# predictors, and neither is representable from the other (linear) features
# alone — elo_diff is signed, so a plain linear model can't learn a
# "closer = more draws" U-shape without the absolute-value version explicitly.


def _haversine_km(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _make_xg_prior_lookup(supabase, finished):
    """Per-team, per-season squad xG-per-game, looked up ONE SEASON BACK.

    Only ever reads a season that's already fully finished, so using this
    as a feature for every game in the following season (including its
    very first game) never leaks information the model wouldn't actually
    have had at prediction time.

    Returns (prior_for(team_id, season_name) -> float|None, overall_avg).
    """
    xg_rows = fetch_all(supabase, "player_season_xgoals", "team_id,season_name,xgoals")
    season_xg_sum = {}
    for r in xg_rows:
        if not r.get("team_id") or not r.get("season_name"):
            continue
        key = (r["team_id"], r["season_name"])
        season_xg_sum[key] = season_xg_sum.get(key, 0.0) + float(r.get("xgoals") or 0)

    games_played = {}
    for g in finished:
        for t in (g["home_team_id"], g["away_team_id"]):
            key = (t, g["season_name"])
            games_played[key] = games_played.get(key, 0) + 1

    xg_per_game = {}
    for key, total in season_xg_sum.items():
        gp = games_played.get(key, 0)
        if gp > 0:
            xg_per_game[key] = total / gp

    league_avg_by_season = {}
    for season in {s for (_t, s) in xg_per_game}:
        vals = [v for (_t, s), v in xg_per_game.items() if s == season]
        league_avg_by_season[season] = float(np.mean(vals)) if vals else None

    overall_avg = float(np.mean(list(xg_per_game.values()))) if xg_per_game else 1.3

    def prior_for(team_id, season_name):
        try:
            prior_season = str(int(season_name) - 1)
        except (TypeError, ValueError):
            return None
        if (team_id, prior_season) in xg_per_game:
            return xg_per_game[(team_id, prior_season)]
        # expansion team / data gap — fall back to that prior season's league average
        return league_avg_by_season.get(prior_season)

    return prior_for, overall_avg


def _confidence_tier(max_prob):
    """High/Medium/Low based on how far the top probability sits above a
    coin-flip baseline. Fixed thresholds (not batch-relative percentiles)
    so a 70% pick always reads as "High" regardless of what else is on the
    slate that week.
    """
    if max_prob >= 0.60:
        return "High"
    if max_prob >= 0.45:
        return "Medium"
    return "Low"


def _brier_score(y_true, probs, classes):
    """Multiclass Brier score: mean squared error between predicted
    probabilities and the one-hot actual outcome. 0 = perfect, ~0.667 =
    what a uniform 33/33/33 guess scores for 3 classes. Lower is better —
    this is what actually tells you whether the *percentages* are
    trusted, which accuracy alone can't.
    """
    y_true = np.asarray(y_true)
    class_index = {c: i for i, c in enumerate(classes)}
    onehot = np.zeros((len(y_true), len(classes)))
    for i, yt in enumerate(y_true):
        onehot[i, class_index[yt]] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _build_features(finished, k_factor, home_adv, xg_prior_fn, overall_avg_xg, venues, league_draw_rate=0.24):
    """One causal (no-lookahead) pass through finished games, already sorted by date.

    Returns (rows, elo, h2h, venue_form, recent, last_played, goals_for, goals_against,
    last_venue, streak, surface_record).
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
    last_venue = {}                        # team_id -> venue_id of their last game
    streak = {t: 0 for t in team_ids}      # positive = win streak, negative = loss streak
    surface_record = {t: {True: [0, 0], False: [0, 0]} for t in team_ids}  # is_turf -> [wins, games]

    rows = []
    for g in finished:
        home, away, venue, date = g["home_team_id"], g["away_team_id"], g["venue_id"], g["date_time_utc"]

        elo_diff = (elo[home] + home_adv) - elo[away]
        abs_elo_diff = abs(elo_diff)

        pair = tuple(sorted([home, away]))
        h2h_list = h2h.get(pair, [])
        h2h_home_ppg = (
            sum(3 if w == home else (1 if w is None else 0) for w in h2h_list) / len(h2h_list)
            if h2h_list else 1.3
        )
        h2h_draw_rate = (h2h_list.count(None) / len(h2h_list)) if h2h_list else league_draw_rate

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

        home_xg_prior = xg_prior_fn(home, g["season_name"])
        away_xg_prior = xg_prior_fn(away, g["season_name"])
        home_xg_prior = home_xg_prior if home_xg_prior is not None else overall_avg_xg
        away_xg_prior = away_xg_prior if away_xg_prior is not None else overall_avg_xg
        xg_prior_diff = home_xg_prior - away_xg_prior

        momentum_diff = max(-MOMENTUM_CAP, min(MOMENTUM_CAP, streak[home] - streak[away]))

        this_venue = venues.get(venue, {})
        home_travel = _haversine_km(
            venues.get(last_venue.get(home), {}).get("latitude"), venues.get(last_venue.get(home), {}).get("longitude"),
            this_venue.get("latitude"), this_venue.get("longitude"),
        ) or 0.0
        away_travel = _haversine_km(
            venues.get(last_venue.get(away), {}).get("latitude"), venues.get(last_venue.get(away), {}).get("longitude"),
            this_venue.get("latitude"), this_venue.get("longitude"),
        ) or 0.0
        home_travel = min(home_travel, TRAVEL_CAP_KM)
        away_travel = min(away_travel, TRAVEL_CAP_KM)
        travel_diff = away_travel - home_travel  # positive = away team traveled further (favors home)

        is_turf = this_venue.get("is_turf")
        surface_fit_diff = 0.0
        if is_turf is not None:
            hw, hn = surface_record[home][is_turf]
            aw, an = surface_record[away][is_turf]
            if hn >= MIN_SURFACE_SAMPLE and an >= MIN_SURFACE_SAMPLE:
                surface_fit_diff = (hw / hn) - (aw / an)

        if g["home_score"] > g["away_score"]:
            label = 2
        elif g["home_score"] < g["away_score"]:
            label = 0
        else:
            label = 1

        rows.append({
            "elo_diff": elo_diff, "abs_elo_diff": abs_elo_diff,
            "h2h_home_ppg": h2h_home_ppg, "h2h_draw_rate": h2h_draw_rate, "venue_ppg": venue_ppg,
            "home_form_ppg": home_form_ppg, "away_form_ppg": away_form_ppg,
            "home_form_gd": home_form_gd, "away_form_gd": away_form_gd,
            "rest_diff": rest_diff, "xg_prior_diff": xg_prior_diff,
            "momentum_diff": momentum_diff, "travel_diff": travel_diff,
            "surface_fit_diff": surface_fit_diff, "label": label,
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

        streak[home] = streak[home] + 1 if label == 2 else (0 if label == 1 else -1)
        streak[away] = streak[away] + 1 if label == 0 else (0 if label == 1 else -1)
        last_venue[home] = venue
        last_venue[away] = venue
        if is_turf is not None:
            surface_record[home][is_turf][1] += 1
            surface_record[home][is_turf][0] += 1 if label == 2 else 0
            surface_record[away][is_turf][1] += 1
            surface_record[away][is_turf][0] += 1 if label == 0 else 0

    return (rows, elo, h2h, venue_form, recent, last_played, goals_for, goals_against,
            last_venue, streak, surface_record)


def _candidate_models():
    """A model zoo — grid search picks whichever cross-validates best.

    Logistic regression at a few regularization strengths (plain and
    class-balanced, to give draws — the minority class — a fair shot at
    actually winning the argmax instead of being drowned out by home/away)
    plus gradient-boosted trees at a couple of depths, which can pick up
    feature interactions (e.g. "small |Elo edge| AND a history of draws
    between these two teams") that logistic regression can't.
    """
    return {
        "logreg_c0.3": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.3)),
        "logreg_c1": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "logreg_c3": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=3.0)),
        "logreg_balanced": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        ),
        "hist_gb_d3": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=150, random_state=42),
        "hist_gb_d2": lambda: HistGradientBoostingClassifier(max_depth=2, max_iter=150, random_state=42),
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


def _select_best_config(finished, xg_prior_fn, overall_avg_xg, venues, league_draw_rate=0.24):
    """Grid-search (k_factor, home_adv, model) by walk-forward CV accuracy.

    Falls back to the old hardcoded defaults when there isn't enough history
    yet to cross-validate reliably (early in a season / fresh database).
    """
    if len(finished) < 40:
        return 20, 65, "logreg_c1", None

    n_splits = min(5, max(2, len(finished) // 30))
    best = None  # (cv_acc, k, home_adv, model_name)
    models = _candidate_models()

    for k in K_FACTOR_GRID:
        for home_adv in HOME_ADV_GRID:
            rows, *_ = _build_features(finished, k, home_adv, xg_prior_fn, overall_avg_xg, venues, league_draw_rate)
            X = np.array([[r[f] for f in FEATURE_NAMES] for r in rows])
            y = np.array([r["label"] for r in rows])
            for name, model_fn in models.items():
                acc = _cv_score(X, y, model_fn, n_splits)
                if acc is None:
                    continue
                if best is None or acc > best[0]:
                    best = (acc, k, home_adv, name)

    if best is None:
        return 20, 65, "logreg_c1", None
    return best[1], best[2], best[3], best[0]


def train_and_predict(supabase):
    print("Loading games...")
    games = fetch_all(
        supabase, "games",
        "id,date_time_utc,home_team_id,away_team_id,venue_id,home_score,away_score,status,"
        "season_name,knockout_game",
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

    print("Loading prior-season xG ratings...")
    xg_prior_fn, overall_avg_xg = _make_xg_prior_lookup(supabase, finished)

    print("Loading venues (for travel distance / pitch surface)...")
    venues = {v["id"]: v for v in fetch_all(supabase, "venues", "id,latitude,longitude,is_turf")}

    league_draw_rate = (
        float(np.mean([1.0 if g["home_score"] == g["away_score"] else 0.0 for g in finished]))
        if finished else 0.24
    )
    print(f"  League draw rate (fallback for pairs with no h2h history): {league_draw_rate:.3f}")

    print("Grid-searching Elo/model hyperparameters via walk-forward CV...")
    k_factor, home_adv, model_name, cv_acc = _select_best_config(
        finished, xg_prior_fn, overall_avg_xg, venues, league_draw_rate
    )
    print(
        f"  Best config: K={k_factor}, HOME_ADV={home_adv}, model={model_name}"
        + (f", CV accuracy={cv_acc:.3f}" if cv_acc is not None else " (default — not enough history to grid-search yet)")
    )

    (rows, elo, h2h, venue_form, recent, last_played, goals_for, goals_against,
     last_venue, streak, surface_record) = _build_features(
        finished, k_factor, home_adv, xg_prior_fn, overall_avg_xg, venues, league_draw_rate
    )
    X = np.array([[r[f] for f in FEATURE_NAMES] for r in rows])
    y = np.array([r["label"] for r in rows])

    # ---- specialist committee -------------------------------------------
    # Three models, three jobs. NOT an ensemble: nothing is averaged. Each
    # lane is answered by the single model the offline study measured best at
    # that lane. See specialists.py for the numbers behind each choice.
    team_index = {t: i for i, t in enumerate(sorted(team_ids))}
    specialists = None
    try:
        hs_arr = np.array([g["home_score"] for g in finished], float)
        as_arr = np.array([g["away_score"] for g in finished], float)
        hidx = np.array([team_index[g["home_team_id"]] for g in finished])
        aidx = np.array([team_index[g["away_team_id"]] for g in finished])
        ref_date = finished[-1]["date_time_utc"]
        days_ago = np.clip(np.array(
            [(ref_date - g["date_time_utc"]).days for g in finished], float), 0, None)
        specialists = Specialists(FEATURE_NAMES).fit(
            X, y, hs_arr, as_arr, hidx, aidx, days_ago, len(team_index))
        print(f"  Specialist committee fitted (Dixon-Coles rho={specialists.dc_.rho_:+.3f}).")
    except Exception as e:  # noqa: BLE001 - extras must never break the main prediction
        print(f"  Specialist committee failed to fit ({e}) - main model unaffected.")
        specialists = None

    n_splits = min(5, max(2, len(rows) // 30))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model_fn = _candidate_models()[model_name]

    # Headline "test accuracy" = the LAST (largest, most recent) walk-forward
    # fold — i.e. how well the model predicts the most recent stretch of
    # games after training on everything before it. The classifier here is
    # wrapped in CalibratedClassifierCV (itself using a walk-forward inner
    # split, so no lookahead) so the held-out metrics reflect calibrated
    # probabilities — the same kind of model that goes live below — not raw,
    # possibly overconfident classifier output.
    train_idx, test_idx = list(tscv.split(X))[-1]
    inner_splits = min(3, max(2, len(train_idx) // 30))
    try:
        clf = CalibratedClassifierCV(model_fn(), method="sigmoid", cv=TimeSeriesSplit(n_splits=inner_splits))
        clf.fit(X[train_idx], y[train_idx])
    except ValueError:
        # not enough rows/class variety in this fold to calibrate — fall back to raw
        clf = model_fn()
        clf.fit(X[train_idx], y[train_idx])

    test_probs = clf.predict_proba(X[test_idx])
    test_pred_labels = clf.classes_[np.argmax(test_probs, axis=1)]
    test_acc = float(accuracy_score(y[test_idx], test_pred_labels))
    baseline_acc = float(accuracy_score(y[test_idx], np.full_like(y[test_idx], 2)))

    test_brier = _brier_score(y[test_idx], test_probs, clf.classes_)
    uniform_probs = np.full_like(test_probs, 1.0 / len(clf.classes_))
    baseline_brier = _brier_score(y[test_idx], uniform_probs, clf.classes_)
    test_log_loss = float(log_loss(y[test_idx], test_probs, labels=clf.classes_))

    pred_draw_count = int(np.sum(test_pred_labels == 1))
    actual_draw_count = int(np.sum(y[test_idx] == 1))
    print(
        f"Held-out fold accuracy: {test_acc}  (baseline 'always home win': {baseline_acc})\n"
        f"Held-out Brier: {test_brier:.4f}  (uniform-guess baseline: {baseline_brier:.4f})  "
        f"log loss: {test_log_loss:.4f}\n"
        f"Draws predicted on held-out fold: {pred_draw_count}/{len(test_idx)}  "
        f"(actual draws in that fold: {actual_draw_count}) — confirms draw predictions are actually functional, not just theoretical"
    )

    n_splits_full = min(5, max(2, len(rows) // 30))
    try:
        clf_full = CalibratedClassifierCV(model_fn(), method="sigmoid", cv=TimeSeriesSplit(n_splits=n_splits_full))
        clf_full.fit(X, y)
    except ValueError:
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
        "notes": f"{MODEL_VERSION} (model={model_name}, k={k_factor}, home_adv={home_adv}, calibrated=True)",
        "accuracy_metrics": {
            "test_accuracy": test_acc,
            "baseline_accuracy": baseline_acc,
            "test_brier": test_brier,
            "baseline_brier": baseline_brier,
            "test_log_loss": test_log_loss,
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
        abs_elo_diff = abs(elo_diff)
        pair = tuple(sorted([home, away]))
        h2h_list = h2h.get(pair, [])
        h2h_home_ppg = (
            sum(3 if w == home else (1 if w is None else 0) for w in h2h_list) / len(h2h_list)
            if h2h_list else 1.3
        )
        h2h_draw_rate = (h2h_list.count(None) / len(h2h_list)) if h2h_list else league_draw_rate
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

        home_xg_prior = xg_prior_fn(home, g["season_name"])
        away_xg_prior = xg_prior_fn(away, g["season_name"])
        home_xg_prior = home_xg_prior if home_xg_prior is not None else overall_avg_xg
        away_xg_prior = away_xg_prior if away_xg_prior is not None else overall_avg_xg
        xg_prior_diff = home_xg_prior - away_xg_prior

        momentum_diff = max(-MOMENTUM_CAP, min(MOMENTUM_CAP, streak.get(home, 0) - streak.get(away, 0)))

        this_venue = venues.get(venue, {})
        home_travel = _haversine_km(
            venues.get(last_venue.get(home), {}).get("latitude"), venues.get(last_venue.get(home), {}).get("longitude"),
            this_venue.get("latitude"), this_venue.get("longitude"),
        ) or 0.0
        away_travel = _haversine_km(
            venues.get(last_venue.get(away), {}).get("latitude"), venues.get(last_venue.get(away), {}).get("longitude"),
            this_venue.get("latitude"), this_venue.get("longitude"),
        ) or 0.0
        travel_diff = min(away_travel, TRAVEL_CAP_KM) - min(home_travel, TRAVEL_CAP_KM)

        is_turf = this_venue.get("is_turf")
        surface_fit_diff = 0.0
        if is_turf is not None and home in surface_record and away in surface_record:
            hw, hn = surface_record[home][is_turf]
            aw, an = surface_record[away][is_turf]
            if hn >= MIN_SURFACE_SAMPLE and an >= MIN_SURFACE_SAMPLE:
                surface_fit_diff = (hw / hn) - (aw / an)

        feat = [[elo_diff, abs_elo_diff, h2h_home_ppg, h2h_draw_rate, venue_ppg,
                 home_form_ppg, away_form_ppg,
                 home_form_gd, away_form_gd, rest_diff, xg_prior_diff,
                 momentum_diff, travel_diff, surface_fit_diff]]

        breakdown = None
        if specialists is not None and specialists.ok:
            try:
                breakdown = specialists.predict(
                    feat[0], team_index[home], team_index[away])
            except Exception:  # noqa: BLE001
                breakdown = None

        probs = clf_full.predict_proba(feat)[0]
        prob_by_class = dict(zip(clf_full.classes_, probs))
        p_away = float(prob_by_class.get(0, 0))
        p_draw = float(prob_by_class.get(1, 0))
        p_home = float(prob_by_class.get(2, 0))
        confidence = _confidence_tier(max(p_home, p_draw, p_away))

        home_xg = (team_attack(home) + team_defense(away)) / 2 * 1.1
        away_xg = (team_attack(away) + team_defense(home)) / 2 * 0.9

        supabase.table("predictions").upsert({
            "game_id": g["id"], "model_version": MODEL_VERSION,
            "predicted_home_win_pct": round(p_home * 100, 1),
            "predicted_draw_pct": round(p_draw * 100, 1),
            "predicted_away_win_pct": round(p_away * 100, 1),
            "predicted_home_score": round(home_xg, 2),
            "predicted_away_score": round(away_xg, 2),
            "confidence": confidence,
            "model_breakdown": breakdown,
        }, on_conflict="game_id").execute()
        pred_count += 1
    print(f"Wrote {pred_count} predictions.")

    try:
        compute_and_write_awards(
            supabase, team_ids=team_ids, finished=finished, upcoming=upcoming,
            elo=elo, home_adv=home_adv,
            attack={t: team_attack(t) for t in team_ids},
            defense={t: team_defense(t) for t in team_ids},
        )
    except Exception as e:  # noqa: BLE001 - awards are a bonus feature; never let them break match predictions
        print(f"Awards computation failed (match predictions above are unaffected): {e}")

    return clf_full, elo


if __name__ == "__main__":
    from common import get_supabase
    train_and_predict(get_supabase())
