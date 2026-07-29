"""
Prediction model — Elo + head-to-head + venue form (Phase 1).

Retrains from scratch every time it runs, using every game with a final
score, then writes fresh predictions for every scheduled game. Safe to call
as often as you like — ingest.py calls this automatically after every live
ingestion run so predictions never go stale.
"""
import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from common import fetch_all

MODEL_VERSION = "elo_h2h_v1"
HOME_ADV = 65
K_FACTOR = 20
BASE_ELO = 1500


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
    elo = {t: float(BASE_ELO) for t in team_ids}
    h2h = {}
    venue_form = {}
    goals_for = {t: [] for t in team_ids}
    goals_against = {t: [] for t in team_ids}

    rows = []
    for g in finished:
        home, away, venue = g["home_team_id"], g["away_team_id"], g["venue_id"]
        elo_diff = (elo[home] + HOME_ADV) - elo[away]
        pair = tuple(sorted([home, away]))
        h2h_list = h2h.get(pair, [])
        if h2h_list:
            home_pts = sum(3 if w == home else (1 if w is None else 0) for w in h2h_list)
            h2h_home_ppg = home_pts / len(h2h_list)
        else:
            h2h_home_ppg = 1.3
        vf_list = venue_form.get((home, venue), [])
        venue_ppg = (sum(vf_list) / len(vf_list)) if vf_list else 1.5

        if g["home_score"] > g["away_score"]:
            label = 2
        elif g["home_score"] < g["away_score"]:
            label = 0
        else:
            label = 1

        rows.append({"elo_diff": elo_diff, "h2h_home_ppg": h2h_home_ppg, "venue_ppg": venue_ppg, "label": label})

        exp_home = 1 / (1 + 10 ** (-elo_diff / 400))
        actual_home = 1.0 if label == 2 else (0.5 if label == 1 else 0.0)
        elo[home] += K_FACTOR * (actual_home - exp_home)
        elo[away] += K_FACTOR * ((1 - actual_home) - (1 - exp_home))

        winner = home if label == 2 else (away if label == 0 else None)
        h2h.setdefault(pair, []).append(winner)
        pts = 3 if label == 2 else (1 if label == 1 else 0)
        venue_form.setdefault((home, venue), []).append(pts)

        goals_for[home].append(g["home_score"]); goals_against[home].append(g["away_score"])
        goals_for[away].append(g["away_score"]); goals_against[away].append(g["home_score"])

    print(f"Computed Elo/H2H/venue features for {len(rows)} games.")

    if len(rows) < 20:
        print("Not enough finished games yet to train a model — skipping.")
        return None, elo

    X = np.array([[r["elo_diff"], r["h2h_home_ppg"], r["venue_ppg"]] for r in rows])
    y = np.array([r["label"] for r in rows])
    split = int(len(X) * 0.85)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    if len(X_test) > 0:
        test_acc = float(accuracy_score(y_test, clf.predict(X_test)))
        baseline_acc = float(accuracy_score(y_test, np.full_like(y_test, 2)))
    else:
        test_acc, baseline_acc = None, None

    print(f"Test accuracy: {test_acc}  (baseline 'always home win': {baseline_acc})")

    clf_full = LogisticRegression(max_iter=1000)
    clf_full.fit(X, y)

    league_avg_goals = float(np.mean([g["home_score"] + g["away_score"] for g in finished]) / 2) if finished else 1.3

    def team_attack(t):
        return float(np.mean(goals_for[t][-20:])) if goals_for[t] else league_avg_goals

    def team_defense(t):
        return float(np.mean(goals_against[t][-20:])) if goals_against[t] else league_avg_goals

    now_iso = datetime.datetime.utcnow().isoformat()
    for t in team_ids:
        supabase.table("teams").update({
            "elo_rating": round(elo[t], 1),
            "attack_rating": round(team_attack(t), 3),
            "defense_rating": round(team_defense(t), 3),
            "ratings_updated_at": now_iso,
        }).eq("id", t).execute()
    print(f"Updated ratings for {len(team_ids)} teams.")

    supabase.table("model_runs").insert({
        "training_row_count": len(rows),
        "notes": MODEL_VERSION,
        "accuracy_metrics": {"test_accuracy": test_acc, "baseline_accuracy": baseline_acc, "test_size": len(X_test)},
    }).execute()

    print(f"Generating predictions for {len(upcoming)} upcoming games...")
    pred_count = 0
    for g in upcoming:
        home, away, venue = g["home_team_id"], g["away_team_id"], g["venue_id"]
        if home not in elo or away not in elo:
            continue
        elo_diff = (elo[home] + HOME_ADV) - elo[away]
        pair = tuple(sorted([home, away]))
        h2h_list = h2h.get(pair, [])
        h2h_home_ppg = (sum(3 if w == home else (1 if w is None else 0) for w in h2h_list) / len(h2h_list)) if h2h_list else 1.3
        vf_list = venue_form.get((home, venue), [])
        venue_ppg = (sum(vf_list) / len(vf_list)) if vf_list else 1.5

        probs = clf_full.predict_proba([[elo_diff, h2h_home_ppg, venue_ppg]])[0]
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
