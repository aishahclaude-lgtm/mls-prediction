"""
Experimental pre-match "power rating" model — NOT wired into the live app.

This is a separate, hand-built alternative to predict.py's calibrated ML model,
built to test an idea: instead of learning feature weights from data, start from
Elo and apply a chain of hand-guessed multipliers for things Elo alone doesn't
see — missing key starters, home field, travel, momentum, pitch surface, and
weather. Every multiplier below is a starting guess (see each constant's comment)
meant to be backtested and retuned, not trusted as-is.

  Rating_team = Elo_team x M_lineup x M_travel x M_momentum x M_surface x M_weather
  Rating_home = Rating_home x M_home        (home field only applies to the home team)

  rating_diff = Rating_home - Rating_away

This script rebuilds Elo the same causal, no-lookahead way predict.py does (never
uses a result to predict itself), computes rating_diff for every finished game,
then checks whether the SIGN and SIZE of rating_diff actually tracks the real
outcome — that's the backtest the graph is built from.

Important honesty notes, not swept under the rug:
  - "Key players" per team are picked from the team's FULL-SEASON totals, which is
    a mild lookahead (a player's early-season games get "key player" status they
    only earned by season's end). Fine for a first exploratory pass; a rigorous
    version would use a rolling/prior-season definition instead.
  - Weather requires calling an external API (Open-Meteo, no key needed) per venue.
    This script does that automatically when it has network access. If the API
    can't be reached (e.g. a network-restricted sandbox), weather silently falls
    back to neutral (M_weather = 1.0 for every game) rather than crashing — check
    the printed warning to know which mode a given run used.

Usage:
    export SUPABASE_URL="https://xxxx.supabase.co"
    export SUPABASE_KEY="your-service-role-key"
    python experimental_rating.py
"""
import datetime
import math

import numpy as np

from common import fetch_all, get_supabase
from awards import _offense_score

MODEL_VERSION = "experimental_multiplier_v1"
BASE_ELO = 1500
K_FACTOR = 20          # same default predict.py falls back to
HOME_ADV = 65           # same default predict.py falls back to

# ---- tunable multiplier guesses — retune all of these once the backtest runs ----
KEY_PLAYERS_PER_TEAM = 4          # how many "difference-makers" per team we track
LINEUP_PENALTY_START = 1.01       # divisor for the 1st missing key player
LINEUP_PENALTY_STEP = 0.05        # each additional missing player adds this much
HOME_MULT = 1.05                  # ~65 Elo pts on a 1500 rating, matches predict.py's own grid search
MOMENTUM_STEP = 0.01              # per game of current win streak, capped below
MOMENTUM_CAP = 0.03               # max +/-3% from streak alone
TRAVEL_FULL_PENALTY_KM = 4000     # distance at which travel penalty maxes out
TRAVEL_MAX_PENALTY = 0.03         # max 3% penalty for a long-haul trip
MIN_SURFACE_SAMPLE = 5            # need this many turf (or grass) games before trusting the split
SURFACE_MAX_SWING = 0.03          # max +/-3% from turf/grass record
MIN_RAIN_SAMPLE = 5               # need this many rain games before trusting the split
RAIN_MAX_SWING = 0.05             # max +/-5% from rain record (shakiest signal — small samples)


def _haversine_km(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _fetch_weather(venues):
    """One Open-Meteo 'historical archive' call per venue, covering every date we
    might need. No API key required. Returns {venue_id: {date: precip_mm}}, or an
    empty dict (with a printed warning) if the API can't be reached — callers must
    treat a missing venue/date as "unknown," not "no rain."
    """
    import requests
    weather = {}
    today = datetime.date.today().isoformat()
    reachable = True
    for v in venues:
        if not v.get("latitude") or not v.get("longitude"):
            continue
        try:
            resp = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": v["latitude"], "longitude": v["longitude"],
                    "start_date": "2024-01-01", "end_date": today,
                    "daily": "precipitation_sum", "timezone": "UTC",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("daily", {})
            days = dict(zip(data.get("time", []), data.get("precipitation_sum", [])))
            weather[v["id"]] = days
        except Exception as e:  # noqa: BLE001 - weather is a bonus signal, never fatal
            reachable = False
            break
    if not reachable or not weather:
        print("  Weather API unreachable from this environment — falling back to "
              "M_weather = 1.0 (neutral) for every game. Run this script somewhere "
              "with normal internet access (e.g. via GitHub Actions) to get real "
              "rain data.")
        return {}
    return weather


def _key_players(xg_rows, players_by_id):
    """team_id -> set of player_ids who are that team's top-K by season offense
    score, across ALL seasons present (simple, mildly lookahead — see module
    docstring)."""
    by_team = {}
    for r in xg_rows:
        if r["player_id"] not in players_by_id:
            continue
        by_team.setdefault(r["team_id"], []).append(r)
    key = {}
    for team_id, rows in by_team.items():
        # collapse multiple season rows per player to their best-scoring season
        best_per_player = {}
        for r in rows:
            s = _offense_score(r)
            if r["player_id"] not in best_per_player or s > best_per_player[r["player_id"]]:
                best_per_player[r["player_id"]] = s
        top = sorted(best_per_player.items(), key=lambda kv: -kv[1])[:KEY_PLAYERS_PER_TEAM]
        key[team_id] = {pid for pid, _ in top}
    return key


def _lineup_multiplier(missing_count):
    if missing_count <= 0:
        return 1.0
    penalty = 1.0
    for i in range(missing_count):
        penalty *= LINEUP_PENALTY_START + LINEUP_PENALTY_STEP * i
    return 1.0 / penalty


def compute_ratings_and_backtest(supabase):
    print("Loading data...")
    games = fetch_all(
        supabase, "games",
        "id,date_time_utc,home_team_id,away_team_id,venue_id,home_score,away_score,status,season_name",
        order_col="date_time_utc",
    )
    venues = {v["id"]: v for v in fetch_all(supabase, "venues", "id,latitude,longitude,is_turf")}
    players = fetch_all(supabase, "players", "id,name")
    players_by_id = {p["id"]: p for p in players}
    xg_rows = fetch_all(supabase, "player_season_xgoals",
                         "player_id,team_id,season_name,goals,primary_assists,xgoals,xassists,points_added")
    pgs = fetch_all(supabase, "player_game_stats", "game_id,player_id,team_id,minutes")

    for g in games:
        g["date_time_utc"] = datetime.datetime.fromisoformat(g["date_time_utc"].replace("Z", "+00:00"))

    finished = [g for g in games if g["status"] == "final" and g["home_score"] is not None]
    finished.sort(key=lambda g: g["date_time_utc"])
    print(f"  {len(finished)} finished games.")

    key_players = _key_players(xg_rows, players_by_id)

    # who actually appeared for which team in which game (for the lineup check)
    appeared = {}  # (game_id, team_id) -> set of player_ids
    for row in pgs:
        appeared.setdefault((row["game_id"], row["team_id"]), set()).add(row["player_id"])

    weather = _fetch_weather(list(venues.values()))
    weather_available = bool(weather)

    team_ids = sorted({g["home_team_id"] for g in finished} | {g["away_team_id"] for g in finished})
    elo = {t: float(BASE_ELO) for t in team_ids}
    last_venue = {}
    streak = {t: 0 for t in team_ids}  # positive = win streak, negative = loss streak
    surface_record = {t: {True: [0, 0], False: [0, 0]} for t in team_ids}  # is_turf -> [wins, games]
    rain_record = {t: [0, 0] for t in team_ids}  # [wins, games] in rain

    results = []  # (rating_diff, home_win, draw, away_win)

    def team_rating(t, game, is_home):
        venue = venues.get(game["venue_id"], {})
        r = elo[t]

        missing = len(key_players.get(t, set()) - appeared.get((game["id"], t), set()))
        r *= _lineup_multiplier(missing)

        prev_v = last_venue.get(t)
        if prev_v is not None and venue.get("latitude") is not None:
            dist = _haversine_km(venues.get(prev_v, {}).get("latitude"), venues.get(prev_v, {}).get("longitude"),
                                  venue.get("latitude"), venue.get("longitude"))
            if dist is not None:
                penalty = min(TRAVEL_MAX_PENALTY, TRAVEL_MAX_PENALTY * dist / TRAVEL_FULL_PENALTY_KM)
                r *= (1 - penalty)

        r *= 1 + max(-MOMENTUM_CAP, min(MOMENTUM_CAP, MOMENTUM_STEP * streak[t]))

        is_turf = venue.get("is_turf")
        if is_turf is not None:
            w, n = surface_record[t][is_turf]
            if n >= MIN_SURFACE_SAMPLE:
                league_avg = 0.45  # rough neutral prior for "win rate," matches home-win baseline elsewhere
                swing = max(-SURFACE_MAX_SWING, min(SURFACE_MAX_SWING, (w / n - league_avg)))
                r *= (1 + swing)

        if weather_available:
            date_str = game["date_time_utc"].date().isoformat()
            precip = weather.get(game["venue_id"], {}).get(date_str)
            is_rain = precip is not None and precip > 1.0  # >1mm counted as a wet match
            if is_rain:
                w, n = rain_record[t]
                if n >= MIN_RAIN_SAMPLE:
                    league_avg = 0.45
                    swing = max(-RAIN_MAX_SWING, min(RAIN_MAX_SWING, (w / n - league_avg)))
                    r *= (1 + swing)

        if is_home:
            r *= HOME_MULT
        return r

    for g in finished:
        home, away = g["home_team_id"], g["away_team_id"]
        if home not in elo or away not in elo:
            continue

        rating_home = team_rating(home, g, is_home=True)
        rating_away = team_rating(away, g, is_home=False)
        rating_diff = rating_home - rating_away

        home_win = g["home_score"] > g["away_score"]
        draw = g["home_score"] == g["away_score"]
        away_win = g["home_score"] < g["away_score"]
        results.append({"rating_diff": rating_diff, "home_win": home_win, "draw": draw, "away_win": away_win,
                         "date": g["date_time_utc"].isoformat(), "season_name": g["season_name"]})

        # ---- update running state for next time (causal — happens AFTER using it above) ----
        exp_home = 1 / (1 + 10 ** (-((elo[home] + HOME_ADV) - elo[away]) / 400))
        actual_home = 1.0 if home_win else (0.5 if draw else 0.0)
        elo[home] += K_FACTOR * (actual_home - exp_home)
        elo[away] += K_FACTOR * ((1 - actual_home) - (1 - exp_home))

        last_venue[home] = g["venue_id"]
        last_venue[away] = g["venue_id"]

        streak[home] = streak[home] + 1 if home_win else (0 if draw else -1)
        streak[away] = streak[away] + 1 if away_win else (0 if draw else -1)

        venue = venues.get(g["venue_id"], {})
        is_turf = venue.get("is_turf")
        if is_turf is not None:
            surface_record[home][is_turf][1] += 1
            surface_record[home][is_turf][0] += 1 if home_win else 0
            surface_record[away][is_turf][1] += 1
            surface_record[away][is_turf][0] += 1 if away_win else 0

        if weather_available:
            date_str = g["date_time_utc"].date().isoformat()
            precip = weather.get(g["venue_id"], {}).get(date_str)
            if precip is not None and precip > 1.0:
                rain_record[home][1] += 1
                rain_record[home][0] += 1 if home_win else 0
                rain_record[away][1] += 1
                rain_record[away][0] += 1 if away_win else 0

    return results, weather_available


def summarize(results):
    diffs = np.array([r["rating_diff"] for r in results])
    home_win = np.array([r["home_win"] for r in results], dtype=float)

    # sign accuracy (excluding games decided as a draw, since this rating has no draw class)
    decisive = [r for r in results if not r["draw"]]
    correct = sum(1 for r in decisive if (r["rating_diff"] > 0) == r["home_win"])
    sign_acc = correct / len(decisive) if decisive else None

    corr = float(np.corrcoef(diffs, home_win)[0, 1]) if len(diffs) > 1 else None

    # bucket by rating_diff decile, report actual home-win rate per bucket
    order = np.argsort(diffs)
    n_buckets = 10
    buckets = np.array_split(order, n_buckets)
    bucket_stats = []
    for b in buckets:
        if len(b) == 0:
            continue
        bucket_stats.append({
            "mean_rating_diff": float(diffs[b].mean()),
            "home_win_rate": float(home_win[b].mean()),
            "n": int(len(b)),
        })
    return {"sign_accuracy": sign_acc, "correlation": corr, "buckets": bucket_stats, "n_games": len(results)}


def _season_scope(results, seasons_back=2):
    """Return the subset of `results` (already chronological) belonging to the
    most recent `seasons_back` distinct season_name values, preserving order."""
    seen_order = []
    for r in results:
        if r["season_name"] not in seen_order:
            seen_order.append(r["season_name"])
    scope_seasons = set(seen_order[-seasons_back:])
    return [r for r in results if r["season_name"] in scope_seasons], sorted(scope_seasons)


def _three_way_accuracy(rows, draw_threshold):
    """Accuracy over 3 classes (home/draw/away). A game is called a predicted
    draw when |rating_diff| <= draw_threshold — this rating has no native draw
    output, so a 'too close to call' band around zero stands in for it."""
    if not rows:
        return None
    correct = 0
    for r in rows:
        d = r["rating_diff"]
        pred = "draw" if abs(d) <= draw_threshold else ("home" if d > 0 else "away")
        actual = "draw" if r["draw"] else ("home" if r["home_win"] else "away")
        correct += int(pred == actual)
    return correct / len(rows)


def _best_draw_threshold(train):
    """Grid-search |rating_diff| thresholds on TRAIN ONLY (never test) for the
    one that maximizes 3-way train accuracy. Candidates are the 5th-95th
    percentiles (step 5) of |rating_diff| within train, plus 0."""
    if not train:
        return 0.0
    diffs = np.array([abs(r["rating_diff"]) for r in train])
    candidates = sorted(set([0.0] + [float(np.percentile(diffs, p)) for p in range(5, 100, 5)]))
    best_thr, best_acc = 0.0, -1.0
    for thr in candidates:
        acc = _three_way_accuracy(train, thr)
        if acc > best_acc:
            best_thr, best_acc = thr, acc
    return best_thr


def train_test_report(results, seasons_back=2, train_frac=0.75):
    """75/25 chronological split, scoped to the most recent `seasons_back`
    season_name values. The draw threshold is picked on the train slice only
    (no peeking at test), then applied unchanged to both slices so the test
    number is a genuine held-out check."""
    scoped, season_list = _season_scope(results, seasons_back)
    split = int(len(scoped) * train_frac)
    train, test = scoped[:split], scoped[split:]

    draw_threshold = _best_draw_threshold(train)
    train_acc = _three_way_accuracy(train, draw_threshold)
    test_acc = _three_way_accuracy(test, draw_threshold)

    def _rate(rows, key):
        return sum(1 for r in rows if r[key]) / len(rows) if rows else None

    baseline_always_home_test = _rate(test, "home_win")
    outcomes_train = ["draw" if r["draw"] else ("home" if r["home_win"] else "away") for r in train]
    majority_class = max(set(outcomes_train), key=outcomes_train.count) if outcomes_train else None
    majority_rate_test = None
    if majority_class:
        key = {"home": "home_win", "away": "away_win", "draw": "draw"}[majority_class]
        majority_rate_test = _rate(test, key)

    return {
        "seasons_in_scope": season_list,
        "n_train": len(train), "n_test": len(test),
        "draw_threshold": draw_threshold,
        "train_accuracy_3way": train_acc,
        "test_accuracy_3way": test_acc,
        "baseline_always_home_test": baseline_always_home_test,
        "baseline_majority_class_test": {"class": majority_class, "accuracy": majority_rate_test},
    }


if __name__ == "__main__":
    sb = get_supabase()
    results, weather_available = compute_ratings_and_backtest(sb)
    summary = summarize(results)
    print(f"\nGames backtested: {summary['n_games']}  (weather data: {'yes' if weather_available else 'NO — see warning above'})")
    print(f"Sign accuracy (higher rating picks the winner, draws excluded): {summary['sign_accuracy']:.3f}")
    print(f"Correlation(rating_diff, home_win): {summary['correlation']:.3f}")
    print("\nrating_diff bucket -> actual home-win rate:")
    for b in summary["buckets"]:
        print(f"  diff~{b['mean_rating_diff']:8.1f}  ->  win rate {b['home_win_rate']:.2f}  (n={b['n']})")

    report = train_test_report(results, seasons_back=2, train_frac=0.75)
    print(f"\n--- 75/25 train/test, last 2 seasons ({', '.join(str(s) for s in report['seasons_in_scope'])}), draws included ---")
    print(f"Train games: {report['n_train']}   Test games: {report['n_test']}")
    print(f"Draw threshold (picked on train only): |rating_diff| <= {report['draw_threshold']:.1f} -> predict draw")
    print(f"Train 3-way accuracy (home/draw/away): {report['train_accuracy_3way']:.3f}")
    print(f"Test  3-way accuracy (home/draw/away): {report['test_accuracy_3way']:.3f}   <-- the real answer")
    print(f"  vs. always-pick-home baseline on test: {report['baseline_always_home_test']:.3f}")
    maj = report["baseline_majority_class_test"]
    print(f"  vs. always-pick-'{maj['class']}' (train majority) baseline on test: {maj['accuracy']:.3f}")

    import json
    with open("experimental_backtest_report.json", "w") as f:
        json.dump({"summary": summary, "train_test": report}, f, indent=2)
    print("\nWrote experimental_backtest_report.json")
