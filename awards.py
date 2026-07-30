"""
Award predictions — MLS end-of-season awards, computed from live stats.

Two tiers, and the dashboard is honest about which is which:

  SOLID  — backed by stats this database actually has: goals, assists,
  ASA's xG/xA/goals-added, standings, and a game-by-game Elo simulation of
  the rest of the season and playoffs. MVP, Golden Boot, Young Player,
  Newcomer, Best XI (attacking slots), Supporters' Shield, MLS Cup.

  PROXY  — awards this database has no real signal for (no coaches table,
  no goalkeeper/tackle/clearance stats, no injury history), so a rough
  stand-in is used instead and flagged low-confidence rather than presented
  as a real pick: Coach of the Year, Goalkeeper of the Year, Defender of the
  Year, Comeback Player of the Year, MLS Cup MVP (plus the GK/D slots inside
  Best XI). The dashboard stars these and shows why.

Five awards aren't attempted at all because nothing in this database bears
on them even loosely: Referee of the Year, Assistant Referee of the Year,
Goal of the Year, Save of the Year, Audi Goals Drive Progress Impact Award —
those are judged on officiating, highlight reels, and community work, none
of which is data this app tracks.

Runs at the end of every predict.py retrain (so on the same ~15-minute cron
as match predictions, via ingest.py) and overwrites award_predictions with a
fresh top-5 per award. Cheap to rerun from scratch every time — no state
carried between runs other than what's already in Supabase.
"""
import datetime
import json
import math
import random

from common import fetch_all

MODEL_VERSION = "awards_v1"
MIN_MINUTES = 450          # ~5 full games — keeps small-sample fluky stat lines out
YOUNG_PLAYER_MAX_AGE = 22
N_SIMULATIONS = 2000
PLAYOFF_FIELD = 8          # top-N teams overall make the simulated bracket
TOP_N = 5

# team_id -> elo used as the simulation's home-field bump. Matches predict.py's
# grid-searched value when called from there; this default only applies if
# awards.py is ever run standalone against a fresh elo dict.
DEFAULT_HOME_ADV = 65
DEFAULT_K = 20


# ============================================================
# Position bucketing — GK / D / M / F from messy source data.
# ============================================================
def _bucket_position(raw):
    """Best-effort GK / D / M / F classification.

    `players.primary_position` is a grab-bag in this database: plain ASA
    codes ('GK', 'FB', 'ST', 'W', ...), full words ('Goalkeeper',
    'Midfielder', ...), and for some rows ingested before a prior roster-diff
    fix, a raw JSON blob string dumped straight from the source API
    (e.g. '{"id":"1","name":"Goalkeeper",...}'). `player_season_xgoals.
    general_position` uses the same short-code style. Handle all of it
    rather than assuming one shape — returns None if nothing matches.
    """
    if not raw:
        return None
    text = raw
    stripped = raw.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            text = obj.get("name") or obj.get("displayName") or obj.get("abbreviation") or ""
        except (ValueError, TypeError, AttributeError):
            text = raw
    t = (text or "").strip().upper()
    if not t:
        return None
    if t in ("GK", "G") or "GOALKEEPER" in t:
        return "GK"
    if t in ("D", "FB", "CB", "WB") or "DEFEND" in t:
        return "D"
    if t in ("M", "CM", "DM", "AM") or "MIDFIELD" in t:
        return "M"
    if t in ("F", "FW", "ST", "W") or "FORWARD" in t or "STRIKER" in t or "WING" in t:
        return "F"
    return None


# ============================================================
# Shared scoring helpers
# ============================================================
def _offense_score(row):
    """Composite attacking-value score from an ASA season-xgoals row.

    Weights raw output (goals, primary assists) above the underlying
    expected-value metrics (xG, xA) and ASA's goals-added ("points_added"),
    which is closer to what MVP/Golden-Boot-style voting actually rewards —
    end product first, quality-of-chances second.
    """
    g = row.get("goals") or 0
    a = row.get("primary_assists") or 0
    pa = row.get("points_added") or 0.0
    xg = row.get("xgoals") or 0.0
    xa = row.get("xassists") or 0.0
    return 3.0 * g + 2.0 * a + 2.5 * pa + 0.5 * xg + 0.5 * xa


def _softmax_pct(scored):
    """[(entity, score), ...] -> [(entity, pct), ...] sorted by pct desc.

    Percentages are shares of a softmax over the WHOLE qualifying pool (not
    just the top 5 we end up displaying), so a narrow leader shows a modest
    lead instead of manufactured near-100% confidence. Temperature adapts to
    the spread of scores in each pool so this works across wildly different
    scales (goals counts vs. simulation win-fractions) without per-award
    tuning.
    """
    if not scored:
        return []
    scores = [s for _, s in scored]
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(variance)
    temperature = max(std, 0.5)
    top = max(scores)
    exps = [math.exp((s - top) / temperature) for s in scores]
    total = sum(exps) or 1.0
    pcts = [100.0 * e / total for e in exps]
    return sorted(zip([e for e, _ in scored], pcts), key=lambda p: -p[1])


def _age(birth_date, as_of):
    if not birth_date:
        return None
    try:
        b = datetime.date.fromisoformat(str(birth_date)[:10])
    except ValueError:
        return None
    return as_of.year - b.year - ((as_of.month, as_of.day) < (b.month, b.day))


def _current_season(finished, upcoming):
    """Whatever season upcoming fixtures belong to; falls back to the most
    recently finished game's season during the off-season (no fixtures yet)."""
    upcoming_seasons = [g["season_name"] for g in upcoming if g.get("season_name")]
    if upcoming_seasons:
        return max(set(upcoming_seasons), key=upcoming_seasons.count)
    finished_with_season = [g for g in finished if g.get("season_name")]
    if finished_with_season:
        return sorted(finished_with_season, key=lambda g: g["date_time_utc"])[-1]["season_name"]
    return None


# ============================================================
# Standings / season simulation (Elo-based, not the full match classifier —
# fast enough to run thousands of trials in a background cron job; the
# per-game predictions elsewhere in the app still use the full model).
# ============================================================
def _standings(finished, team_ids):
    pts = {t: 0 for t in team_ids}
    played = {t: 0 for t in team_ids}
    for g in finished:
        h, a, hs, aw = g["home_team_id"], g["away_team_id"], g["home_score"], g["away_score"]
        if h not in pts or a not in pts or hs is None or aw is None:
            continue
        played[h] += 1
        played[a] += 1
        if hs > aw:
            pts[h] += 3
        elif hs < aw:
            pts[a] += 3
        else:
            pts[h] += 1
            pts[a] += 1
    return pts, played


def _league_draw_rate(finished):
    scored = [g for g in finished if g["home_score"] is not None and g["away_score"] is not None]
    if not scored:
        return 0.24  # rough long-run MLS draw rate, used only if there's no history yet
    draws = sum(1 for g in scored if g["home_score"] == g["away_score"])
    return draws / len(scored)


def _knockout_round(seeds, sim_elo, home_adv):
    """Pair 1v8, 2v7, ... (standard bracket), higher seed hosts, no draws."""
    winners = []
    n = len(seeds)
    for i in range(n // 2):
        higher, lower = seeds[i], seeds[n - 1 - i]
        exp_higher = 1 / (1 + 10 ** (-((sim_elo[higher] + home_adv) - sim_elo[lower]) / 400))
        winners.append(higher if random.random() < exp_higher else lower)
    return winners


def simulate_season_and_playoffs(finished, upcoming, team_ids, elo, home_adv,
                                  n_sims=N_SIMULATIONS, playoff_field=PLAYOFF_FIELD):
    """Monte Carlo the rest of the regular season + a simplified playoff bracket.

    Each trial: start from real current standings and real current Elo,
    resolve every remaining REGULAR-SEASON game (knockout_game = false) with
    an Elo win probability (home Elo + home-field edge vs away Elo, league
    draw rate carved out of the remainder), updating Elo as results land so
    later games reflect earlier ones within the same simulated world. The
    Shield goes to whoever leads points at the end. The top `playoff_field`
    teams by points then run a single-elimination bracket seeded by that
    same finish, again on Elo, to produce an MLS Cup winner.

    This is a deliberate simplification, not a re-run of the full featured
    match model (which stays in charge of the individual game predictions
    shown elsewhere) — and MLS's real playoff format/qualification rules
    (conference splits, wild cards) aren't reproduced since this database
    doesn't track conference assignment. Treat the Cup number as
    directional, not exact.

    Returns (shield_pct: {team_id: pct}, cup_pct: {team_id: pct}).
    """
    regular = [g for g in upcoming if not g.get("knockout_game")]
    base_pts, _ = _standings(finished, team_ids)
    draw_rate = _league_draw_rate(finished)

    shield_wins = {t: 0 for t in team_ids}
    cup_wins = {t: 0 for t in team_ids}

    for _ in range(n_sims):
        pts = dict(base_pts)
        sim_elo = dict(elo)
        for g in regular:
            h, a = g["home_team_id"], g["away_team_id"]
            if h not in sim_elo or a not in sim_elo:
                continue
            exp_home = 1 / (1 + 10 ** (-((sim_elo[h] + home_adv) - sim_elo[a]) / 400))
            p_home = exp_home * (1 - draw_rate)
            p_away = (1 - exp_home) * (1 - draw_rate)
            r = random.random()
            if r < p_home:
                pts[h] += 3
                outcome = 1.0
            elif r < p_home + p_away:
                pts[a] += 3
                outcome = 0.0
            else:
                pts[h] += 1
                pts[a] += 1
                outcome = 0.5
            sim_elo[h] += DEFAULT_K * (outcome - exp_home)
            sim_elo[a] += DEFAULT_K * ((1 - outcome) - (1 - exp_home))

        ranked = sorted(team_ids, key=lambda t: -pts[t])
        shield_wins[ranked[0]] += 1

        field = ranked[:playoff_field]
        if len(field) >= 2:
            seeds = field
            while len(seeds) > 1:
                seeds = _knockout_round(seeds, sim_elo, home_adv)
            cup_wins[seeds[0]] += 1

    shield_pct = {t: 100.0 * shield_wins[t] / n_sims for t in team_ids}
    cup_pct = {t: 100.0 * cup_wins[t] / n_sims for t in team_ids}
    return shield_pct, cup_pct


# ============================================================
# Main entry point — called from predict.py right after it updates Elo/
# attack/defense ratings, so awards always reflect the same fresh retrain.
# ============================================================
def compute_and_write_awards(supabase, *, team_ids, finished, upcoming, elo, home_adv, attack, defense):
    current_season = _current_season(finished, upcoming)
    if not current_season:
        print("Awards: couldn't determine a current season — skipping.")
        return
    if len(finished) < 10:
        print("Awards: not enough finished games yet — skipping.")
        return

    print(f"Computing award predictions for season {current_season}...")
    today = datetime.date.today()

    teams_by_id = {t["id"]: t for t in fetch_all(supabase, "teams", "id,name")}
    players = fetch_all(supabase, "players", "id,name,birth_date,primary_position,current_team_id")
    players_by_id = {p["id"]: p for p in players}

    xg_rows = fetch_all(
        supabase, "player_season_xgoals",
        "player_id,team_id,season_name,general_position,minutes_played,goals,"
        "xgoals,primary_assists,xassists,points_added",
    )
    this_season = [r for r in xg_rows if r.get("season_name") == current_season]
    by_player_season = {(r["player_id"], r["season_name"]): r for r in xg_rows}

    history = fetch_all(supabase, "player_team_history", "player_id,season_name")
    first_season = {}
    for h in history:
        sn = h.get("season_name")
        if sn is None or not str(sn).isdigit():
            continue
        pid = h["player_id"]
        if pid not in first_season or int(sn) < int(first_season[pid]):
            first_season[pid] = sn

    def team_name(tid):
        return (teams_by_id.get(tid) or {}).get("name", "Unknown team")

    def position_of(pid, row):
        p = players_by_id.get(pid) or {}
        return _bucket_position(p.get("primary_position")) or _bucket_position(row.get("general_position"))

    rows_out = []

    def emit(award_key, award_name, ranked, entity_type, id_fn, name_fn, subtitle_fn,
              is_proxy=False, proxy_note=None):
        for i, (entity, pct) in enumerate(ranked[:TOP_N]):
            rows_out.append({
                "award_key": award_key,
                "award_name": award_name,
                "season_name": current_season,
                "model_version": MODEL_VERSION,
                "rank": i + 1,
                "entity_type": entity_type,
                "entity_id": id_fn(entity),
                "entity_name": name_fn(entity),
                "subtitle": subtitle_fn(entity),
                "win_pct": round(pct, 1),
                "is_proxy": is_proxy,
                "proxy_note": proxy_note,
            })

    # ---- qualifying pool for the individual-performance awards ----
    qualified = [r for r in this_season if (r.get("minutes_played") or 0) >= MIN_MINUTES
                 and r["player_id"] in players_by_id]
    if len(qualified) < 3:  # too small a sample (very early season) — use everyone with any minutes
        qualified = [r for r in this_season if (r.get("minutes_played") or 0) > 0
                     and r["player_id"] in players_by_id]

    season_pts, season_played = _standings(
        [g for g in finished if g.get("season_name") == current_season], team_ids
    )

    def player_subtitle(row):
        g = row.get("goals") or 0
        a = row.get("primary_assists") or 0
        return f"{team_name(row['team_id'])} — {g}G {a}A"

    # ---- Landon Donovan MLS MVP ----
    mvp_scored = [
        (r, _offense_score(r) + 0.15 * season_pts.get(r["team_id"], 0)) for r in qualified
    ]
    emit("mvp", "Landon Donovan MLS MVP", _softmax_pct(mvp_scored), "player",
         lambda r: r["player_id"], lambda r: players_by_id[r["player_id"]]["name"], player_subtitle)

    # ---- MLS Golden Boot ----
    boot_scored = [(r, (r.get("goals") or 0) + 0.001 * (r.get("goals") or 0) / max(r.get("minutes_played") or 1, 1))
                   for r in qualified if (r.get("goals") or 0) > 0]
    emit("golden_boot", "MLS Golden Boot", _softmax_pct(boot_scored), "player",
         lambda r: r["player_id"], lambda r: players_by_id[r["player_id"]]["name"], player_subtitle)

    # ---- Young Player of the Year ----
    young_scored = []
    for r in qualified:
        age = _age(players_by_id[r["player_id"]].get("birth_date"), today)
        if age is not None and age <= YOUNG_PLAYER_MAX_AGE:
            young_scored.append((r, _offense_score(r)))
    emit("young_player", "Young Player of the Year", _softmax_pct(young_scored), "player",
         lambda r: r["player_id"], lambda r: players_by_id[r["player_id"]]["name"], player_subtitle)

    # ---- Newcomer of the Year ----
    newcomer_scored = [
        (r, _offense_score(r)) for r in qualified
        if first_season.get(r["player_id"]) == current_season
    ]
    emit("newcomer", "Newcomer of the Year", _softmax_pct(newcomer_scored), "player",
         lambda r: r["player_id"], lambda r: players_by_id[r["player_id"]]["name"], player_subtitle)

    # ---- Comeback Player of the Year (PROXY: season-over-season jump, not
    #      injury-aware — we have no injury data, so this really just finds
    #      "most improved," which overlaps with but isn't the same thing) ----
    prev_season = str(int(current_season) - 1) if str(current_season).isdigit() else None
    comeback_scored = []
    if prev_season:
        for r in qualified:
            prev = by_player_season.get((r["player_id"], prev_season))
            if not prev or (prev.get("minutes_played") or 0) < MIN_MINUTES:
                continue
            this_rate = _offense_score(r) / max(r.get("minutes_played") or 1, 1) * 900
            prev_rate = _offense_score(prev) / max(prev.get("minutes_played") or 1, 1) * 900
            comeback_scored.append((r, this_rate - prev_rate))
    emit("comeback_player", "Comeback Player of the Year", _softmax_pct(comeback_scored), "player",
         lambda r: r["player_id"], lambda r: players_by_id[r["player_id"]]["name"], player_subtitle,
         is_proxy=True,
         proxy_note="No injury data in this database — this is really \"most statistically improved "
                     "year-over-year,\" which isn't the same thing as a real comeback story. Don't trust it.")

    # ---- Defender / Goalkeeper of the Year (PROXY: no tackles/clearances/
    #      saves data at all — stands in with "plays a lot for a stingy
    #      defense," which rewards the team more than the individual) ----
    def defense_proxy_pool(bucket):
        pool = []
        for r in this_season:
            if (r.get("minutes_played") or 0) < MIN_MINUTES:
                continue
            if position_of(r["player_id"], r) != bucket:
                continue
            d = defense.get(r["team_id"])
            if d is None:
                continue
            score = -d + 0.0003 * (r.get("minutes_played") or 0)
            pool.append((r, score))
        return pool

    emit("defender", "Defender of the Year", _softmax_pct(defense_proxy_pool("D")), "player",
         lambda r: r["player_id"], lambda r: players_by_id[r["player_id"]]["name"],
         lambda r: f"{team_name(r['team_id'])} — {defense.get(r['team_id'], 0):.2f} goals conceded/game",
         is_proxy=True,
         proxy_note="No tackles/clearances/interceptions data in this database — this just ranks players "
                     "who log heavy minutes for the team that concedes the least. It rewards the team's "
                     "defense more than the individual. Don't trust it.")

    emit("goalkeeper", "Goalkeeper of the Year", _softmax_pct(defense_proxy_pool("GK")), "player",
         lambda r: r["player_id"], lambda r: players_by_id[r["player_id"]]["name"],
         lambda r: f"{team_name(r['team_id'])} — {defense.get(r['team_id'], 0):.2f} goals conceded/game",
         is_proxy=True,
         proxy_note="No saves/goals-against-per-keeper data in this database — this just picks the "
                     "starting goalkeeper for the team that concedes the least. Don't trust it.")

    # ---- MLS Best XI (attacking slots are the solid MVP-style score;
    #      GK/D slots reuse the same low-confidence proxy as above) ----
    mids = _softmax_pct([(r, _offense_score(r)) for r in qualified if position_of(r["player_id"], r) == "M"])
    fwds = _softmax_pct([(r, _offense_score(r)) for r in qualified if position_of(r["player_id"], r) == "F"])
    emit("best_xi_gk", "MLS Best XI — Goalkeeper", _softmax_pct(defense_proxy_pool("GK")), "player",
         lambda r: r["player_id"], lambda r: players_by_id[r["player_id"]]["name"], player_subtitle,
         is_proxy=True, proxy_note="Same no-goalkeeper-stats caveat as Goalkeeper of the Year. Don't trust it.")
    emit("best_xi_d", "MLS Best XI — Defenders", _softmax_pct(defense_proxy_pool("D")), "player",
         lambda r: r["player_id"], lambda r: players_by_id[r["player_id"]]["name"], player_subtitle,
         is_proxy=True, proxy_note="Same no-defensive-stats caveat as Defender of the Year. Don't trust it.")
    emit("best_xi_m", "MLS Best XI — Midfielders", mids, "player",
         lambda r: r["player_id"], lambda r: players_by_id[r["player_id"]]["name"], player_subtitle)
    emit("best_xi_f", "MLS Best XI — Forwards", fwds, "player",
         lambda r: r["player_id"], lambda r: players_by_id[r["player_id"]]["name"], player_subtitle)

    # ---- Supporters' Shield + MLS Cup (Elo Monte Carlo) ----
    shield_pct, cup_pct = simulate_season_and_playoffs(finished, upcoming, team_ids, elo, home_adv)
    shield_ranked = sorted(((t, p) for t, p in shield_pct.items()), key=lambda x: -x[1])
    cup_ranked = sorted(((t, p) for t, p in cup_pct.items()), key=lambda x: -x[1])

    def team_subtitle(tid):
        return f"{season_pts.get(tid, 0)} pts, Elo {elo.get(tid, 1500):.0f}"

    emit("supporters_shield", "MLS Supporters' Shield", shield_ranked, "team",
         lambda tid: tid, lambda tid: team_name(tid), team_subtitle)
    emit("mls_cup", "MLS Cup Trophy", cup_ranked, "team",
         lambda tid: tid, lambda tid: team_name(tid), team_subtitle)

    # ---- MLS Cup MVP (PROXY: depends entirely on a bracket that hasn't
    #      happened yet — best offensive player on each of the top simulated
    #      contenders, weighted by that team's Cup odds) ----
    cup_mvp_scored = []
    top_contenders = [t for t, _ in cup_ranked[:3] if cup_pct[t] > 0]
    for tid in top_contenders:
        roster = [r for r in qualified if r["team_id"] == tid]
        if not roster:
            continue
        best = max(roster, key=_offense_score)
        cup_mvp_scored.append((best, _offense_score(best) * cup_pct[tid]))
    emit("mls_cup_mvp", "MLS Cup MVP", _softmax_pct(cup_mvp_scored), "player",
         lambda r: r["player_id"], lambda r: players_by_id[r["player_id"]]["name"], player_subtitle,
         is_proxy=True,
         proxy_note="The final hasn't been set — this guesses the best attacker on whichever teams the "
                     "simulation currently likes to win it all. Extremely speculative. Don't trust it.")

    # ---- Sigi Schmid Coach of the Year (PROXY: no coaches table at all —
    #      this names a TEAM, not a coach) ----
    coach_scored = []
    for t in team_ids:
        played = season_played.get(t, 0)
        if played < 3:
            continue
        actual_ppg = season_pts.get(t, 0) / played
        exp_share = 1 / (1 + 10 ** (-(elo.get(t, 1500) - 1500) / 400))
        expected_ppg = 3 * exp_share
        coach_scored.append((t, actual_ppg - expected_ppg))
    emit("coach_of_the_year", "Sigi Schmid Coach of the Year", _softmax_pct(coach_scored), "team",
         lambda tid: tid, lambda tid: team_name(tid),
         lambda tid: f"Outperforming Elo-implied pace by {dict(coach_scored).get(tid, 0):+.2f} pts/game",
         is_proxy=True,
         proxy_note="No coaches table in this database — this names the TEAM overperforming its "
                     "Elo-implied points rate, not an actual coach. Don't trust it as a real pick.")

    supabase.table("award_predictions").delete().neq("award_key", "__never__").execute()
    if rows_out:
        for i in range(0, len(rows_out), 200):
            supabase.table("award_predictions").insert(rows_out[i:i + 200]).execute()
    print(f"Wrote {len(rows_out)} award-prediction rows across "
          f"{len(set(r['award_key'] for r in rows_out))} awards.")
