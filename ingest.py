"""
Live ingestion job — meant to run on a schedule (see .github/workflows/ingest.yml).

Each run:
  1. Pulls ESPN's scoreboard for yesterday/today/tomorrow and upserts game status,
     scores, and venue info. When a game just went final, pulls the box score and
     writes team/player game stats, which is what the dashboard reacts to live.
  2. Diffs each team's current ESPN roster against what we have stored, so a trade
     gets picked up automatically (closes the old player_team_history stint, opens
     a new one) instead of needing manual updates.

Usage:
    export SUPABASE_URL="https://xxxx.supabase.co"
    export SUPABASE_KEY="your-service-role-key"
    python ingest.py
"""
import datetime
from common import get_supabase, espn_get, id_map
from predict import train_and_predict

STATUS_MAP = {"pre": "scheduled", "in": "live", "post": "final"}


def upsert_venue(supabase, venue_cache, espn_venue: dict):
    """Return our internal venue uuid for an ESPN venue object, creating/matching as needed."""
    if not espn_venue or not espn_venue.get("id"):
        return None
    espn_id = str(espn_venue["id"])
    if espn_id in venue_cache:
        return venue_cache[espn_id]

    # Try to match an existing (ASA-sourced) venue by name before creating a new row.
    name = espn_venue.get("fullName")
    existing = supabase.table("venues").select("id,name").is_("espn_venue_id", "null").execute().data
    match = next((v for v in existing if v["name"] and name and v["name"].strip().lower() == name.strip().lower()), None)

    if match:
        supabase.table("venues").update({"espn_venue_id": espn_id}).eq("id", match["id"]).execute()
        venue_cache[espn_id] = match["id"]
        return match["id"]

    address = espn_venue.get("address", {}) or {}
    inserted = supabase.table("venues").insert({
        "espn_venue_id": espn_id,
        "name": name,
        "city": address.get("city"),
        "province": address.get("state"),
        "country": address.get("country"),
    }).execute().data
    new_id = inserted[0]["id"]
    venue_cache[espn_id] = new_id
    return new_id


def find_or_create_game(supabase, espn_event_id, home_id, away_id, date_iso, venue_id, existing_by_espn, existing_by_teams_date):
    if espn_event_id in existing_by_espn:
        return existing_by_espn[espn_event_id], False

    # Fall back to matching a backfilled (ASA-sourced) row with no espn_event_id yet,
    # by same home/away teams and a date within an 18-hour window (kickoff times can
    # shift slightly between providers / TV reschedules).
    game_date = datetime.datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
    key = (home_id, away_id)
    for candidate in existing_by_teams_date.get(key, []):
        cand_date = candidate["date_time_utc"]
        if abs((cand_date - game_date).total_seconds()) < 18 * 3600:
            supabase.table("games").update({"espn_event_id": espn_event_id}).eq("id", candidate["id"]).execute()
            return candidate["id"], False

    inserted = supabase.table("games").insert({
        "espn_event_id": espn_event_id,
        "date_time_utc": date_iso,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "venue_id": venue_id,
        "status": "scheduled",
    }).execute().data
    return inserted[0]["id"], True


def parse_boxscore(summary_json):
    """
    Best-effort parse of ESPN's boxscore. ESPN's exact response shape can vary by
    sport/game, so this tries the standard `boxscore.players[].statistics[].athletes[]`
    shape first (used across most ESPN sports) and falls back to team-level stats only
    if that isn't present. If your games consistently come back with no player rows,
    print(summary_json) for one finished game and adjust this function to match what
    you actually get back — the team-level stats will still populate either way.
    """
    team_stats = []
    player_stats = []
    box = summary_json.get("boxscore", {})

    for team_block in box.get("teams", []):
        stats = {s.get("name"): s.get("displayValue") for s in team_block.get("statistics", [])}
        team_stats.append({
            "espn_team_id": str(team_block.get("team", {}).get("id")),
            "possession_pct": _to_num(stats.get("possessionPct")),
            "shots": _to_num(stats.get("totalShots")),
            "shots_on_target": _to_num(stats.get("shotsOnTarget")),
            "fouls": _to_num(stats.get("foulsCommitted")),
            "corners": _to_num(stats.get("wonCorners")),
        })

    for player_block in box.get("players", []):
        team_espn_id = str(player_block.get("team", {}).get("id"))
        for stat_group in player_block.get("statistics", []):
            labels = stat_group.get("labels", [])
            for athlete_entry in stat_group.get("athletes", []):
                athlete = athlete_entry.get("athlete", {})
                values = athlete_entry.get("stats", [])
                stat_dict = dict(zip(labels, values))
                player_stats.append({
                    "espn_player_id": str(athlete.get("id")),
                    "name": athlete.get("displayName"),
                    "espn_team_id": team_espn_id,
                    "stats": stat_dict,
                })

    return team_stats, player_stats


def _to_num(val):
    if val is None:
        return None
    try:
        f = float(str(val).replace("%", ""))
        # Postgres integer columns (home_score, shots, etc.) reject "0.0" — hand
        # back a plain int for whole numbers so upserts don't fail.
        return int(f) if f.is_integer() else f
    except ValueError:
        return None


def sync_scoreboard(supabase):
    print("Syncing scoreboard...")
    team_ids = id_map(supabase, "teams", "espn_team_id")
    venue_cache = id_map(supabase, "venues", "espn_venue_id")

    existing_games = supabase.table("games").select(
        "id,espn_event_id,home_team_id,away_team_id,date_time_utc,status"
    ).execute().data
    for g in existing_games:
        if g["date_time_utc"]:
            g["date_time_utc"] = datetime.datetime.fromisoformat(g["date_time_utc"].replace("Z", "+00:00"))
    existing_by_espn = {g["espn_event_id"]: g for g in existing_games if g["espn_event_id"]}
    existing_by_teams_date = {}
    for g in existing_games:
        if not g["espn_event_id"]:
            existing_by_teams_date.setdefault((g["home_team_id"], g["away_team_id"]), []).append(g)

    today = datetime.date.today()
    # -1..+14 days: ASA (the historical source) doesn't expose future fixtures, so this
    # scoreboard scan is the only source of upcoming/scheduled games for the model to predict.
    dates = [(today + datetime.timedelta(days=d)).strftime("%Y%m%d") for d in range(-1, 15)]

    for date_str in dates:
        data = espn_get("scoreboard", params={"dates": date_str})
        for event in data.get("events", []):
            comp = event["competitions"][0]
            state = comp["status"]["type"]["state"]
            new_status = STATUS_MAP.get(state, "scheduled")

            home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
            away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
            home_id = team_ids.get(str(home["team"]["id"]))
            away_id = team_ids.get(str(away["team"]["id"]))
            if not home_id or not away_id:
                print(f"  Skipping event {event['id']}: team not found in DB "
                      f"(espn ids {home['team']['id']}/{away['team']['id']}) — check teams table.")
                continue

            venue_id = upsert_venue(supabase, venue_cache, comp.get("venue"))

            game_row = existing_by_espn.get(event["id"])
            if game_row is None:
                game_id, _ = find_or_create_game(
                    supabase, event["id"], home_id, away_id, comp["date"], venue_id,
                    existing_by_espn, existing_by_teams_date,
                )
                was_final_already = False
            else:
                game_id = game_row["id"]
                was_final_already = game_row["status"] == "final"

            supabase.table("games").update({
                "status": new_status,
                "home_score": _to_num(home.get("score")),
                "away_score": _to_num(away.get("score")),
                "venue_id": venue_id,
                "last_updated_utc": datetime.datetime.utcnow().isoformat(),
            }).eq("id", game_id).execute()

            if new_status == "final" and not was_final_already:
                print(f"  Game just went final: {home['team']['displayName']} vs "
                      f"{away['team']['displayName']} — pulling box score...")
                write_final_game_stats(supabase, game_id, event["id"], team_ids)


def write_final_game_stats(supabase, game_id, espn_event_id, team_ids_by_espn):
    try:
        summary = espn_get("summary", params={"event": espn_event_id})
    except Exception as e:  # noqa: BLE001
        print(f"    Couldn't fetch summary for event {espn_event_id}: {e}")
        return

    team_stats, player_stats = parse_boxscore(summary)

    for ts in team_stats:
        team_id = team_ids_by_espn.get(ts["espn_team_id"])
        if not team_id:
            continue
        supabase.table("team_game_stats").upsert({
            "game_id": game_id,
            "team_id": team_id,
            "possession_pct": ts["possession_pct"],
            "shots": ts["shots"],
            "shots_on_target": ts["shots_on_target"],
            "fouls": ts["fouls"],
            "corners": ts["corners"],
        }, on_conflict="game_id,team_id").execute()

    if not player_stats:
        print("    No player-level box score rows found for this game — "
              "team stats were still saved. See parse_boxscore() docstring in ingest.py.")
        return

    player_espn_ids = id_map(supabase, "players", "espn_player_id")
    for ps in player_stats:
        player_id = player_espn_ids.get(ps["espn_player_id"])
        team_id = team_ids_by_espn.get(ps["espn_team_id"])
        if not player_id:
            # New/unmatched player — create a minimal row keyed by espn id so the
            # stat line isn't lost; backfill.py or sync_rosters() can enrich it later.
            inserted = supabase.table("players").insert({
                "espn_player_id": ps["espn_player_id"],
                "name": ps["name"],
                "current_team_id": team_id,
            }).execute().data
            player_id = inserted[0]["id"]
        if not team_id:
            continue
        s = ps["stats"]
        supabase.table("player_game_stats").upsert({
            "game_id": game_id,
            "player_id": player_id,
            "team_id": team_id,
            "goals": _to_num(s.get("goals")),
            "assists": _to_num(s.get("assists") or s.get("primaryAssists")),
            "shots": _to_num(s.get("totalShots")),
            "shots_on_target": _to_num(s.get("shotsOnTarget")),
            "yellow_cards": _to_num(s.get("yellowCards")),
            "red_cards": _to_num(s.get("redCards")),
            "fouls_committed": _to_num(s.get("foulsCommitted")),
        }, on_conflict="game_id,player_id").execute()
    print(f"    Wrote {len(team_stats)} team stat rows and {len(player_stats)} player stat rows.")


def sync_rosters(supabase):
    """Detect trades: diff each team's live ESPN roster against our stored history."""
    print("Syncing rosters (trade detection)...")
    teams = supabase.table("teams").select("id,espn_team_id,name").execute().data
    players_by_espn = id_map(supabase, "players", "espn_player_id")
    all_players = supabase.table("players").select("id,name,espn_player_id").execute().data
    players_by_name = {p["name"].strip().lower(): p["id"] for p in all_players if p.get("name")}

    open_stints = supabase.table("player_team_history").select(
        "id,player_id,team_id"
    ).is_("end_date", "null").execute().data
    open_by_player = {s["player_id"]: s for s in open_stints}

    today_str = datetime.date.today().isoformat()

    for team in teams:
        if not team["espn_team_id"]:
            continue
        try:
            roster = espn_get(f"teams/{team['espn_team_id']}/roster")
        except Exception as e:  # noqa: BLE001
            print(f"  Couldn't fetch roster for {team['name']}: {e}")
            continue

        for athlete in roster.get("athletes", []):
            espn_pid = str(athlete["id"])
            player_id = players_by_espn.get(espn_pid)

            if not player_id:
                # Try matching to an ASA-sourced player row by name before creating a new one.
                player_id = players_by_name.get(athlete.get("fullName", "").strip().lower())
                if player_id:
                    supabase.table("players").update({"espn_player_id": espn_pid}).eq("id", player_id).execute()
                else:
                    inserted = supabase.table("players").insert({
                        "espn_player_id": espn_pid,
                        "name": athlete.get("fullName"),
                        "primary_position": athlete.get("position"),
                        "current_team_id": team["id"],
                    }).execute().data
                    player_id = inserted[0]["id"]
                players_by_espn[espn_pid] = player_id

            current_stint = open_by_player.get(player_id)
            if current_stint is None:
                # No open stint at all — brand new to the league or missed by backfill.
                supabase.table("player_team_history").insert({
                    "player_id": player_id, "team_id": team["id"],
                    "start_date": today_str, "source": "roster_diff",
                }).execute()
                supabase.table("players").update({"current_team_id": team["id"]}).eq("id", player_id).execute()
            elif current_stint["team_id"] != team["id"]:
                # Player's open stint is on a DIFFERENT team than this roster — a trade.
                print(f"  Trade detected: {athlete.get('fullName')} -> {team['name']}")
                supabase.table("player_team_history").update({"end_date": today_str}) \
                    .eq("id", current_stint["id"]).execute()
                supabase.table("player_team_history").insert({
                    "player_id": player_id, "team_id": team["id"],
                    "start_date": today_str, "source": "roster_diff",
                }).execute()
                supabase.table("players").update({"current_team_id": team["id"]}).eq("id", player_id).execute()
            # else: already correctly on this team, nothing to do.


if __name__ == "__main__":
    sb = get_supabase()
    sync_scoreboard(sb)
    sync_rosters(sb)
    print("Ingestion run complete.")
    print("Retraining model on updated data...")
    train_and_predict(sb)
