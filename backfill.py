"""
One-time historical backfill: pulls all MLS teams, venues, players, the last
3 seasons of games, and season-level advanced player stats, and loads them
into Supabase.

Run this ONCE before the live ingestion job starts (see README.md). Safe to
re-run — every write is an upsert keyed on the source's own ID, so running it
twice won't create duplicates (player_team_history rows are the one
exception; see the note near the bottom).

Usage:
    export SUPABASE_URL="https://xxxx.supabase.co"
    export SUPABASE_KEY="your-service-role-key"
    python backfill.py
"""
import datetime
from common import get_supabase, asa_get, espn_get, id_map, chunked

CURRENT_YEAR = datetime.date.today().year
SEASONS = [str(CURRENT_YEAR - 2), str(CURRENT_YEAR - 1), str(CURRENT_YEAR)]


def backfill_teams(supabase):
    print("Fetching teams from ASA + ESPN...")
    asa_teams = asa_get("teams")
    espn_teams = espn_get("teams")
    # ESPN's team list is nested under sports[0].leagues[0].teams[].team
    espn_list = []
    try:
        for entry in espn_teams["sports"][0]["leagues"][0]["teams"]:
            espn_list.append(entry["team"])
    except (KeyError, IndexError):
        print("  Warning: couldn't parse ESPN teams response, continuing with ASA only.")

    # Match ASA <-> ESPN by abbreviation (both use short codes like ATL, ATX, MTL...)
    espn_by_abbr = {t.get("abbreviation", "").upper(): t for t in espn_list}

    records = []
    for t in asa_teams:
        abbr = (t.get("team_abbreviation") or "").upper()
        espn_match = espn_by_abbr.get(abbr)
        records.append({
            "asa_team_id": t["team_id"],
            "espn_team_id": espn_match["id"] if espn_match else None,
            "name": t.get("team_name"),
            "short_name": t.get("team_short_name"),
            "abbreviation": t.get("team_abbreviation"),
            "logo_url": (espn_match.get("logos", [{}])[0].get("href")
                         if espn_match and espn_match.get("logos") else None),
        })

    for batch in chunked(records):
        supabase.table("teams").upsert(batch, on_conflict="asa_team_id").execute()
    print(f"  Upserted {len(records)} teams "
          f"({sum(1 for r in records if r['espn_team_id'])} matched to an ESPN id).")


def backfill_venues(supabase):
    print("Fetching stadia from ASA...")
    stadia = asa_get("stadia")
    records = []
    for s in stadia:
        records.append({
            "asa_stadium_id": s["stadium_id"],
            "name": s.get("stadium_name"),
            "city": s.get("city"),
            "province": s.get("province"),
            "country": s.get("country"),
            "latitude": s.get("latitude"),
            "longitude": s.get("longitude"),
            "capacity": s.get("capacity"),
            "is_turf": s.get("turf"),
            "has_roof": s.get("roof"),
            "year_built": s.get("year_built"),
        })
    for batch in chunked(records):
        supabase.table("venues").upsert(batch, on_conflict="asa_stadium_id").execute()
    print(f"  Upserted {len(records)} venues.")
    print("  Note: ESPN venue IDs get filled in automatically as the live ingestion")
    print("  job encounters games at each venue (see ingest.py).")


def backfill_players(supabase):
    print("Fetching player biographical data from ASA...")
    players = asa_get("players")
    records = []
    for p in players:
        height_in = None
        if p.get("height_ft") is not None and p.get("height_in") is not None:
            height_in = p["height_ft"] * 12 + p["height_in"]
        records.append({
            "asa_player_id": p["player_id"],
            "name": p.get("player_name"),
            "birth_date": p.get("birth_date") or None,
            "height_in": height_in,
            "weight_lb": p.get("weight_lb"),
            "nationality": p.get("nationality"),
            "primary_position": p.get("primary_general_position"),
        })
    for batch in chunked(records):
        supabase.table("players").upsert(batch, on_conflict="asa_player_id").execute()
    print(f"  Upserted {len(records)} players.")


def backfill_games_and_season_stats(supabase):
    team_ids = id_map(supabase, "teams", "asa_team_id")
    venue_ids = id_map(supabase, "venues", "asa_stadium_id")
    player_ids = id_map(supabase, "players", "asa_player_id")

    for season in SEASONS:
        print(f"Fetching {season} games from ASA...")
        games = asa_get("games", params={"season_name": season})
        game_records = []
        for g in games:
            home_id = team_ids.get(g.get("home_team_id"))
            away_id = team_ids.get(g.get("away_team_id"))
            if not home_id or not away_id:
                continue  # skip games for teams outside our loaded team list (rare)
            status = "final" if g.get("status") == "FullTime" else "scheduled"
            game_records.append({
                "asa_game_id": g["game_id"],
                "date_time_utc": g.get("date_time_utc"),
                "season_name": g.get("season_name"),
                "matchday": g.get("matchday"),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "venue_id": venue_ids.get(g.get("stadium_id")),
                "home_score": g.get("home_score"),
                "away_score": g.get("away_score"),
                "status": status,
                "attendance": g.get("attendance"),
                "knockout_game": g.get("knockout_game", False),
                "last_updated_utc": g.get("last_updated_utc"),
            })
        for batch in chunked(game_records):
            supabase.table("games").upsert(batch, on_conflict="asa_game_id").execute()
        print(f"  Upserted {len(game_records)} games for {season}.")

        print(f"Fetching {season} player season xgoals from ASA...")
        xg = asa_get("players/xgoals", params={"season_name": season, "minimum_minutes": 1})
        xg_records = []
        history_records = []
        for row in xg:
            p_id = player_ids.get(row.get("player_id"))
            t_id = team_ids.get(row.get("team_id"))
            if not p_id:
                continue
            xg_records.append({
                "player_id": p_id,
                "team_id": t_id,
                "season_name": season,
                "general_position": row.get("general_position"),
                "minutes_played": row.get("minutes_played"),
                "shots": row.get("shots"),
                "shots_on_target": row.get("shots_on_target"),
                "goals": row.get("goals"),
                "xgoals": row.get("xgoals"),
                "key_passes": row.get("key_passes"),
                "primary_assists": row.get("primary_assists"),
                "xassists": row.get("xassists"),
                "points_added": row.get("points_added"),
                "xpoints_added": row.get("xpoints_added"),
            })
            if t_id:
                history_records.append({
                    "player_id": p_id,
                    "team_id": t_id,
                    "season_name": season,
                    "source": "backfill",
                })
        for batch in chunked(xg_records):
            supabase.table("player_season_xgoals").upsert(
                batch, on_conflict="player_id,team_id,season_name"
            ).execute()
        print(f"  Upserted {len(xg_records)} player-season xgoals rows for {season}.")

        # player_team_history has no unique constraint (a player can have multiple
        # stints), so guard against re-running this script by only inserting rows
        # that don't already exist for this player+team+season+source combo.
        existing = supabase.table("player_team_history") \
            .select("player_id,team_id,season_name") \
            .eq("season_name", season).eq("source", "backfill").execute().data
        existing_keys = {(r["player_id"], r["team_id"]) for r in existing}
        new_history = [h for h in history_records
                        if (h["player_id"], h["team_id"]) not in existing_keys]
        for batch in chunked(new_history):
            supabase.table("player_team_history").insert(batch).execute()
        print(f"  Inserted {len(new_history)} new player_team_history rows for {season}.")


def backfill_home_venues(supabase):
    """Best-effort: set each team's home_venue_id to the venue they played 'home' at most often."""
    print("Computing each team's home venue from game history...")
    games = supabase.table("games").select("home_team_id,venue_id").execute().data
    from collections import Counter
    counts = {}
    for g in games:
        if not g["home_team_id"] or not g["venue_id"]:
            continue
        counts.setdefault(g["home_team_id"], Counter())[g["venue_id"]] += 1
    updates = 0
    for team_id, counter in counts.items():
        top_venue_id = counter.most_common(1)[0][0]
        supabase.table("teams").update({"home_venue_id": top_venue_id}).eq("id", team_id).execute()
        updates += 1
    print(f"  Set home_venue_id for {updates} teams.")


if __name__ == "__main__":
    sb = get_supabase()
    backfill_teams(sb)
    backfill_venues(sb)
    backfill_players(sb)
    backfill_games_and_season_stats(sb)
    backfill_home_venues(sb)
    print("\nBackfill complete. Next: set up ingest.py to run on a schedule (see README.md).")
