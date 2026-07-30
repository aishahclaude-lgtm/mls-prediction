"""
MLS Prediction Dashboard — Streamlit front end.

Reads straight from the Supabase database populated by backfill.py / ingest.py
(or the Colab notebook doing the same job). No local state, no writes — this
is a read-only viewer so you can compare the model's predictions against your
own bets.

Deploy for free on Streamlit Community Cloud (share.streamlit.io):
  1. Push this whole folder to a GitHub repo (see README.md).
  2. On share.streamlit.io: New app -> pick the repo -> main file = streamlit_app.py.
  3. In the app's Settings -> Secrets, paste:
        SUPABASE_URL = "https://xxxx.supabase.co"
        SUPABASE_KEY = "your-service-role-key"
     (Same two values from Supabase Project Settings -> API. Secrets are
     private to you — visitors to the deployed app never see them.)
  4. Deploy. The app re-queries Supabase every 60s, so it reflects new
     results/predictions shortly after your ingestion job runs.
"""
import datetime
import json
import re

import pandas as pd
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="MLS Predictions", page_icon="⚽", layout="wide")

# ============================================================
# Look & feel — plain white background, monospace/typewriter text.
# ============================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Courier New", Courier, monospace !important;
}
.stApp {
    background-color: #ffffff;
    color: #111111;
}
h1, h2, h3, h4 {
    font-family: "Courier New", Courier, monospace !important;
    font-weight: 700;
    letter-spacing: -0.5px;
}
[data-testid="stMetricValue"] {
    font-family: "Courier New", Courier, monospace !important;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #ddd;
}
.stTabs [data-baseweb="tab"] {
    font-family: "Courier New", Courier, monospace !important;
}
hr { border-top: 1px solid #ddd; }

/* ---- Mobile tweaks: tighter padding, smaller text, scrollable tabs ---- */
@media (max-width: 640px) {
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 1.5rem !important;
    }
    h1 { font-size: 1.4rem !important; }
    h2, h3 { font-size: 1.1rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.7rem !important; }
    div[data-testid="stDataFrame"] { font-size: 0.75rem !important; }
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.8rem !important;
        padding: 0.4rem 0.55rem !important;
        white-space: nowrap !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Supabase connection
# ============================================================
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    st.error(
        "Missing Supabase credentials. Add SUPABASE_URL and SUPABASE_KEY "
        "in this app's Settings -> Secrets (see the comment at the top of "
        "streamlit_app.py for the exact steps)."
    )
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_all(table, columns="*", order_col=None, desc=False):
    rows = []
    start = 0
    page = 1000
    while True:
        q = supabase.table(table).select(columns)
        if order_col:
            q = q.order(order_col, desc=desc)
        batch = q.range(start, start + page - 1).execute().data
        rows.extend(batch)
        if len(batch) < page:
            break
        start += page
    return rows


@st.cache_data(ttl=60)
def load_data():
    teams = pd.DataFrame(fetch_all("teams"))
    venues = pd.DataFrame(fetch_all("venues"))
    games = pd.DataFrame(fetch_all("games"))
    predictions = pd.DataFrame(fetch_all("predictions"))
    players = pd.DataFrame(fetch_all("players"))
    player_season_xgoals = pd.DataFrame(fetch_all("player_season_xgoals"))
    model_runs = pd.DataFrame(fetch_all("model_runs", order_col="trained_at", desc=True))
    return teams, venues, games, predictions, players, player_season_xgoals, model_runs


teams, venues, games, predictions, players, player_xg, model_runs = load_data()

if teams.empty or games.empty:
    st.warning(
        "No data yet. Run the historical backfill (backfill.py or the Colab "
        "notebook's backfill cell) at least once, then reload this page."
    )
    st.stop()

team_name = dict(zip(teams["id"], teams["name"]))
team_abbr = dict(zip(teams["id"], teams["abbreviation"]))
venue_name = dict(zip(venues["id"], venues["name"])) if not venues.empty else {}

st.title("MLS Predictions")
st.caption(
    f"Reads live from Supabase, refreshes every 60s. "
    f"Last model run: {model_runs.iloc[0]['trained_at'] if not model_runs.empty else 'never'}."
)

tab_pred, tab_standings, tab_h2h, tab_players, tab_chat = st.tabs(
    ["Predictions", "Standings", "H2H", "Players", "Ask"]
)

# ============================================================
# Tab 1 — Upcoming predictions
# ============================================================
with tab_pred:
    upcoming = games[games["status"] == "scheduled"].copy()
    if upcoming.empty:
        st.info("No upcoming games with predictions right now — check back after the next ingestion run.")
    else:
        upcoming["date_time_utc"] = pd.to_datetime(upcoming["date_time_utc"])
        upcoming = upcoming.sort_values("date_time_utc")
        preds_by_game = {p["game_id"]: p for p in predictions.to_dict("records")} if not predictions.empty else {}

        for _, g in upcoming.iterrows():
            pred = preds_by_game.get(g["id"])
            home = team_name.get(g["home_team_id"], "?")
            away = team_name.get(g["away_team_id"], "?")
            venue = venue_name.get(g["venue_id"], "")
            when = g["date_time_utc"].strftime("%a %b %d, %Y — %H:%M UTC")

            with st.container(border=True):
                # Matchup info gets its own full-width line (readable at any screen
                # size) instead of sharing a row with the metrics — a [3,1,1,1]
                # column split gets uncomfortably cramped on a phone-width screen.
                st.markdown(f"**{home}** vs **{away}**  \n{when}" + (f"  \n_{venue}_" if venue else ""))
                if pred:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Home win", f"{pred['predicted_home_win_pct']:.0f}%")
                    m2.metric("Draw", f"{pred['predicted_draw_pct']:.0f}%")
                    m3.metric("Away win", f"{pred['predicted_away_win_pct']:.0f}%")
                    st.caption(
                        f"Predicted score: {home} {pred['predicted_home_score']:.1f} — "
                        f"{pred['predicted_away_score']:.1f} {away}"
                    )
                else:
                    st.caption("(no prediction yet)")

# ============================================================
# Tab 2 — Elo standings / power rankings
# ============================================================
with tab_standings:
    st.subheader("Power rankings (Elo rating)")
    if "elo_rating" not in teams.columns or teams["elo_rating"].isna().all():
        st.info("Ratings haven't been computed yet — run the prediction model cell/script once.")
    else:
        standings = teams.copy()
        standings = standings.sort_values("elo_rating", ascending=False).reset_index(drop=True)
        standings.insert(0, "Rank", standings.index + 1)
        standings = standings[["Rank", "name", "abbreviation", "elo_rating", "attack_rating", "defense_rating"]]
        standings.columns = ["Rank", "Team", "Abbr", "Elo", "Attack", "Defense"]
        st.caption("Attack/Defense = avg goals scored/conceded per game (last 20). Swipe the table sideways to see every column.")
        st.dataframe(standings, use_container_width=True, hide_index=True)

    if not model_runs.empty:
        st.divider()
        st.subheader("Model accuracy over time")
        st.caption(
            "Overall accuracy of the model across ALL games, not broken down by team. "
            "Each row is one retrain (happens automatically on every ingestion run). "
            "'Test accuracy' = % of held-out past games where the model correctly called "
            "the outcome (home win / draw / away win). 'Baseline' = accuracy you'd get by "
            "always guessing 'home team wins' — the bar the model needs to beat."
        )
        acc_rows = []
        for _, r in model_runs.iterrows():
            m = r.get("accuracy_metrics") or {}
            acc_rows.append({
                "Trained at": r["trained_at"],
                "Test acc.": m.get("test_accuracy"),
                "Baseline": m.get("baseline_accuracy"),
                "CV acc.": m.get("cv_accuracy"),
                "Rows": r.get("training_row_count"),
            })
        st.dataframe(pd.DataFrame(acc_rows), use_container_width=True, hide_index=True)

# ============================================================
# Tab 3 — Head-to-head lookup
# ============================================================
with tab_h2h:
    st.subheader("Head-to-head history")
    team_options = sorted(team_name.values())
    c1, c2 = st.columns(2)
    team_a_name = c1.selectbox("Team A", team_options, index=0 if team_options else None)
    team_b_name = c2.selectbox("Team B", team_options, index=1 if len(team_options) > 1 else 0)

    if team_a_name and team_b_name and team_a_name != team_b_name:
        id_by_name = {v: k for k, v in team_name.items()}
        a_id, b_id = id_by_name[team_a_name], id_by_name[team_b_name]
        finished = games[
            (games["status"] == "final")
            & (games["home_score"].notna())
            & (
                ((games["home_team_id"] == a_id) & (games["away_team_id"] == b_id))
                | ((games["home_team_id"] == b_id) & (games["away_team_id"] == a_id))
            )
        ].copy()
        finished["date_time_utc"] = pd.to_datetime(finished["date_time_utc"])
        finished = finished.sort_values("date_time_utc", ascending=False)

        if finished.empty:
            st.info("No completed meetings between these two teams yet.")
        else:
            a_wins = b_wins = draws = 0
            display_rows = []
            for _, g in finished.iterrows():
                h, aw = team_name.get(g["home_team_id"]), team_name.get(g["away_team_id"])
                hs, aws = g["home_score"], g["away_score"]
                if hs == aws:
                    draws += 1
                elif (hs > aws and g["home_team_id"] == a_id) or (hs < aws and g["away_team_id"] == a_id):
                    a_wins += 1
                else:
                    b_wins += 1
                display_rows.append({
                    "Date": g["date_time_utc"].strftime("%Y-%m-%d"),
                    "Home": h, "Away": aw, "Score": f"{int(hs)} - {int(aws)}",
                })
            m1, m2, m3 = st.columns(3)
            m1.metric(f"{team_a_name} wins", a_wins)
            m2.metric("Draws", draws)
            m3.metric(f"{team_b_name} wins", b_wins)
            st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

# ============================================================
# Tab 4 — Player stats
# ============================================================
with tab_players:
    st.subheader("Player season stats (xG / xA / goals added)")
    st.caption(
        "This table only includes players with recorded shot/chance-creation data for "
        "the selected season (sourced from ASA's advanced-stats feed) — that's usually "
        "a subset of the full squad, since bench players and some defenders/keepers "
        "may show zero qualifying actions. See 'Full roster' below for everyone on the "
        "team, defenders included, with 'Goals Added' as the stat that best reflects "
        "defensive contribution (ASA's goals-added model blends attacking AND defending "
        "actions like tackles/interceptions/positioning into one number — it isn't "
        "shot-dependent, so it's meaningful for center-backs and defensive mids too)."
    )
    team_pick = st.selectbox("Team (optional)", ["All"] + sorted(team_name.values()))

    merged = pd.DataFrame()
    seasons = []
    season_pick = None
    if not player_xg.empty and not players.empty:
        merged = player_xg.merge(players[["id", "name"]], left_on="player_id", right_on="id", how="left")
        merged["Team"] = merged["team_id"].map(team_name)
        seasons = sorted(merged["season_name"].dropna().unique(), reverse=True)
        season_pick = st.selectbox("Season", seasons) if seasons else None

        view = merged[merged["season_name"] == season_pick] if season_pick else merged
        if team_pick != "All":
            view = view[view["Team"] == team_pick]

        view = view[[
            "name", "Team", "general_position", "minutes_played", "goals", "xgoals",
            "primary_assists", "xassists", "shots", "shots_on_target", "points_added",
        ]].rename(columns={
            "name": "Player", "general_position": "Pos", "minutes_played": "Min",
            "goals": "Goals", "xgoals": "xG", "primary_assists": "Assists",
            "xassists": "xA", "shots": "Shots", "shots_on_target": "SOT",
            "points_added": "Goals Added",
        })
        view = view.sort_values("Goals", ascending=False)
        st.dataframe(view, use_container_width=True, hide_index=True)
    else:
        st.info("No player season stats yet — run the backfill.")

    st.divider()
    st.subheader("Full roster")
    if team_pick == "All":
        st.caption("Pick a specific team above to see its full current roster, stats included.")
    elif "current_team_id" not in players.columns:
        st.info("Roster data not available yet.")
    else:
        id_by_name = {v: k for k, v in team_name.items()}
        team_id_pick = id_by_name.get(team_pick)
        roster = players[players["current_team_id"] == team_id_pick].copy()
        if roster.empty:
            st.info(
                "No roster synced for this team yet — the live ingestion job "
                "(every 15 min) fills this in from ESPN roster data."
            )
        else:
            # Position sometimes comes through as a raw ESPN {"abbreviation": ...} blob
            # for players first added via roster sync — unpack it into a clean string.
            def _clean_pos(val):
                if isinstance(val, str) and val.strip().startswith("{"):
                    try:
                        d = json.loads(val)
                        return d.get("abbreviation") or d.get("name") or val
                    except Exception:
                        return val
                return val
            roster["Pos"] = roster["primary_position"].apply(_clean_pos)

            # Left-join season stats (if a season is selected) so EVERY roster player
            # gets a row here — including defenders/keepers who had zero shots and so
            # were filtered out of the table above. Goals Added is the column to look
            # at for non-attackers since it's not shot-dependent.
            stat_cols = ["Min", "Goals", "xG", "Assists", "xA", "Shots", "SOT", "Goals Added"]
            if not merged.empty and season_pick:
                season_stats = merged[merged["season_name"] == season_pick][[
                    "player_id", "minutes_played", "goals", "xgoals",
                    "primary_assists", "xassists", "shots", "shots_on_target", "points_added",
                ]].rename(columns={
                    "minutes_played": "Min", "goals": "Goals", "xgoals": "xG",
                    "primary_assists": "Assists", "xassists": "xA", "shots": "Shots",
                    "shots_on_target": "SOT", "points_added": "Goals Added",
                })
                roster = roster.merge(season_stats, left_on="id", right_on="player_id", how="left")
            else:
                for c in stat_cols:
                    roster[c] = None

            roster = roster[["name", "Pos", "nationality"] + stat_cols].rename(columns={
                "name": "Player", "nationality": "Nationality",
            })
            roster["Goals Added"] = roster["Goals Added"].fillna(0)
            roster = roster.sort_values("Goals Added", ascending=False)
            st.caption(
                f"{len(roster)} players currently on {team_pick}'s roster"
                + (f" — stats for {season_pick}." if season_pick else ". Pick a season above for stats.")
            )
            st.dataframe(roster, use_container_width=True, hide_index=True)

# ============================================================
# Tab 5 — Ask (rule-based matchup assistant, no external AI)
# ============================================================
with tab_chat:
    st.subheader("Ask about a matchup")
    st.caption(
        "This isn't a general chatbot — it's a small rule-based helper that only answers "
        "from this app's own predictions and ratings. If it can't find a real scheduled game "
        "between the two teams, or isn't sure who you mean, it asks before guessing instead "
        "of making one up."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": (
                "Ask me something like \"who's going to win Miami vs Vancouver?\" — I'll look "
                "for a real scheduled match between them first."
            ),
        }]
    if "pending_hypothetical" not in st.session_state:
        st.session_state.pending_hypothetical = None

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Words like "City"/"United"/"FC" show up in half the league's names, so
    # matching on any single one of them (the old logic) made almost every
    # query look "ambiguous" (e.g. "New York City FC" tripped on "City" and
    # pulled in Orlando City, Sporting KC, St. Louis City too). Generic words
    # are excluded from matching entirely; only distinctive words count.
    _GENERIC_TEAM_WORDS = {
        "fc", "sc", "city", "united", "real", "union", "town", "county",
        "athletic", "club",
    }

    def _find_teams_in_text(text):
        text_l = text.lower()
        matches = []
        for tid, name in team_name.items():
            name_l = name.lower()
            if name_l in text_l:
                matches.append(tid)
                continue
            clean = re.sub(r"[^a-z0-9\s]", "", name_l)
            words = [
                w for w in clean.split()
                if len(w) >= 4 and w not in _GENERIC_TEAM_WORDS
            ]
            if words and any(re.search(rf"\b{re.escape(w)}\b", text_l) for w in words):
                matches.append(tid)
        for tid, abbr in team_abbr.items():
            abbr_l = (abbr or "").lower()
            if abbr_l and re.search(rf"\b{re.escape(abbr_l)}\b", text_l) and tid not in matches:
                matches.append(tid)
        return matches

    def _answer_matchup(a_id, b_id):
        # Only answer from a REAL scheduled game with a real model prediction —
        # never fabricate one from scratch, per the "don't blind-guess" rule.
        cand = games[
            (games["status"] == "scheduled")
            & (
                ((games["home_team_id"] == a_id) & (games["away_team_id"] == b_id))
                | ((games["home_team_id"] == b_id) & (games["away_team_id"] == a_id))
            )
        ].copy()
        if len(cand) == 1:
            g = cand.iloc[0]
            pred = None
            if not predictions.empty:
                m = predictions[predictions["game_id"] == g["id"]]
                if not m.empty:
                    pred = m.iloc[0]
            home, away = team_name.get(g["home_team_id"]), team_name.get(g["away_team_id"])
            when = pd.to_datetime(g["date_time_utc"]).strftime("%a %b %d, %Y — %H:%M UTC")
            if pred is None:
                return (
                    f"Found the scheduled match **{home}** vs **{away}** on {when}, but there's "
                    f"no model prediction for it yet — check back after the next retrain."
                )
            return (
                f"**{home}** vs **{away}** — {when}\n\n"
                f"- Home win: {pred['predicted_home_win_pct']:.0f}%\n"
                f"- Draw: {pred['predicted_draw_pct']:.0f}%\n"
                f"- Away win: {pred['predicted_away_win_pct']:.0f}%\n\n"
                f"Predicted score: {home} {pred['predicted_home_score']:.1f} — "
                f"{pred['predicted_away_score']:.1f} {away}\n\n"
                f"_Based on the model's Elo, recent form, head-to-head, venue, and rest-day "
                f"features for this specific matchup._"
            )
        elif len(cand) > 1:
            opts = "; ".join(
                f"{team_name.get(r['home_team_id'])} vs {team_name.get(r['away_team_id'])} on "
                f"{pd.to_datetime(r['date_time_utc']).strftime('%b %d')}"
                for _, r in cand.iterrows()
            )
            return f"There's more than one scheduled meeting between these teams: {opts}. Which one did you mean?"
        else:
            st.session_state.pending_hypothetical = (a_id, b_id)
            return (
                f"I don't see an upcoming scheduled match between **{team_name.get(a_id)}** and "
                f"**{team_name.get(b_id)}** in the data, so I don't want to guess blind. Want a "
                f"rough hypothetical instead, using each team's current Elo rating on a neutral "
                f"field (no venue, rest, form, or head-to-head factored in — much less reliable "
                f"than a real prediction)? Reply \"yes\" if so, or tell me the actual matchup you meant."
            )

    def _hypothetical_answer(a_id, b_id):
        if "elo_rating" not in teams.columns:
            return "Ratings haven't been computed yet, so I can't even do a rough estimate — run the model once first."
        elo = dict(zip(teams["id"], teams["elo_rating"]))
        ea, eb = elo.get(a_id), elo.get(b_id)
        if ea is None or eb is None or pd.isna(ea) or pd.isna(eb):
            return "Missing Elo ratings for one of these teams — can't estimate."
        diff = ea - eb
        p_a = 1 / (1 + 10 ** (-diff / 400))
        return (
            f"Rough neutral-field estimate only (Elo-only — ignores venue/rest/form/head-to-head, "
            f"and doesn't model draws, so treat this as a coin-flip-plus-lean, not a real prediction): "
            f"**{team_name.get(a_id)}** ({ea:.0f} Elo) vs **{team_name.get(b_id)}** ({eb:.0f} Elo) "
            f"→ **{team_name.get(a_id)}** ~{p_a * 100:.0f}% / **{team_name.get(b_id)}** ~{(1 - p_a) * 100:.0f}%. "
            f"For a real prediction, wait for this matchup to be added to the schedule."
        )

    user_msg = st.chat_input('e.g. "who wins Miami vs Vancouver"')
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        pending = st.session_state.pending_hypothetical
        low = user_msg.strip().lower()
        if pending and low in ("yes", "y", "sure", "yeah", "yep", "ok", "okay"):
            reply = _hypothetical_answer(*pending)
            st.session_state.pending_hypothetical = None
        else:
            st.session_state.pending_hypothetical = None
            found = _find_teams_in_text(user_msg)
            if len(found) == 0:
                reply = (
                    "I couldn't match any team names in that. Try naming both teams, e.g. "
                    '"who wins Inter Miami vs Vancouver Whitecaps".'
                )
            elif len(found) == 1:
                reply = f"Got **{team_name.get(found[0])}** — who are they playing?"
            elif len(found) > 2:
                names = ", ".join(team_name.get(t) for t in found)
                reply = f"That matched more than two teams ({names}) — which two did you mean?"
            else:
                reply = _answer_matchup(found[0], found[1])

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()
