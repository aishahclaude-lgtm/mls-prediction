# MLS Prediction Dashboard — Setup

The database, the one-time historical load, the live update job, the
prediction model, and the dashboard website. Every run of the live ingestion
job also retrains the model and refreshes predictions for upcoming games —
nothing extra to run by hand once this is deployed. Total cost: **$0**, using
free tiers only.

## What you need

- A **Supabase** account (free) — https://supabase.com
- A **GitHub** account (free) — https://github.com
- ~20 minutes for setup, plus however long the historical backfill takes to run

---

## Step 1 — Create the Supabase project and database

1. Go to https://supabase.com, sign up/log in, click **New project**.
2. Pick any name (e.g. `mls-prediction`), set a database password (save it somewhere), pick the region closest to you, and create the project. Wait ~2 minutes for it to finish provisioning.
3. In the left sidebar, click the **SQL Editor**, click **New query**, open `schema.sql` from this folder, paste its entire contents in, and click **Run**. You should see "Success" and a new set of tables appear under **Table Editor**.
4. In the left sidebar, click the gear icon **Project Settings -> API**. You'll need two values from this page in Step 3:
   - **Project URL** (looks like `https://xxxxxxxx.supabase.co`)
   - **service_role key** (under "Project API keys" — NOT the `anon` key; the service role key is what lets the scripts write data. Keep it secret, it's not meant for a browser.)

## Step 2 — Put this project on GitHub

From a terminal, inside this folder:

```bash
git init
git add .
git commit -m "Initial MLS prediction pipeline"
```

Then on GitHub: click **New repository**, name it (e.g. `mls-prediction`), leave it empty (no README/license), create it, and follow the "push an existing repository" instructions it shows you, e.g.:

```bash
git remote add origin https://github.com/YOUR_USERNAME/mls-prediction.git
git branch -M main
git push -u origin main
```

The repo can be public or private — public just guarantees unlimited free GitHub Actions minutes, which matters since the ingestion job runs every 15 minutes, all day, every day.

## Step 3 — Add your Supabase credentials as GitHub secrets

In your new GitHub repo: **Settings -> Secrets and variables -> Actions -> New repository secret**. Add two:

- `SUPABASE_URL` = the Project URL from Step 1
- `SUPABASE_KEY` = the service_role key from Step 1

## Step 4 — Run the one-time historical backfill

In your repo: **Actions** tab -> you should see two workflows listed (**Live ingestion** and **Historical backfill (manual)**) -> click **Historical backfill (manual)** -> **Run workflow** -> **Run workflow** (green button).

This pulls all MLS teams, stadiums, players, and the last 3 seasons of games/stats. Watch it run under the Actions tab; it can take a few minutes. When it finishes, go back to Supabase's **Table Editor** and spot-check: `teams` should have ~30 rows, `venues` should have real stadium names, `games` should have hundreds of rows across 3 seasons, `player_season_xgoals` should be populated.

If it fails partway, it's safe to just re-run it — everything is upserted, so it won't duplicate what already loaded.

## Step 5 — Confirm the live job is scheduled

The **Live ingestion** workflow (`ingest.yml`) is already scheduled to run every 15 minutes once it's pushed to GitHub — nothing more to do. You can trigger it manually the same way (Actions -> Live ingestion -> Run workflow) to test it immediately rather than waiting.

**Important GitHub quirk:** GitHub automatically disables scheduled workflows on a repo after **60 days with no commits**. If you don't touch the repo for two months, the live job will silently stop — just re-enable it from the Actions tab (or push any small commit) if that happens.

---

## Step 6 — Deploy the dashboard website (free)

The dashboard (`streamlit_app.py`) reads straight from Supabase — no
separate backend needed. Deploy it for free on Streamlit Community Cloud:

1. Go to https://share.streamlit.io, sign in with your GitHub account, click **Create app** -> **Yep, I have an app** (or **New app**).
2. Pick the repo you pushed in Step 2, branch `main`, main file path `streamlit_app.py`.
3. Click **Advanced settings** -> **Secrets**, and paste:
   ```
   SUPABASE_URL = "https://xxxxxxxx.supabase.co"
   SUPABASE_KEY = "your-service-role-key"
   ```
   (Same two values from Step 1. These secrets are private to you — nobody who visits your deployed app can see them.)
4. Click **Deploy**. In a minute or two you'll get a permanent URL like `https://your-app-name.streamlit.app` — bookmark it.

The page re-queries Supabase every 60 seconds, so it reflects new
scores/predictions shortly after each live-ingestion run. It has four tabs:
upcoming-game predictions (win/draw/loss %, predicted score), Elo power
rankings for every team, a head-to-head lookup between any two teams, and
player season stats (goals, xG, assists, xA).

## Debugging tip: player box scores

ESPN's free box score endpoint doesn't publish a perfectly documented, stable shape for player-level stats — `ingest.py`'s `parse_boxscore()` function tries the standard shape, but if you check the `player_game_stats` table after a game finishes and it's empty while `team_game_stats` has data, do this:

```python
from common import espn_get
summary = espn_get("summary", params={"event": "SOME_FINISHED_EVENT_ID"})
print(summary["boxscore"].keys())
```

and adjust `parse_boxscore()` in `ingest.py` to match whatever keys you actually see. Team-level stats and all score/status/venue data are unaffected either way — this only affects the depth of individual player game logs.

## Running things locally / in Colab instead of waiting on GitHub Actions

Both scripts are plain Python and work fine outside GitHub Actions — useful for testing before you push, or for one-off manual runs:

```python
import os
os.environ["SUPABASE_URL"] = "https://xxxx.supabase.co"
os.environ["SUPABASE_KEY"] = "your-service-role-key"
!pip install -r requirements.txt
!python backfill.py
```

---

## What's next

This gets the data foundation live: all MLS teams, 3 seasons of history, trades tracked automatically, venues linked, and new results flowing in within 15 minutes of a game ending. Two things are intentionally not built yet:

1. **The dashboard** — the actual live-updating UI you'll look at, reading straight from this Supabase database.
2. **The prediction model** — trained on this data once there's a real backlog to train on.

Once you've run the backfill and confirmed data looks right in Supabase's Table Editor, that's the signal to move to the dashboard next.
