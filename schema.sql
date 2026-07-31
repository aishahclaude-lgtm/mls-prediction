-- MLS Prediction Dashboard — Supabase / Postgres schema
-- Run this once in Supabase: Project -> SQL Editor -> New query -> paste -> Run.
--
-- Design note: we pull from two data providers (American Soccer Analysis = "asa",
-- and ESPN's public scoreboard = "espn") which use different, incompatible IDs for
-- the same real-world team/venue/player. Every table below carries both external ID
-- columns so the backfill and ingestion scripts can upsert safely from either source
-- without creating duplicates.

create extension if not exists "uuid-ossp";

-- ============================================================
-- VENUES
-- ============================================================
create table venues (
    id              uuid primary key default uuid_generate_v4(),
    asa_stadium_id  text unique,
    espn_venue_id   text unique,
    name            text not null,
    city            text,
    province        text,
    country         text,
    latitude        numeric,
    longitude       numeric,
    capacity        integer,
    is_turf         boolean,          -- true = artificial turf, false = grass
    has_roof        boolean,
    year_built      integer,
    created_at      timestamptz default now(),
    updated_at      timestamptz default now()
);

-- ============================================================
-- TEAMS
-- ============================================================
create table teams (
    id              uuid primary key default uuid_generate_v4(),
    asa_team_id     text unique,
    espn_team_id    text unique,
    name            text not null,
    short_name      text,
    abbreviation    text,
    conference      text,             -- 'Eastern' / 'Western' — filled in manually, ASA doesn't expose this cleanly
    home_venue_id   uuid references venues(id),
    logo_url        text,
    elo_rating       numeric default 1500,   -- updated by predict.py every retrain
    attack_rating    numeric,                -- avg goals scored, last 20 games
    defense_rating   numeric,                -- avg goals conceded, last 20 games
    ratings_updated_at timestamptz,
    created_at      timestamptz default now(),
    updated_at      timestamptz default now()
);

-- ============================================================
-- PLAYERS
-- players.current_team_id is a convenience pointer only — the source of truth for
-- "who played for whom, when" is player_team_history below.
-- ============================================================
create table players (
    id                      uuid primary key default uuid_generate_v4(),
    asa_player_id           text unique,
    espn_player_id          text unique,
    name                    text not null,
    birth_date              date,
    height_in               integer,   -- total height in inches (ft*12 + in)
    weight_lb               integer,
    nationality             text,
    primary_position        text,
    current_team_id         uuid references teams(id),
    created_at              timestamptz default now(),
    updated_at              timestamptz default now()
);

-- ============================================================
-- PLAYER TEAM HISTORY — this is what makes trades work correctly.
-- One row per stint. end_date is null while the stint is current.
-- Historical backfill inserts one row per season+team (ASA's granularity);
-- the live ingestion job's daily roster-diff refines this with real dates
-- going forward and closes out a stint the moment a trade is detected.
-- ============================================================
create table player_team_history (
    id           uuid primary key default uuid_generate_v4(),
    player_id    uuid not null references players(id),
    team_id      uuid not null references teams(id),
    season_name  text,               -- e.g. '2025' — set when the row comes from historical backfill
    start_date   date,
    end_date     date,               -- null = current stint
    source       text default 'backfill',   -- 'backfill' or 'roster_diff'
    created_at   timestamptz default now()
);
create index idx_player_team_history_player on player_team_history(player_id);
create index idx_player_team_history_team on player_team_history(team_id);

-- ============================================================
-- GAMES
-- ============================================================
create table games (
    id               uuid primary key default uuid_generate_v4(),
    asa_game_id      text unique,
    espn_event_id    text unique,
    date_time_utc    timestamptz not null,
    season_name      text,
    matchday         integer,
    home_team_id     uuid references teams(id),
    away_team_id     uuid references teams(id),
    venue_id         uuid references venues(id),
    home_score       integer,
    away_score       integer,
    status           text not null default 'scheduled', -- 'scheduled' | 'live' | 'final'
    attendance       integer,
    knockout_game    boolean default false,
    last_updated_utc timestamptz default now(),
    created_at       timestamptz default now()
);
create index idx_games_home_team on games(home_team_id);
create index idx_games_away_team on games(away_team_id);
create index idx_games_venue on games(venue_id);
create index idx_games_status on games(status);
create index idx_games_date on games(date_time_utc);

-- ============================================================
-- TEAM GAME STATS (per team, per game)
-- ============================================================
create table team_game_stats (
    id               uuid primary key default uuid_generate_v4(),
    game_id          uuid not null references games(id),
    team_id          uuid not null references teams(id),
    possession_pct   numeric,
    shots            integer,
    shots_on_target  integer,
    fouls            integer,
    corners          integer,
    xg               numeric,
    unique(game_id, team_id)
);

-- ============================================================
-- PLAYER GAME STATS (per player, per game)
-- Sourced from ESPN's per-game boxscore. Field availability from ESPN's free
-- endpoint varies game to game (e.g. "minutes" isn't always populated) — treat
-- nulls as "not reported" rather than zero.
-- team_id = the team the player actually played FOR in this specific game,
-- so a mid-season trade never misattributes a stat line to the wrong team.
-- ============================================================
create table player_game_stats (
    id                 uuid primary key default uuid_generate_v4(),
    game_id            uuid not null references games(id),
    player_id          uuid not null references players(id),
    team_id            uuid not null references teams(id),
    minutes            integer,
    goals              integer,
    assists            integer,
    shots              integer,
    shots_on_target    integer,
    yellow_cards       integer,
    red_cards          integer,
    fouls_committed    integer,
    passes_completed   integer,
    passes_attempted   integer,
    unique(game_id, player_id)
);
create index idx_player_game_stats_player on player_game_stats(player_id);
create index idx_player_game_stats_game on player_game_stats(game_id);

-- ============================================================
-- PLAYER SEASON XGOALS — ASA's advanced metrics (xG, xA, goals added) are only
-- available at season granularity on the free API, not per game. This table
-- holds that authoritative season-level data separately from the per-game
-- boxscore above, refreshed periodically (weekly is plenty — it's cumulative).
-- ============================================================
create table player_season_xgoals (
    id                      uuid primary key default uuid_generate_v4(),
    player_id               uuid not null references players(id),
    team_id                 uuid references teams(id),
    season_name             text not null,
    general_position        text,
    minutes_played          integer,
    shots                   integer,
    shots_on_target         integer,
    goals                   integer,
    xgoals                  numeric,
    key_passes              integer,
    primary_assists         integer,
    xassists                numeric,
    points_added            numeric,
    xpoints_added           numeric,
    updated_at              timestamptz default now(),
    unique(player_id, team_id, season_name)
);

-- ============================================================
-- PREDICTIONS
-- ============================================================
create table predictions (
    id                        uuid primary key default uuid_generate_v4(),
    game_id                   uuid not null references games(id),
    model_version              text,
    predicted_home_win_pct    numeric,
    predicted_draw_pct        numeric,
    predicted_away_win_pct    numeric,
    predicted_home_score      numeric,
    predicted_away_score      numeric,
    confidence                text,     -- 'High' | 'Medium' | 'Low' — how far the top
                                         -- probability sits above a coin-flip guess
    created_at                timestamptz default now()
);
create index idx_predictions_game on predictions(game_id);
alter table predictions add constraint predictions_game_id_key unique (game_id);

-- ============================================================
-- MODEL RUNS — a log of each retrain, so accuracy over time is auditable.
-- ============================================================
create table model_runs (
    id                  uuid primary key default uuid_generate_v4(),
    trained_at          timestamptz default now(),
    training_row_count  integer,
    notes               text,
    accuracy_metrics    jsonb
);

-- ============================================================
-- AWARD PREDICTIONS — MLS end-of-season awards (MVP, Golden Boot, Shield,
-- MLS Cup, etc), recomputed by awards.py at the end of every predict.py
-- retrain. Fully overwritten each run (delete-all + reinsert) rather than
-- upserted, since the shape of the top-5 for an award can change entirely
-- between runs. is_proxy/proxy_note flag the awards this database only has
-- a rough stand-in signal for (no coaches/goalkeeper/defensive stats) —
-- the dashboard should show those visibly lower-confidence, not as real
-- picks. See awards.py for exactly what each award_key measures.
-- ============================================================
create table award_predictions (
    id              uuid primary key default uuid_generate_v4(),
    award_key       text not null,      -- e.g. 'mvp', 'golden_boot', 'supporters_shield'
    award_name      text not null,      -- display name, e.g. 'Landon Donovan MLS MVP'
    season_name     text,
    model_version   text,
    rank            integer not null,   -- 1 = current favorite
    entity_type     text not null,      -- 'player' | 'team'
    entity_id       uuid,               -- player id or team id
    entity_name     text not null,
    subtitle        text,               -- team + stat line, or points/Elo for team awards
    win_pct         numeric not null,
    is_proxy        boolean not null default false,
    proxy_note      text,
    updated_at      timestamptz default now()
);
create index idx_award_predictions_award on award_predictions(award_key);

-- ============================================================
-- Realtime: tell Supabase to broadcast changes on the tables the dashboard
-- needs to react to live (run this after creating the tables above).
-- ============================================================
alter publication supabase_realtime add table games;
alter publication supabase_realtime add table team_game_stats;
alter publication supabase_realtime add table player_game_stats;
alter publication supabase_realtime add table predictions;
alter publication supabase_realtime add table award_predictions;
