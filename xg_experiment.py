"""Measure whether game-level xG features help, BEFORE putting them in the model.

Run this via GitHub Actions (workflow_dispatch). It writes nothing — it fetches,
backtests, prints, exits. Nothing about the live pipeline changes until you
decide something here is worth keeping.

Why this exists as a separate script: the offline study found that adding eight
plausible new features at once cost 3.5 points of accuracy, because ~1,050
training rows can't support that many extra parameters. So features get added
ONE AT A TIME and each one has to earn its place against the current baseline.
That is what this measures.

What it tests, each on its own and then together:
    xg_form_diff       net xG differential over the last N games (home - away)
    xg_overperf_diff   goals minus xG over the last N (positive = riding luck)
    xpoints_form_diff  ASA's expected-points, rolled forward
and the same three REPLACING the current prior-season xg_prior_diff rather
than adding to it, since that's the cheaper change parameter-wise.
"""
import datetime
import os
import sys

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from common import asa_get, fetch_all, get_supabase

FORM_N = 8            # games in the rolling xG window
SEASONS = ("2023", "2024", "2025", "2026")

# The 8 features predict.py's Poisson stage currently uses.
BASE_FEATS = ["elo_diff", "home_form_ppg", "away_form_ppg", "home_form_gd",
              "away_form_gd", "xg_prior_diff", "venue_ppg", "h2h_home_ppg"]


# ---------------------------------------------------------------------------
# 1. pull game-level xG from ASA and join it to our games
# ---------------------------------------------------------------------------
def load_game_xg(supabase):
    """{our_game_uuid: dict of xG fields}, joined via games.asa_game_id."""
    rows = fetch_all(supabase, "games", "id,asa_game_id")
    by_asa = {r["asa_game_id"]: r["id"] for r in rows if r.get("asa_game_id")}
    print(f"  {len(by_asa)} games in our DB carry an asa_game_id")

    out = {}
    for season in SEASONS:
        try:
            data = asa_get("games/xgoals", params={"season_name": season})
        except Exception as e:  # noqa: BLE001
            print(f"  {season}: fetch failed ({e}) — skipping")
            continue
        hit = 0
        for g in data or []:
            gid = by_asa.get(g.get("game_id"))
            if not gid:
                continue
            try:
                out[gid] = {
                    "hxg": float(g["home_team_xgoals"]),
                    "axg": float(g["away_team_xgoals"]),
                    "hg": float(g["home_goals"]),
                    "ag": float(g["away_goals"]),
                    "hxp": float(g.get("home_xpoints") or 0.0),
                    "axp": float(g.get("away_xpoints") or 0.0),
                }
                hit += 1
            except (TypeError, ValueError, KeyError):
                continue
        print(f"  {season}: {len(data or [])} ASA rows, {hit} matched to our games")
    return out


# ---------------------------------------------------------------------------
# 2. build the candidate features, causally
# ---------------------------------------------------------------------------
def build_xg_features(finished, xg):
    """One forward pass. Every value for game i uses only games BEFORE i.

    Returns (feature_dict_list, coverage_fraction).
    """
    xg_for, xg_against = {}, {}      # team -> rolling list
    goals_for = {}                   # team -> rolling list (for over-performance)
    xpoints = {}                     # team -> rolling list
    rows, covered = [], 0

    for g in finished:
        h, a = g["home_team_id"], g["away_team_id"]

        def m(d, t):
            v = d.get(t, [])[-FORM_N:]
            return float(np.mean(v)) if v else None

        hxg, hxga = m(xg_for, h), m(xg_against, h)
        axg, axga = m(xg_for, a), m(xg_against, a)
        hgf, agf = m(goals_for, h), m(goals_for, a)
        hxp, axp = m(xpoints, h), m(xpoints, a)

        # net xG differential: (what I create - what I concede), home minus away
        if None not in (hxg, hxga, axg, axga):
            xg_form_diff = (hxg - hxga) - (axg - axga)
        else:
            xg_form_diff = 0.0

        # over-performance: scoring above your xG is usually luck, and it regresses
        if None not in (hgf, hxg, agf, axg):
            xg_overperf_diff = (hgf - hxg) - (agf - axg)
        else:
            xg_overperf_diff = 0.0

        xpoints_form_diff = (hxp - axp) if None not in (hxp, axp) else 0.0

        rows.append({"xg_form_diff": xg_form_diff,
                     "xg_overperf_diff": xg_overperf_diff,
                     "xpoints_form_diff": xpoints_form_diff})

        rec = xg.get(g["id"])
        if rec:
            covered += 1
            xg_for.setdefault(h, []).append(rec["hxg"])
            xg_against.setdefault(h, []).append(rec["axg"])
            xg_for.setdefault(a, []).append(rec["axg"])
            xg_against.setdefault(a, []).append(rec["hxg"])
            goals_for.setdefault(h, []).append(rec["hg"])
            goals_for.setdefault(a, []).append(rec["ag"])
            xpoints.setdefault(h, []).append(rec["hxp"])
            xpoints.setdefault(a, []).append(rec["axp"])

    return rows, (covered / len(finished) if finished else 0.0)


# ---------------------------------------------------------------------------
# 3. the same Poisson + walk-forward evaluation the offline study used
# ---------------------------------------------------------------------------
def poisson_fit(A, y):
    """log(lambda) = Xb by direct MLE. Mirrors the offline study's PoissonGLM."""
    from scipy import optimize
    Xd = np.column_stack([np.ones(len(A)), A])

    def nll(b):
        eta = np.clip(Xd @ b, -20, 20)
        return -np.sum(y * eta - np.exp(eta)) + 1e-3 * np.sum(b[1:] ** 2)

    def grad(b):
        eta = np.clip(Xd @ b, -20, 20)
        gr = -Xd.T @ (y - np.exp(eta))
        gr[1:] += 2e-3 * b[1:]
        return gr

    b0 = np.zeros(Xd.shape[1])
    b0[0] = np.log(max(y.mean(), 1e-3))
    return optimize.minimize(nll, b0, jac=grad, method="L-BFGS-B").x


def three_way(lam, mu, maxg=12):
    from scipy import special
    ks = np.arange(maxg + 1)
    ph = np.exp(-lam[:, None] + ks * np.log(lam[:, None] + 1e-12) - special.gammaln(ks + 1))
    pa = np.exp(-mu[:, None] + ks * np.log(mu[:, None] + 1e-12) - special.gammaln(ks + 1))
    J = ph[:, :, None] * pa[:, None, :]
    ix = np.arange(maxg + 1)
    p = np.column_stack([J[:, ix[:, None] < ix[None, :]].sum(1),
                         J[:, ix[:, None] == ix[None, :]].sum(1),
                         J[:, ix[:, None] > ix[None, :]].sum(1)])
    return p / p.sum(1, keepdims=True)


def evaluate(X, y, hs, as_, label, out):
    P, Y = [], []
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        s = StandardScaler().fit(X[tr])
        A, B = s.transform(X[tr]), s.transform(X[te])
        bh, ba = poisson_fit(A, hs[tr]), poisson_fit(A, as_[tr])
        Bd = np.column_stack([np.ones(len(B)), B])
        P.append(three_way(np.exp(np.clip(Bd @ bh, -20, 20)),
                           np.exp(np.clip(Bd @ ba, -20, 20))))
        Y.append(y[te])
    p, y3 = np.vstack(P), np.concatenate(Y)
    oh = np.zeros_like(p)
    oh[np.arange(len(y3)), y3] = 1
    out.append({"label": label, "n": X.shape[1],
                "acc": float(np.mean(p.argmax(1) == y3)),
                "brier": float(np.mean(np.sum((p - oh) ** 2, axis=1)))})
    return p, y3


def main():
    sb = get_supabase()
    print("Loading games...")
    games = fetch_all(
        sb, "games",
        "id,date_time_utc,home_team_id,away_team_id,venue_id,home_score,away_score,"
        "status,season_name", order_col="date_time_utc")
    for g in games:
        g["date_time_utc"] = datetime.datetime.fromisoformat(
            g["date_time_utc"].replace("Z", "+00:00"))
    finished = [g for g in games if g["status"] == "final"
                and g["home_score"] is not None and g["away_score"] is not None]
    print(f"  {len(finished)} finished games")

    print("Fetching game-level xG from ASA...")
    xg = load_game_xg(sb)
    if not xg:
        print("\nNo game-level xG matched. Nothing to test — check asa_game_id joins.")
        return 1

    print("Building candidate features...")
    xrows, cov = build_xg_features(finished, xg)
    print(f"  xG coverage: {cov*100:.1f}% of finished games")
    if cov < 0.5:
        print("  WARNING: under half the games have xG. Results below are unreliable.")

    # rebuild the model's existing features by importing predict.py's own code,
    # so this is a like-for-like comparison rather than a reimplementation
    import predict as pr
    xg_prior_fn, overall_avg = pr._make_xg_prior_lookup(sb, finished)
    venues = {v["id"]: v for v in fetch_all(sb, "venues", "id,latitude,longitude,is_turf")}
    draw_rate = float(np.mean([1.0 if g["home_score"] == g["away_score"] else 0.0
                               for g in finished]))
    base_rows, *_ = pr._build_features(finished, 30, 90, xg_prior_fn, overall_avg,
                                       venues, draw_rate)
    assert len(base_rows) == len(xrows) == len(finished)

    Xb = np.array([[r[f] for f in BASE_FEATS] for r in base_rows], float)
    y = np.array([r["label"] for r in base_rows])
    hs = np.array([g["home_score"] for g in finished], float)
    as_ = np.array([g["away_score"] for g in finished], float)
    NEW = ["xg_form_diff", "xg_overperf_diff", "xpoints_form_diff"]
    Xn = np.array([[r[f] for f in NEW] for r in xrows], float)

    out = []
    evaluate(Xb, y, hs, as_, "BASELINE — current 8 features", out)
    for i, f in enumerate(NEW):
        evaluate(np.column_stack([Xb, Xn[:, [i]]]), y, hs, as_, f"+ {f}", out)
    evaluate(np.column_stack([Xb, Xn]), y, hs, as_, "+ all three at once", out)

    # cheaper alternative: swap out the stale prior-season xG instead of adding
    j = BASE_FEATS.index("xg_prior_diff")
    for i, f in enumerate(NEW):
        Xs = Xb.copy()
        Xs[:, j] = Xn[:, i]
        evaluate(Xs, y, hs, as_, f"REPLACE xg_prior_diff with {f}", out)

    b = out[0]
    print("\n" + "=" * 78)
    print(f"{'feature set':<42}{'feats':>6}{'acc':>8}{'d acc':>8}{'d brier':>10}")
    print("-" * 78)
    for r in out:
        print(f"{r['label']:<42}{r['n']:>6}{r['acc']*100:>7.1f}%"
              f"{(r['acc']-b['acc'])*100:>+8.1f}{r['brier']-b['brier']:>+10.4f}")
    print("=" * 78)
    print("\nHow to read this: the offline study's error bar is about +/-3 points, so")
    print("treat anything under +1.0 as noise. d brier NEGATIVE = better calibrated,")
    print("and that is the number worth caring about for a dashboard of percentages.")
    print("Ship a feature only if it helps on its own AND doesn't hurt Brier.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
