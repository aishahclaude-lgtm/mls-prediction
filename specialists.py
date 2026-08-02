"""Three specialist models, each the measured best at one job.

This is NOT an ensemble in the usual sense. An ensemble is many models voting on
the same question. Here each model answers a DIFFERENT question, because the
offline study found no single model was best at all three:

    WIN / LOSS      a 2-class home-vs-away logistic trained on ALL games, with
                    each draw counted as half-evidence for each side.
                    AUC 0.627 -- the best pure home/away discrimination measured.
                    Beats training on decisive games only (0.585), because
                    dropping draws throws away 23% of the data AND the games
                    that most clearly say "these two are evenly matched".

    DRAWS           Dixon-Coles with per-team attack/defence + a low-score
                    dependence term (rho) + exponential time decay. The only
                    model measured that calls draws better than the base rate
                    (34.2% precision vs a 23.2% base rate). Suggestive, not
                    proven -- p = 0.08.

    PERCENTAGES     the Poisson goal model. Brier 0.6243 -- the best-calibrated
                    SINGLE model measured. It also owns the draw-risk ranking
                    (AUC 0.546, the best draw ranker), because it models goals
                    and lets P(draw) fall out as "both sides land on the same
                    number" rather than predicting draws directly.

Deliberately NOT averaged. Pooling the three does score better on calibration
(Brier 0.6206), but that is ensembling -- one blended number nobody owns. Each
lane here is answered by the single model measured best at that lane, so every
figure on the dashboard traces to one identifiable model. The cost of that
choice is about 0.004 of Brier.

Everything here was validated by walk-forward (chronological) backtest on
~1,100 out-of-sample games. Nothing is tuned on data it was scored against.
"""
import numpy as np
from scipy import optimize, special
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

MAXG = 12
# The 8 features the Poisson stage uses (goal-rate drivers only).
POISSON_FEATURES = ["elo_diff", "home_form_ppg", "away_form_ppg", "home_form_gd",
                    "away_form_gd", "xg_prior_diff", "venue_ppg", "h2h_home_ppg"]


# ---------------------------------------------------------------------------
# Poisson goal model
# ---------------------------------------------------------------------------
class _PoissonGLM:
    """log(lambda) = X @ beta, fitted by direct MLE."""

    def fit(self, X, y, l2=1e-3):
        Xd = np.column_stack([np.ones(len(X)), X])

        def nll(b):
            eta = np.clip(Xd @ b, -20, 20)
            return -np.sum(y * eta - np.exp(eta)) + l2 * np.sum(b[1:] ** 2)

        def grad(b):
            eta = np.clip(Xd @ b, -20, 20)
            g = -Xd.T @ (y - np.exp(eta))
            g[1:] += 2 * l2 * b[1:]
            return g

        b0 = np.zeros(Xd.shape[1])
        b0[0] = np.log(max(float(np.mean(y)), 1e-3))
        self.beta_ = optimize.minimize(nll, b0, jac=grad, method="L-BFGS-B").x
        return self

    def rate(self, X):
        Xd = np.column_stack([np.ones(len(X)), X])
        return np.exp(np.clip(Xd @ self.beta_, -20, 20))


def _poisson_matrix(lam, mu):
    ks = np.arange(MAXG + 1)
    ph = np.exp(-lam[:, None] + ks * np.log(lam[:, None] + 1e-12) - special.gammaln(ks + 1))
    pa = np.exp(-mu[:, None] + ks * np.log(mu[:, None] + 1e-12) - special.gammaln(ks + 1))
    return ph[:, :, None] * pa[:, None, :]


def _three_way(J):
    ix = np.arange(J.shape[1])
    p = np.column_stack([J[:, ix[:, None] < ix[None, :]].sum(1),
                         J[:, ix[:, None] == ix[None, :]].sum(1),
                         J[:, ix[:, None] > ix[None, :]].sum(1)])
    return p / p.sum(1, keepdims=True)


# ---------------------------------------------------------------------------
# Dixon-Coles with team parameters + time decay
# ---------------------------------------------------------------------------
def _dc_tau(x, y, lam, mu, rho):
    t = np.ones_like(lam, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


class _DixonColes:
    """log lam = att_home - def_away + gamma ; log mu = att_away - def_home."""

    def __init__(self, n_teams, xi=0.0018, l2=1e-3):
        self.n, self.xi, self.l2 = n_teams, xi, l2

    def _unpack(self, th):
        n = self.n
        att = np.concatenate([th[:n - 1], [-np.sum(th[:n - 1])]])
        return att, th[n - 1:2 * n - 1], th[2 * n - 1], np.tanh(th[2 * n]) * 0.9

    def fit(self, hi, ai, hs, as_, days_ago):
        n = self.n
        w = np.exp(-self.xi * np.asarray(days_ago, float))

        def nll(th):
            att, dfn, gamma, rho = self._unpack(th)
            lam = np.exp(np.clip(att[hi] - dfn[ai] + gamma, -6, 6))
            mu = np.exp(np.clip(att[ai] - dfn[hi], -6, 6))
            t = _dc_tau(hs, as_, lam, mu, rho)
            if np.any(t <= 0):
                return 1e10
            ll = (np.log(t) - lam + hs * np.log(lam + 1e-12)
                  - mu + as_ * np.log(mu + 1e-12))
            return -np.sum(w * ll) + self.l2 * (np.sum(att ** 2) + np.sum(dfn ** 2))

        th0 = np.zeros(2 * n + 1)
        th0[2 * n - 1] = 0.25
        r = optimize.minimize(nll, th0, method="L-BFGS-B",
                              options={"maxiter": 4000, "maxfun": 40000})
        self.att_, self.dfn_, self.gamma_, self.rho_ = self._unpack(r.x)
        return self

    def predict_proba(self, hi, ai):
        lam = np.exp(np.clip(self.att_[hi] - self.dfn_[ai] + self.gamma_, -6, 6))
        mu = np.exp(np.clip(self.att_[ai] - self.dfn_[hi], -6, 6))
        J = _poisson_matrix(lam, mu)
        ks = np.arange(MAXG + 1)
        xx, yy = np.meshgrid(ks, ks, indexing="ij")
        L, M = lam[:, None, None], mu[:, None, None]
        T = np.ones_like(J)
        T = np.where((xx == 0) & (yy == 0), 1 - L * M * self.rho_, T)
        T = np.where((xx == 0) & (yy == 1), 1 + L * self.rho_, T)
        T = np.where((xx == 1) & (yy == 0), 1 + M * self.rho_, T)
        T = np.where((xx == 1) & (yy == 1), 1 - self.rho_, T)
        J = J * np.clip(T, 1e-12, None)
        return _three_way(J / J.sum(axis=(1, 2), keepdims=True))


# ---------------------------------------------------------------------------
# the committee
# ---------------------------------------------------------------------------
class Specialists:
    """Fit all three, then answer each question with the model that owns it."""

    def __init__(self, feature_names):
        self.feature_names = list(feature_names)
        self.pois_ix = [self.feature_names.index(f) for f in POISSON_FEATURES]
        self.ok = False

    def fit(self, X, y, hs, as_, home_idx, away_idx, days_ago, n_teams):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.scaler_ = StandardScaler().fit(X)
        A = self.scaler_.transform(X)

        # --- Poisson ------------------------------------------------------
        self.ph_ = _PoissonGLM().fit(A[:, self.pois_ix], hs)
        self.pa_ = _PoissonGLM().fit(A[:, self.pois_ix], as_)

        # --- Dixon-Coles --------------------------------------------------
        self.dc_ = _DixonColes(n_teams).fit(home_idx, away_idx, hs, as_, days_ago)

        # --- multinomial logistic (calibrated) ----------------------------
        inner = TimeSeriesSplit(n_splits=min(3, max(2, len(X) // 30)))
        try:
            self.lr_ = CalibratedClassifierCV(
                LogisticRegression(max_iter=2000, C=3.0),
                method="sigmoid", cv=inner).fit(A, y)
        except ValueError:
            self.lr_ = LogisticRegression(max_iter=2000, C=3.0).fit(A, y)
        self.lr_ix_ = [list(self.lr_.classes_).index(c) for c in (0, 1, 2)]

        # --- W/L specialist: draws split 50/50 between the two sides -------
        # Duplicating the drawn rows with the opposite label, each at weight
        # 0.5, is what lets a 2-class model use them instead of discarding
        # them. Measured AUC 0.627 vs 0.585 for decisive-games-only.
        drawn = (y == 1)
        Xw = np.vstack([A, A[drawn]])
        yw = np.concatenate([(y == 2).astype(int), np.ones(int(drawn.sum()), int)])
        ww = np.concatenate([np.where(drawn, 0.5, 1.0), np.full(int(drawn.sum()), 0.5)])
        self.wl_ = LogisticRegression(max_iter=2000, C=1.0).fit(Xw, yw, sample_weight=ww)

        # --- draw-risk cutpoints, from TRAINING games only -----------------
        # Terciles of the Poisson's P(draw) -- measured 19.2% / 24.9% / 25.5%
        # actual draw rate across the three bands.
        tr_pois = _three_way(_poisson_matrix(self.ph_.rate(A[:, self.pois_ix]),
                                             self.pa_.rate(A[:, self.pois_ix])))
        self.draw_q_ = np.quantile(tr_pois[:, 1], [1 / 3, 2 / 3])
        self.ok = True
        return self

    def predict(self, feat_row, home_idx, away_idx):
        """One game in, a dict of every specialist's answer out."""
        A = self.scaler_.transform(np.asarray(feat_row, float).reshape(1, -1))
        hi = np.array([home_idx])
        ai = np.array([away_idx])

        pois = _three_way(_poisson_matrix(self.ph_.rate(A[:, self.pois_ix]),
                                          self.pa_.rate(A[:, self.pois_ix])))[0]
        dc = self.dc_.predict_proba(hi, ai)[0]
        lr = self.lr_.predict_proba(A)[:, self.lr_ix_][0]

        p_home_decisive = float(self.wl_.predict_proba(A)[0, 1])
        # draw-risk band comes from the Poisson, the measured best draw ranker
        band = ("Low" if pois[1] < self.draw_q_[0]
                else ("High" if pois[1] >= self.draw_q_[1] else "Medium"))

        return {
            # displayed percentages -- the Poisson owns this lane
            "probs": [float(pois[0]), float(pois[1]), float(pois[2])],
            # the home/away call -- best discrimination we have
            "wl_pick": "home" if p_home_decisive >= 0.5 else "away",
            "wl_home_pct": p_home_decisive,
            # draw lane -- Dixon-Coles is the only model that calls draws
            # above the base rate, so it gets its own explicit flag
            "draw_risk": band,
            "dc_draw_pct": float(dc[1]),
            "dc_calls_draw": bool(dc.argmax() == 1),
            "components": {
                "poisson": [round(float(v), 4) for v in pois],
                "dixon_coles": [round(float(v), 4) for v in dc],
                "logistic": [round(float(v), 4) for v in lr],
            },
        }
