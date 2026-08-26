"""Fitted, calibrated baseline predictors with a common interface.

Every baseline here exposes ``fit(train) -> self`` and ``predict(data) ->
(score, lower, upper)``, so the evaluation code cannot accidentally treat one of
them more favourably than another. Two things are shared deliberately:

* **Score calibration.** A DTW distance and a boosted-tree output live on
  different scales; both are mapped onto the quality scale using only the
  training split. Isotonic regression is used for the DTW distance because the
  relationship is monotone but not linear, and monotone calibration is exactly
  what a rank metric is invariant to -- so this cannot inflate Spearman, only
  make relative L2 meaningful.
* **Uncertainty.** Both get *split-conformal* intervals (Vovk et al.; Lei et al.,
  2018): the ``1 - alpha`` quantile of absolute residuals on a held-out
  calibration slice, applied symmetrically. That gives the baselines a
  distribution-free interval with a finite-sample coverage guarantee, which is a
  stronger uncertainty story than the neural model's quantile head has. If the
  neural intervals do not beat conformal ones on width at equal coverage, the
  honest conclusion is that the quantile head is not earning its place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from ..data.dataset import SequenceDataset
from ..data.generator import GeneratorConfig, reference_sequence
from .dtw import dtw_similarity, framewise_distance
from .kinematic import feature_matrix


def conformal_halfwidth(residuals: np.ndarray, level: float = 0.90) -> float:
    """Split-conformal interval half-width from calibration residuals.

    Uses the ``ceil((n+1)(1-alpha)) / n`` empirical quantile, which is the finite-
    sample-valid version; the plain ``1-alpha`` quantile under-covers slightly at
    small ``n``, and small ``n`` is the regime this project runs in.
    """
    r = np.asarray(residuals, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    q = min(1.0, np.ceil((r.size + 1) * level) / r.size)
    return float(np.quantile(r, q))


@dataclass
class DTWBaseline:
    """Reference-based DTW similarity, calibrated to the quality scale.

    Attributes:
        distance: Frame distance function.
        band: Sakoe-Chiba band fraction, or ``None``.
        aligned: ``False`` uses the frame-wise variant with no warping.
        level: Nominal interval coverage.
    """

    distance: str = "per_joint_norm"
    band: float | None = 0.15
    aligned: bool = True
    level: float = 0.90
    #: ``"canonical"`` uses the generator's ideal, style-neutral execution.
    #: ``"exemplar"`` uses the medoid of the *clean training sequences* of that
    #: action -- a real reference performance by a real subject. The second is
    #: much the stronger choice and is the default, because a canonical reference
    #: makes every inter-subject style difference look like a defect, and
    #: reporting only that version would be a strawman.
    reference_source: str = "exemplar"
    #: Fit one isotonic calibrator per action class. Distances are not comparable
    #: across actions (a throw and a gait cycle have different spatial extents),
    #: so a single global map conflates "which action" with "how good" and costs
    #: the baseline most of its signal. Off only for the ablation that shows it.
    per_action_calibration: bool = True
    _calibrators: dict[str, IsotonicRegression] = field(default_factory=dict, repr=False)
    _halfwidth: float = field(default=float("nan"), repr=False)
    _references: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    generator: GeneratorConfig | None = None

    @property
    def name(self) -> str:
        tag = "dtw" if self.aligned else "framewise"
        band = "unbanded" if self.band is None else f"band{self.band:g}"
        return f"{tag}_{self.distance}_{band}_{self.reference_source}"

    def _reference_for(self, action: str) -> np.ndarray:
        cfg = self.generator or GeneratorConfig()
        if action not in self._references:
            self._references[action] = reference_sequence(action, cfg)
        return self._references[action]

    def _fit_references(self, train: SequenceDataset, max_candidates: int = 12) -> None:
        """Pick a medoid clean exemplar per action from the training split."""
        if self.reference_source != "exemplar":
            return
        actions = np.asarray(train.actions)
        quality = np.asarray(train.quality)
        for action in sorted(set(actions.tolist())):
            pool = np.where((actions == action) & (quality >= 0.999))[0]
            if pool.size == 0:  # no defect-free example: fall back to the best one
                pool = np.where(actions == action)[0]
                pool = pool[np.argsort(-quality[pool])][:max_candidates]
            pool = pool[:max_candidates]
            seqs = [train.samples[i].positions for i in pool]
            if len(seqs) == 1:
                self._references[action] = seqs[0]
                continue
            cost = np.zeros(len(seqs))
            for i, a in enumerate(seqs):
                for j, b in enumerate(seqs):
                    if i < j:
                        d = dtw_similarity(a, b, self.distance, self.band)
                        cost[i] += d
                        cost[j] += d
            self._references[action] = seqs[int(np.argmin(cost))]

    def raw_distances(self, data: SequenceDataset) -> np.ndarray:
        """``(N,)`` distance to the reference execution of each item's action."""
        out = np.empty(len(data), dtype=np.float64)
        for i, sample in enumerate(data.samples):
            ref = self._reference_for(sample.action)
            seq = sample.positions
            out[i] = (
                dtw_similarity(seq, ref, self.distance, self.band)
                if self.aligned
                else framewise_distance(seq, ref, self.distance)
            )
        return out

    def fit(self, train: SequenceDataset, calib_fraction: float = 0.3,
            seed: int = 0) -> DTWBaseline:
        """Fit the monotone distance->quality map and the conformal half-width.

        The calibration slice for the interval is disjoint from the slice used to
        fit the isotonic map; reusing one slice for both would make the conformal
        coverage guarantee void.
        """
        self._fit_references(train)
        d = self.raw_distances(train)
        y = np.asarray(train.quality, dtype=np.float64)
        actions = np.asarray(train.actions)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(d.size)
        n_cal = max(2, int(round(calib_fraction * d.size)))
        cal, fit = perm[:n_cal], perm[n_cal:]

        groups = sorted(set(actions.tolist())) if self.per_action_calibration else ["__all__"]
        for g in groups:
            m = np.ones(d.size, dtype=bool) if g == "__all__" else actions == g
            idx = np.array([i for i in fit if m[i]])
            if idx.size < 2:  # not enough to fit a monotone map for this action
                idx = fit
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
            iso.fit(d[idx], y[idx])
            self._calibrators[g] = iso

        self._halfwidth = conformal_halfwidth(
            np.abs(self._apply(d[cal], actions[cal]) - y[cal]), self.level
        )
        return self

    def _apply(self, distances: np.ndarray, actions: np.ndarray) -> np.ndarray:
        out = np.empty(distances.size, dtype=np.float64)
        for i, (dist, act) in enumerate(zip(distances, actions, strict=True)):
            iso = self._calibrators.get(str(act)) or self._calibrators.get("__all__")
            if iso is None:
                iso = next(iter(self._calibrators.values()))
            out[i] = float(iso.predict([dist])[0])
        return out

    def predict(self, data: SequenceDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(score, lower, upper)``.

        Raises:
            RuntimeError: if called before :meth:`fit`.
        """
        if not self._calibrators:
            raise RuntimeError("DTWBaseline.predict called before fit")
        score = self._apply(self.raw_distances(data), np.asarray(data.actions))
        return score, score - self._halfwidth, score + self._halfwidth


@dataclass
class KinematicGBRBaseline:
    """Handcrafted kinematic features into gradient boosting.

    Attributes:
        n_estimators: Boosting rounds.
        max_depth: Tree depth.
        learning_rate: Shrinkage.
        level: Nominal interval coverage.
        quantile_intervals: If ``True``, fit two extra quantile-loss regressors
            for the interval instead of using a conformal half-width. Both are
            available because a constant-width conformal interval and an
            input-dependent quantile interval answer different questions, and the
            width-conditional-on-error table needs the latter to be interesting.
    """

    n_estimators: int = 300
    max_depth: int = 3
    learning_rate: float = 0.05
    level: float = 0.90
    quantile_intervals: bool = True
    seed: int = 0
    _model: GradientBoostingRegressor | None = field(default=None, repr=False)
    _lo: GradientBoostingRegressor | None = field(default=None, repr=False)
    _hi: GradientBoostingRegressor | None = field(default=None, repr=False)
    _halfwidth: float = field(default=float("nan"), repr=False)
    _names: list[str] = field(default_factory=list, repr=False)
    action_classes: tuple[str, ...] = ()

    name = "kinematic_gbr"

    def _features(self, data: SequenceDataset) -> np.ndarray:
        x, names = feature_matrix(
            data.coords, list(data.actions), self.action_classes
        )
        self._names = names
        return x

    def fit(self, train: SequenceDataset, calib_fraction: float = 0.25) -> KinematicGBRBaseline:
        x = self._features(train)
        y = np.asarray(train.quality, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(x.shape[0])
        n_cal = max(2, int(round(calib_fraction * x.shape[0])))
        cal, fit = perm[:n_cal], perm[n_cal:]

        common = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.seed,
        }
        self._model = GradientBoostingRegressor(**common).fit(x[fit], y[fit])
        self._halfwidth = conformal_halfwidth(
            np.abs(self._model.predict(x[cal]) - y[cal]), self.level
        )
        if self.quantile_intervals:
            alpha = (1.0 - self.level) / 2.0
            self._lo = GradientBoostingRegressor(loss="quantile", alpha=alpha,
                                                 **common).fit(x[fit], y[fit])
            self._hi = GradientBoostingRegressor(loss="quantile", alpha=1.0 - alpha,
                                                 **common).fit(x[fit], y[fit])
        return self

    def predict(self, data: SequenceDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("KinematicGBRBaseline.predict called before fit")
        x = self._features(data)
        score = self._model.predict(x)
        if self._lo is not None and self._hi is not None:
            lo, hi = self._lo.predict(x), self._hi.predict(x)
            # Quantile regressors are fitted independently and *can* cross; the
            # neural head cannot. Rather than hide it, the crossings are fixed by
            # sorting and the rate is reported by the metrics module.
            return score, np.minimum(lo, hi), np.maximum(lo, hi)
        return score, score - self._halfwidth, score + self._halfwidth

    def feature_importance(self) -> dict[str, float]:
        """Gini importance per named feature, descending.

        This is the baseline's *explanation*, and it is a global one: it says
        which features matter on average, never which joint failed in *this*
        execution. That asymmetry is the attribution argument of the project, so
        the number is reported and its scope is stated.
        """
        if self._model is None:
            raise RuntimeError("feature_importance called before fit")
        pairs = sorted(
            zip(self._names, self._model.feature_importances_, strict=True),
            key=lambda kv: -kv[1],
        )
        return {k: float(v) for k, v in pairs}
