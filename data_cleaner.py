"""Data Cleaner (Module 2A) - analysis and recommendations for ML pipelines.

This module provides the ``DataCleaner`` class, which consumes an input
DataFrame together with the upstream output of
``DataExplorer.get_pipeline_summary()`` to produce auditable cleaning
recommendations before any destructive transformations are applied.

The goal of Part 1 (this module) is to analyze and recommend. Execution of
transformations belongs to a downstream step (Part 2B).

Key design principles
---------------------
- Explainable and auditable recommendations.
- Decision-level logging with timestamped records.
- Dry-run safety by default (``dry_run=True``).
- Graceful behavior for empty/small edge cases.

Usage example
-------------
>>> import pandas as pd
>>> from data_cleaner import DataCleaner
>>> df = pd.read_csv("dataset.csv")
>>> pipeline_summary = {"numeric_cols": ["x1", "x2"], "target_col": "y"}
>>> cleaner = DataCleaner(df, pipeline_summary=pipeline_summary, target_col="y")
>>> outputs = cleaner.run_full_analysis()
>>> report = outputs["recommendation_report"]

Pipeline context
----------------
Typical flow in a modular pipeline:
``DataExplorer`` -> ``DataCleaner`` (this module, Part 2A) -> execution module
(Part 2B) -> feature engineering -> modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any
import warnings

import numpy as np
import pandas as pd


class CleaningError(Exception):
    """Raised for critical cleaning analysis failures."""


class CleaningWarning(UserWarning):
    """Warning used for non-critical cleaning analysis issues."""


@dataclass(frozen=True)
class _NullRec:
    """Internal typed holder for null analysis recommendations."""

    column: str
    null_pct: float
    action: str
    imputation_strategy: str | None
    reason: str


class DataCleaner:
    """Analyze data quality and create auditable cleaning recommendations.

    Args:
        df: Input dataset.
        pipeline_summary: Output dictionary from
            ``DataExplorer.get_pipeline_summary()``. If ``None``, basic schema
            metadata is recomputed internally.
        target_col: Target variable name.
        id_col: Identifier column name.
        dry_run: If ``True``, analyze/recommend only and never mutate data.
        correlation_method: Correlation method for numeric analyses.
            Supported values: ``'pearson'`` or ``'spearman'``.
        verbose: If ``True``, print detailed progress and summaries.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If ``correlation_method`` is invalid.
    """

    _ALLOWED_SEVERITIES: tuple[str, ...] = (
        "AUTO_REMOVE",
        "RECOMMEND_REMOVE",
        "RECOMMEND_GROUP",
        "RECOMMEND_ENCODE",
        "RECOMMEND_IMPUTE",
        "RECOMMEND_REVIEW",
        "INFO",
    )

    def __init__(
        self,
        df: pd.DataFrame,
        pipeline_summary: dict[str, Any] | None = None,
        target_col: str | None = None,
        id_col: str | None = None,
        dry_run: bool = True,
        correlation_method: str = "pearson",
        verbose: bool = True,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        method = correlation_method.lower().strip()
        if method not in ("pearson", "spearman"):
            raise ValueError("correlation_method must be 'pearson' or 'spearman'")

        self.df_original = df.copy(deep=True)
        self.df = df.copy(deep=True)
        self.pipeline_summary = pipeline_summary or self._build_basic_pipeline_summary()

        # Explicit args take precedence over summary values.
        self.target_col = target_col if target_col is not None else self.pipeline_summary.get("target_col")
        self.id_col = id_col if id_col is not None else self.pipeline_summary.get("id_col")

        self.dry_run = dry_run
        self.correlation_method = method
        self.verbose = verbose

        self.decision_log: list[dict[str, Any]] = []

        # Caches for cross-method reuse.
        self._null_cache: pd.DataFrame | None = None
        self._cardinality_cache: pd.DataFrame | None = None
        self._correlation_cache: pd.DataFrame | None = None
        self._outlier_cache: pd.DataFrame | None = None

    def _print(self, *args: Any, **kwargs: Any) -> None:
        """Print helper controlled by ``verbose``.

        Args:
            *args: Positional print arguments.
            **kwargs: Keyword print arguments.

        Returns:
            None.

        Raises:
            None.
        """
        if self.verbose:
            print(*args, **kwargs)

    def _warn(self, message: str) -> None:
        """Emit a ``CleaningWarning``.

        Args:
            message: Warning message.

        Returns:
            None.

        Raises:
            None.
        """
        warnings.warn(message, CleaningWarning, stacklevel=3)

    def _build_basic_pipeline_summary(self) -> dict[str, Any]:
        """Build a minimal pipeline summary when upstream summary is missing.

        Args:
            None.

        Returns:
            Dictionary with inferred type groups and quality primitives.

        Raises:
            None.
        """
        df = self.df
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        boolean_cols = df.select_dtypes(include="bool").columns.tolist()
        datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

        categorical_cols: list[str] = []
        freetext_cols: list[str] = []
        for col in df.columns:
            if col in numeric_cols or col in boolean_cols or col in datetime_cols:
                continue
            if df[col].dtype.name == "category":
                categorical_cols.append(col)
            elif df[col].dtype == object:
                nunique = df[col].nunique(dropna=True)
                n_nonnull = max(df[col].count(), 1)
                if nunique / n_nonnull > 0.5:
                    freetext_cols.append(col)
                else:
                    categorical_cols.append(col)

        return {
            "shape": tuple(df.shape),
            "null_columns": df.columns[df.isnull().any()].tolist(),
            "high_null_columns": df.columns[df.isnull().mean() > 0.30].tolist(),
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "boolean_cols": boolean_cols,
            "freetext_cols": freetext_cols,
            "low_variance_cols": [],
            "outlier_cols": [],
            "non_normal_cols": [],
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_ids": None,
            "target_col": None,
            "id_col": None,
            "is_tidy": True,
            "is_time_series": False,
            "alerts": pd.DataFrame(),
        }

    def _numeric_cols(self, exclude_target: bool = False, exclude_id: bool = True) -> list[str]:
        """Return numeric columns with optional exclusions.

        Args:
            exclude_target: Whether to exclude ``target_col``.
            exclude_id: Whether to exclude ``id_col``.

        Returns:
            Numeric column names.

        Raises:
            None.
        """
        cols = self.df.select_dtypes(include="number").columns.tolist()
        if exclude_id and self.id_col and self.id_col in cols:
            cols.remove(self.id_col)
        if exclude_target and self.target_col and self.target_col in cols:
            cols.remove(self.target_col)
        return cols

    def _categorical_cols(self, exclude_target: bool = True, exclude_id: bool = True) -> list[str]:
        """Return categorical/object columns with optional exclusions.

        Args:
            exclude_target: Whether to exclude ``target_col``.
            exclude_id: Whether to exclude ``id_col``.

        Returns:
            Categorical-like column names.

        Raises:
            None.
        """
        cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        if exclude_id and self.id_col and self.id_col in cols:
            cols.remove(self.id_col)
        if exclude_target and self.target_col and self.target_col in cols:
            cols.remove(self.target_col)
        return cols

    def _log_decision(
        self,
        action: str,
        columns: list[str],
        reason: str,
        severity: str,
        executed: bool,
        metric: float | None = None,
    ) -> None:
        """Append a cleaning recommendation/action to the audit decision log.

        Args:
            action: Short action label (for example ``'drop_column'``).
            columns: Columns affected by this decision.
            reason: Human-readable justification.
            severity: Decision severity level. Must be one of:
                ``AUTO_REMOVE``, ``RECOMMEND_REMOVE``, ``RECOMMEND_GROUP``,
                ``RECOMMEND_ENCODE``, ``RECOMMEND_IMPUTE``,
                ``RECOMMEND_REVIEW``, ``INFO``.
            executed: Whether the action was executed.
            metric: Optional numeric metric associated with the decision.

        Returns:
            None.

        Raises:
            CleaningError: If ``severity`` is invalid.
        """
        if severity not in self._ALLOWED_SEVERITIES:
            raise CleaningError(
                f"Invalid severity '{severity}'. Allowed: {self._ALLOWED_SEVERITIES}"
            )

        self.decision_log.append(
            {
                "action": action,
                "columns": columns,
                "reason": reason,
                "severity": severity,
                "metric": metric,
                "executed": executed,
                "timestamp": pd.Timestamp.utcnow().isoformat(),
            }
        )

    def _null_stats(self) -> pd.DataFrame:
        """Compute null percentages and return standardized null stats.

        Args:
            None.

        Returns:
            DataFrame with columns ``column`` and ``null_pct``.

        Raises:
            None.
        """
        if self._null_cache is not None:
            return self._null_cache.copy()

        df = self.df
        if df.empty:
            out = pd.DataFrame(columns=["column", "null_pct"])
            self._null_cache = out
            return out.copy()

        null_pct = df.isnull().mean() * 100
        out = (
            pd.DataFrame({"column": null_pct.index, "null_pct": null_pct.values})
            .sort_values("null_pct", ascending=False)
            .reset_index(drop=True)
        )
        self._null_cache = out
        return out.copy()

    def analyze_high_null_columns(
        self,
        auto_remove_threshold: float = 0.9,
        recommend_remove_threshold: float = 0.6,
    ) -> pd.DataFrame:
        """Analyze missingness and create explainable null-handling recommendations.

        Args:
            auto_remove_threshold: Fraction threshold (0-1) for
                ``AUTO_REMOVE`` recommendations.
            recommend_remove_threshold: Fraction threshold (0-1) for
                ``RECOMMEND_REMOVE`` recommendations.

        Returns:
            DataFrame with columns ``column``, ``null_pct``, ``action``,
            ``imputation_strategy``, and ``reason``.

        Raises:
            ValueError: If thresholds are outside valid ranges.
        """
        if not (0 <= recommend_remove_threshold <= 1 and 0 <= auto_remove_threshold <= 1):
            raise ValueError("thresholds must be in [0, 1]")
        if recommend_remove_threshold > auto_remove_threshold:
            raise ValueError("recommend_remove_threshold must be <= auto_remove_threshold")

        if self.df.shape[1] == 0:
            return pd.DataFrame(
                columns=["column", "null_pct", "action", "imputation_strategy", "reason"]
            )

        null_stats = self._null_stats()
        if null_stats.empty:
            return pd.DataFrame(
                columns=["column", "null_pct", "action", "imputation_strategy", "reason"]
            )

        # Prefer upstream null metadata when available, then enrich with
        # recomputed percentages for those columns.
        summary_null_cols = set(self.pipeline_summary.get("null_columns", []))
        if summary_null_cols:
            null_stats = null_stats[null_stats["column"].isin(summary_null_cols)]
            if null_stats.empty:
                # Fallback to recomputed values if upstream metadata is stale.
                null_stats = self._null_stats()

        non_normal_cols = set(self.pipeline_summary.get("non_normal_cols", []))
        is_time_series = bool(self.pipeline_summary.get("is_time_series", False))

        rows: list[_NullRec] = []

        for rec in null_stats.itertuples(index=False):
            col = str(rec.column)
            null_pct = float(rec.null_pct)
            frac = null_pct / 100.0

            if null_pct == 0:
                continue

            if frac >= auto_remove_threshold:
                reason = (
                    f"{null_pct:.1f}% nulls exceeds auto-removal threshold "
                    f"({auto_remove_threshold * 100:.1f}%)"
                )
                rows.append(_NullRec(col, null_pct, "AUTO_REMOVE", None, reason))
                self._log_decision(
                    action="drop_column",
                    columns=[col],
                    reason=reason,
                    severity="AUTO_REMOVE",
                    executed=False,
                    metric=null_pct,
                )
                continue

            if frac >= recommend_remove_threshold:
                reason = (
                    f"{null_pct:.1f}% nulls suggests potential removal "
                    f"(>= {recommend_remove_threshold * 100:.1f}%)"
                )
                rows.append(_NullRec(col, null_pct, "RECOMMEND_REMOVE", None, reason))
                self._log_decision(
                    action="consider_drop_column",
                    columns=[col],
                    reason=reason,
                    severity="RECOMMEND_REMOVE",
                    executed=False,
                    metric=null_pct,
                )
                continue

            strategy = ""
            reason = ""
            if col in self._numeric_cols(exclude_target=False, exclude_id=False):
                skew = float(self.df[col].dropna().skew()) if self.df[col].dropna().size > 2 else 0.0
                if is_time_series:
                    strategy = "forward-fill or interpolation"
                    reason = "Numeric in time-series context; preserve temporal continuity"
                elif col in non_normal_cols or abs(skew) >= 0.5:
                    strategy = "median imputation"
                    reason = f"Numeric skewed distribution (skew={skew:.2f}); median is robust"
                else:
                    strategy = "mean imputation"
                    reason = f"Numeric roughly normal (|skew|={abs(skew):.2f} < 0.5)"

                if 10 <= null_pct < 60:
                    strategy += " + evaluate model-based imputation"
                    reason += "; high-null range (10-60%)"
            else:
                strategy = "mode imputation or 'Unknown' category"
                reason = "Categorical/object feature with missing values"
                if 10 <= null_pct < 60:
                    strategy += " + evaluate model-based imputation"
                    reason += "; high-null range (10-60%)"

            rows.append(_NullRec(col, null_pct, "RECOMMEND_IMPUTE", strategy, reason))
            self._log_decision(
                action="recommend_imputation",
                columns=[col],
                reason=reason,
                severity="RECOMMEND_IMPUTE",
                executed=False,
                metric=null_pct,
            )

        result = pd.DataFrame(
            [
                {
                    "column": r.column,
                    "null_pct": round(r.null_pct, 4),
                    "action": r.action,
                    "imputation_strategy": r.imputation_strategy,
                    "reason": r.reason,
                }
                for r in rows
            ]
        )

        self._print("\n=== High Null Column Analysis ===")
        if result.empty:
            self._print("No columns with null recommendations.")
        else:
            self._print(result.to_string(index=False))
        self.null_analysis = result
        return result

    def analyze_dominant_classes(
        self,
        dominance_threshold: float = 0.95,
    ) -> pd.DataFrame:
        """Analyze dominant classes and near-constant numeric behavior.

        Args:
            dominance_threshold: Dominance ratio threshold in [0, 1] to flag
                categorical columns as removable recommendations.

        Returns:
            DataFrame with columns ``column``, ``dominant_value``,
            ``dominant_pct``, ``cv``, ``action``, and ``reason``.

        Raises:
            ValueError: If threshold is outside [0, 1].
        """
        if not (0 <= dominance_threshold <= 1):
            raise ValueError("dominance_threshold must be in [0, 1]")

        rows: list[dict[str, Any]] = []

        for col in self._categorical_cols(exclude_target=True, exclude_id=True):
            s = self.df[col].dropna()
            if s.empty:
                continue
            vc = s.value_counts(normalize=True)
            dominant_val = vc.index[0]
            dominant_pct = float(vc.iloc[0]) * 100

            action = "INFO"
            reason = "No dominant class concern"
            severity = "INFO"
            if dominant_pct / 100 >= dominance_threshold:
                action = "RECOMMEND_REMOVE"
                severity = "RECOMMEND_REMOVE"
                reason = (
                    f"Dominant class covers {dominant_pct:.1f}% of values "
                    "- near-constant variable"
                )
                self._warn(f"{col}: {reason}")
            elif dominant_pct / 100 >= 0.85:
                action = "RECOMMEND_REVIEW"
                severity = "RECOMMEND_REVIEW"
                reason = (
                    f"Single class covers {dominant_pct:.1f}% - low "
                    "discriminative power"
                )

            self._log_decision(
                action="analyze_dominance",
                columns=[col],
                reason=reason,
                severity=severity,
                executed=False,
                metric=dominant_pct,
            )

            rows.append(
                {
                    "column": col,
                    "dominant_value": str(dominant_val),
                    "dominant_pct": round(dominant_pct, 4),
                    "cv": None,
                    "action": action,
                    "reason": reason,
                }
            )

        for col in self._numeric_cols(exclude_target=True, exclude_id=True):
            s = self.df[col].dropna()
            if s.empty:
                continue
            mean = float(s.mean())
            std = float(s.std())
            if np.isclose(mean, 0.0):
                cv = 0.0 if np.isclose(std, 0.0) else np.inf
            else:
                cv = abs(std / mean)

            action = "INFO"
            reason = "Numeric variability appears acceptable"
            severity = "INFO"
            if np.isfinite(cv) and cv < 0.01:
                action = "RECOMMEND_REMOVE"
                severity = "RECOMMEND_REMOVE"
                reason = "Near-constant numeric variable (CV < 0.01)"
                self._warn(f"{col}: {reason}")

            self._log_decision(
                action="analyze_numeric_variability",
                columns=[col],
                reason=reason,
                severity=severity,
                executed=False,
                metric=float(cv) if np.isfinite(cv) else None,
            )

            rows.append(
                {
                    "column": col,
                    "dominant_value": None,
                    "dominant_pct": None,
                    "cv": None if not np.isfinite(cv) else round(float(cv), 6),
                    "action": action,
                    "reason": reason,
                }
            )

        result = pd.DataFrame(
            rows,
            columns=["column", "dominant_value", "dominant_pct", "cv", "action", "reason"],
        )
        self._print("\n=== Dominant Class / Near-Constant Analysis ===")
        if result.empty:
            self._print("No categorical or numeric columns to evaluate.")
        else:
            self._print(result.to_string(index=False))
        self.dominant_analysis = result
        return result

    def analyze_correlations(
        self,
        target_corr_remove_threshold: float = 0.95,
        target_corr_low_threshold: float = 0.02,
        inter_feature_threshold: float = 0.90,
    ) -> pd.DataFrame:
        """Analyze target correlation and inter-feature multicollinearity.

        Args:
            target_corr_remove_threshold: Absolute target correlation threshold
                above which leakage is suspected and review is recommended.
            target_corr_low_threshold: Absolute target correlation threshold
                below which low predictive value is recommended for removal.
            inter_feature_threshold: Absolute pairwise feature-correlation
                threshold above which redundancy is flagged.

        Returns:
            DataFrame with columns ``column``, ``corr_with_target``,
            ``max_inter_corr``, ``correlated_with``, ``action``, and ``reason``.

        Raises:
            ValueError: If thresholds are outside [0, 1].
        """
        for val, name in [
            (target_corr_remove_threshold, "target_corr_remove_threshold"),
            (target_corr_low_threshold, "target_corr_low_threshold"),
            (inter_feature_threshold, "inter_feature_threshold"),
        ]:
            if not (0 <= val <= 1):
                raise ValueError(f"{name} must be in [0, 1]")

        rows: list[dict[str, Any]] = []
        numeric_features = self._numeric_cols(exclude_target=True, exclude_id=True)
        if len(numeric_features) < 1:
            out = pd.DataFrame(
                columns=[
                    "column",
                    "corr_with_target",
                    "max_inter_corr",
                    "correlated_with",
                    "action",
                    "reason",
                ]
            )
            self._correlation_cache = out
            self.correlation_analysis = out
            return out

        corr_with_target: dict[str, float | None] = {c: None for c in numeric_features}

        # Target correlation analysis.
        has_target = bool(self.target_col and self.target_col in self.df.columns)
        target_numeric = False
        if has_target:
            target_numeric = pd.api.types.is_numeric_dtype(self.df[self.target_col])

        if not has_target:
            self._warn("target_col not set; skipping target correlation analysis")
        elif not target_numeric:
            self._warn("target_col is non-numeric; skipping numeric target correlation analysis")
        else:
            for col in numeric_features:
                valid = self.df[[col, self.target_col]].dropna()
                if len(valid) < 3:
                    continue
                corr = valid[col].corr(valid[self.target_col], method=self.correlation_method)
                if pd.isna(corr):
                    continue
                abs_corr = float(abs(corr))
                corr_with_target[col] = abs_corr

                if abs_corr >= target_corr_remove_threshold:
                    reason = (
                        f"Correlation {abs_corr:.3f} with target - possible "
                        "data leakage, review before modeling"
                    )
                    self._log_decision(
                        action="review_leakage_risk",
                        columns=[col],
                        reason=reason,
                        severity="RECOMMEND_REVIEW",
                        executed=False,
                        metric=abs_corr,
                    )
                    rows.append(
                        {
                            "column": col,
                            "corr_with_target": abs_corr,
                            "max_inter_corr": None,
                            "correlated_with": None,
                            "action": "RECOMMEND_REVIEW",
                            "reason": reason,
                        }
                    )
                elif abs_corr <= target_corr_low_threshold:
                    reason = "Near-zero correlation with target - low predictive value"
                    self._log_decision(
                        action="consider_drop_low_target_corr",
                        columns=[col],
                        reason=reason,
                        severity="RECOMMEND_REMOVE",
                        executed=False,
                        metric=abs_corr,
                    )
                    rows.append(
                        {
                            "column": col,
                            "corr_with_target": abs_corr,
                            "max_inter_corr": None,
                            "correlated_with": None,
                            "action": "RECOMMEND_REMOVE",
                            "reason": reason,
                        }
                    )

        # Inter-feature multicollinearity.
        if len(numeric_features) >= 2:
            corr_mat = self.df[numeric_features].corr(method=self.correlation_method).abs()
            for i, c1 in enumerate(numeric_features):
                for j in range(i + 1, len(numeric_features)):
                    c2 = numeric_features[j]
                    pair_corr = float(corr_mat.loc[c1, c2])
                    if np.isnan(pair_corr) or pair_corr < inter_feature_threshold:
                        continue

                    if has_target and target_numeric:
                        c1_t = corr_with_target.get(c1)
                        c2_t = corr_with_target.get(c2)

                        c1_t_val = -1.0 if c1_t is None else c1_t
                        c2_t_val = -1.0 if c2_t is None else c2_t

                        drop_col = c1 if c1_t_val < c2_t_val else c2
                        keep_col = c2 if drop_col == c1 else c1
                        reason = (
                            f"Correlated {pair_corr:.3f} with {keep_col} - "
                            "redundant feature"
                        )
                        self._log_decision(
                            action="consider_drop_multicollinear",
                            columns=[drop_col],
                            reason=reason,
                            severity="RECOMMEND_REMOVE",
                            executed=False,
                            metric=pair_corr,
                        )
                        rows.append(
                            {
                                "column": drop_col,
                                "corr_with_target": corr_with_target.get(drop_col),
                                "max_inter_corr": pair_corr,
                                "correlated_with": keep_col,
                                "action": "RECOMMEND_REMOVE",
                                "reason": reason,
                            }
                        )
                    else:
                        reason = (
                            f"Correlated {pair_corr:.3f} with {c2} - review pair "
                            "without target guidance"
                        )
                        self._log_decision(
                            action="review_multicollinearity",
                            columns=[c1],
                            reason=reason,
                            severity="RECOMMEND_REVIEW",
                            executed=False,
                            metric=pair_corr,
                        )
                        self._log_decision(
                            action="review_multicollinearity",
                            columns=[c2],
                            reason=reason,
                            severity="RECOMMEND_REVIEW",
                            executed=False,
                            metric=pair_corr,
                        )
                        rows.extend(
                            [
                                {
                                    "column": c1,
                                    "corr_with_target": None,
                                    "max_inter_corr": pair_corr,
                                    "correlated_with": c2,
                                    "action": "RECOMMEND_REVIEW",
                                    "reason": reason,
                                },
                                {
                                    "column": c2,
                                    "corr_with_target": None,
                                    "max_inter_corr": pair_corr,
                                    "correlated_with": c1,
                                    "action": "RECOMMEND_REVIEW",
                                    "reason": reason,
                                },
                            ]
                        )

        result = pd.DataFrame(
            rows,
            columns=[
                "column",
                "corr_with_target",
                "max_inter_corr",
                "correlated_with",
                "action",
                "reason",
            ],
        )
        self._correlation_cache = result.copy()
        self.correlation_analysis = result

        self._print("\n=== Correlation Analysis ===")
        if result.empty:
            self._print("No correlation-based recommendations.")
        else:
            self._print(result.to_string(index=False))
        return result

    def analyze_cardinality(
        self,
        high_cardinality_threshold: int = 50,
        grouping_min: int = 7,
        grouping_max: int = 30,
    ) -> pd.DataFrame:
        """Analyze categorical cardinality and encoding/grouping recommendations.

        Args:
            high_cardinality_threshold: Upper threshold for ``HIGH`` level.
            grouping_min: Minimum cardinality for ``GROUPABLE`` level.
            grouping_max: Maximum cardinality for ``GROUPABLE`` level.

        Returns:
            DataFrame with columns ``column``, ``n_unique``,
            ``cardinality_level``, ``rare_categories``, ``rare_pct_of_rows``,
            ``action``, ``encoding_suggestion``, and ``reason``.

        Raises:
            ValueError: If thresholds are invalid.
        """
        if grouping_min < 1 or grouping_max < grouping_min:
            raise ValueError("Invalid grouping_min/grouping_max")
        if high_cardinality_threshold < grouping_max:
            raise ValueError("high_cardinality_threshold must be >= grouping_max")

        n_rows = max(len(self.df), 1)
        rows: list[dict[str, Any]] = []

        for col in self._categorical_cols(exclude_target=False, exclude_id=True):
            s = self.df[col]
            n_unique = int(s.nunique(dropna=True))
            vc = s.value_counts(dropna=True, normalize=True)
            rare_values = vc[vc < 0.05].index.tolist() if not vc.empty else []
            rare_pct_rows = float(vc[vc < 0.05].sum() * 100) if not vc.empty else 0.0

            level = ""
            action = ""
            suggestion = ""
            reason = ""
            severity = "INFO"

            if n_unique == 1:
                level = "CONSTANT"
                action = "AUTO_REMOVE"
                severity = "AUTO_REMOVE"
                suggestion = "Drop feature"
                reason = "Single unique value - zero information"
            elif n_unique == 2:
                level = "BINARY"
                action = "INFO"
                severity = "INFO"
                suggestion = "Label encoding or keep as-is"
                reason = "Binary - suitable for label encoding or as-is"
            elif 3 <= n_unique <= 6:
                level = "LOW"
                action = "RECOMMEND_ENCODE"
                severity = "RECOMMEND_ENCODE"
                suggestion = "One-Hot Encoding"
                reason = "Low cardinality - suitable for One-Hot Encoding"
            elif grouping_min <= n_unique <= grouping_max:
                level = "GROUPABLE"
                action = "RECOMMEND_GROUP"
                severity = "RECOMMEND_GROUP"
                suggestion = "Group rare values into 'Other' category before One-Hot"
                reason = (
                    "Cardinality in groupable range - consider grouping rare "
                    "categories (<5% frequency) before One-Hot Encoding"
                )
            elif n_unique <= high_cardinality_threshold:
                level = "HIGH"
                action = "RECOMMEND_ENCODE"
                severity = "RECOMMEND_ENCODE"
                suggestion = "Target Encoding or Frequency Encoding"
                reason = (
                    "High cardinality - consider Target/Frequency Encoding "
                    "instead of One-Hot"
                )
            else:
                level = "VERY_HIGH"
                action = "RECOMMEND_REVIEW"
                severity = "RECOMMEND_REVIEW"
                suggestion = "Review drop/embedding strategy"
                reason = (
                    "Very high cardinality - may be free-text or ID-like; "
                    "consider dropping or embedding-based encoding"
                )

            self._log_decision(
                action="analyze_cardinality",
                columns=[col],
                reason=reason,
                severity=severity,
                executed=False,
                metric=float(n_unique),
            )

            rows.append(
                {
                    "column": col,
                    "n_unique": n_unique,
                    "cardinality_level": level,
                    "rare_categories": rare_values,
                    "rare_pct_of_rows": round(rare_pct_rows, 4),
                    "action": action,
                    "encoding_suggestion": suggestion,
                    "reason": reason,
                }
            )

        result = pd.DataFrame(
            rows,
            columns=[
                "column",
                "n_unique",
                "cardinality_level",
                "rare_categories",
                "rare_pct_of_rows",
                "action",
                "encoding_suggestion",
                "reason",
            ],
        )
        self._cardinality_cache = result.copy()
        self.cardinality_analysis = result

        self._print("\n=== Cardinality Analysis ===")
        if result.empty:
            self._print("No categorical/object columns to analyze.")
        else:
            self._print(result.to_string(index=False))
        return result

    def detect_redundant_columns(self) -> pd.DataFrame:
        """Detect redundant columns by value, correlation, and name similarity.

        Args:
            None.

        Returns:
            DataFrame with columns ``column``, ``redundant_with``,
            ``detection_method``, ``action``, and ``reason``.

        Raises:
            None.
        """
        cols = self.df.columns.tolist()
        rows: list[dict[str, Any]] = []
        flagged_pairs: set[tuple[str, str, str]] = set()

        # 1) Perfect correlation on numeric columns.
        numeric = self._numeric_cols(exclude_target=False, exclude_id=False)
        if len(numeric) >= 2:
            corr_mat = self.df[numeric].corr(method=self.correlation_method)
            for i, c1 in enumerate(numeric):
                for j in range(i + 1, len(numeric)):
                    c2 = numeric[j]
                    corr = corr_mat.loc[c1, c2]
                    if pd.isna(corr) or not np.isclose(abs(corr), 1.0):
                        continue
                    key = tuple(sorted((c1, c2)) + ["perfect_corr"])
                    if key in flagged_pairs:
                        continue
                    flagged_pairs.add(key)
                    reason = f"Perfect correlation with {c2} - exact duplicate information"
                    self._log_decision(
                        action="consider_drop_redundant",
                        columns=[c1],
                        reason=reason,
                        severity="RECOMMEND_REMOVE",
                        executed=False,
                        metric=float(abs(corr)),
                    )
                    rows.append(
                        {
                            "column": c1,
                            "redundant_with": c2,
                            "detection_method": "PERFECT_CORRELATION",
                            "action": "RECOMMEND_REMOVE",
                            "reason": reason,
                        }
                    )

        # 2) Identical values.
        identical_pairs: set[tuple[str, str]] = set()
        for i, c1 in enumerate(cols):
            s1 = self.df[c1]
            for j in range(i + 1, len(cols)):
                c2 = cols[j]
                if s1.equals(self.df[c2]):
                    pair = tuple(sorted((c1, c2)))
                    identical_pairs.add(pair)
                    reason = f"Identical values to {c2} - exact duplicate column"
                    self._log_decision(
                        action="drop_identical_column",
                        columns=[c1],
                        reason=reason,
                        severity="AUTO_REMOVE",
                        executed=False,
                        metric=1.0,
                    )
                    rows.append(
                        {
                            "column": c1,
                            "redundant_with": c2,
                            "detection_method": "IDENTICAL_VALUES",
                            "action": "AUTO_REMOVE",
                            "reason": reason,
                        }
                    )

        # 3) Name similarity (> 0.85), excluding already flagged pairs.
        flagged_name_pairs = {tuple(sorted((r["column"], r["redundant_with"]))) for r in rows}
        for i, c1 in enumerate(cols):
            for j in range(i + 1, len(cols)):
                c2 = cols[j]
                pair = tuple(sorted((c1, c2)))
                if pair in flagged_name_pairs or pair in identical_pairs:
                    continue
                ratio = SequenceMatcher(None, str(c1).lower(), str(c2).lower()).ratio()
                if ratio > 0.85:
                    reason = f"Name similarity with {c2} - possible duplicate, verify manually"
                    self._log_decision(
                        action="review_name_similarity",
                        columns=[c1],
                        reason=reason,
                        severity="RECOMMEND_REVIEW",
                        executed=False,
                        metric=float(ratio),
                    )
                    rows.append(
                        {
                            "column": c1,
                            "redundant_with": c2,
                            "detection_method": "NAME_SIMILARITY",
                            "action": "RECOMMEND_REVIEW",
                            "reason": reason,
                        }
                    )

        result = pd.DataFrame(
            rows,
            columns=["column", "redundant_with", "detection_method", "action", "reason"],
        )
        self.redundancy_analysis = result

        self._print("\n=== Redundancy Detection ===")
        if result.empty:
            self._print("No redundant columns detected.")
        else:
            self._print(result.to_string(index=False))
        return result

    def _compute_iqr_outliers(self) -> pd.DataFrame:
        """Compute outlier percentages and bounds via IQR.

        Args:
            None.

        Returns:
            DataFrame with ``column``, ``outlier_pct``, ``lower_bound``,
            and ``upper_bound``.

        Raises:
            None.
        """
        numeric = self._numeric_cols(exclude_target=False, exclude_id=True)
        records: list[dict[str, Any]] = []
        for col in numeric:
            s = self.df[col].dropna()
            if s.empty:
                continue
            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            up = q3 + 1.5 * iqr
            mask = (s < low) | (s > up)
            pct = float(mask.mean() * 100)
            records.append(
                {
                    "column": col,
                    "outlier_pct": pct,
                    "lower_bound": low,
                    "upper_bound": up,
                }
            )
        return pd.DataFrame(records)

    def analyze_outliers(self) -> pd.DataFrame:
        """Generate outlier handling recommendations by outlier prevalence.

        Args:
            None.

        Returns:
            DataFrame with columns ``column``, ``outlier_pct``,
            ``lower_bound``, ``upper_bound``, ``action``, and
            ``recommendation``.

        Raises:
            None.
        """
        base = self._compute_iqr_outliers()
        if base.empty:
            result = pd.DataFrame(
                columns=[
                    "column",
                    "outlier_pct",
                    "lower_bound",
                    "upper_bound",
                    "action",
                    "recommendation",
                ]
            )
            self._outlier_cache = result
            self.outlier_analysis = result
            return result

        summary_outlier_cols = set(self.pipeline_summary.get("outlier_cols", []))

        rows: list[dict[str, Any]] = []
        for rec in base.itertuples(index=False):
            col = str(rec.column)
            pct = float(rec.outlier_pct)
            low = float(rec.lower_bound)
            up = float(rec.upper_bound)

            if pct < 1:
                action = "INFO"
                recommendation = "Minor outliers, likely safe to keep"
                severity = "INFO"
            elif pct < 5:
                action = "RECOMMEND_REVIEW"
                recommendation = (
                    "Moderate outliers - consider capping (Winsorization) "
                    "at 1st/99th percentile"
                )
                severity = "RECOMMEND_REVIEW"
            else:
                action = "RECOMMEND_REVIEW"
                recommendation = (
                    "High outlier rate - consider robust scaling or log "
                    "transformation. Check if distribution is naturally skewed."
                )
                severity = "RECOMMEND_REVIEW"

            if col == self.target_col:
                action = "RECOMMEND_REVIEW"
                severity = "RECOMMEND_REVIEW"
                recommendation += " Outliers in target variable affect all model metrics"

            if col in summary_outlier_cols and action == "INFO":
                action = "RECOMMEND_REVIEW"
                severity = "RECOMMEND_REVIEW"
                recommendation += " Upstream summary also flagged this feature."

            self._log_decision(
                action="analyze_outliers",
                columns=[col],
                reason=recommendation,
                severity=severity,
                executed=False,
                metric=pct,
            )

            rows.append(
                {
                    "column": col,
                    "outlier_pct": round(pct, 4),
                    "lower_bound": round(low, 6),
                    "upper_bound": round(up, 6),
                    "action": action,
                    "recommendation": recommendation,
                }
            )

        result = pd.DataFrame(rows)
        self._outlier_cache = result.copy()
        self.outlier_analysis = result

        self._print("\n=== Outlier Recommendations ===")
        self._print(result.to_string(index=False))
        return result

    def generate_fe_hints(self) -> pd.DataFrame:
        """Generate actionable feature-engineering hints before transformations.

        Args:
            None.

        Returns:
            DataFrame with columns ``column`` (or pair), ``hint_type``,
            ``suggestion``, and ``priority``.

        Raises:
            None.
        """
        hints: list[dict[str, Any]] = []
        n_rows = len(self.df)

        # Datetime hints.
        dt_cols = self.pipeline_summary.get("datetime_cols") or self.df.select_dtypes(include="datetime").columns.tolist()
        is_time_series = bool(self.pipeline_summary.get("is_time_series", False))
        for col in dt_cols:
            hints.append(
                {
                    "column": col,
                    "hint_type": "DATETIME_DECOMPOSITION",
                    "suggestion": "Extract year, month, day_of_week, quarter, is_weekend, days_since_reference",
                    "priority": "High",
                }
            )
            if is_time_series:
                hints.append(
                    {
                        "column": col,
                        "hint_type": "TIME_SERIES_FEATURES",
                        "suggestion": "Add lag features, rolling mean/std, and Fourier terms for seasonality",
                        "priority": "High",
                    }
                )

        # Skewed numeric hints.
        for col in self._numeric_cols(exclude_target=True, exclude_id=True):
            s = self.df[col].dropna()
            if len(s) < 5:
                continue
            skew = float(s.skew())
            if abs(skew) > 1:
                if float(s.min()) >= 0:
                    suggestion = f"Skewed (skew={skew:.2f}): consider log1p transformation"
                else:
                    suggestion = f"Skewed (skew={skew:.2f}): consider Box-Cox (after positivity shift)"
                hints.append(
                    {
                        "column": col,
                        "hint_type": "SKEWNESS_TRANSFORMATION",
                        "suggestion": suggestion,
                        "priority": "Medium",
                    }
                )

        # Bounded-range numeric hints.
        for col in self._numeric_cols(exclude_target=True, exclude_id=True):
            s = self.df[col].dropna()
            if s.empty:
                continue
            min_v = float(s.min())
            max_v = float(s.max())
            if 0 <= min_v and max_v <= 1:
                hints.append(
                    {
                        "column": col,
                        "hint_type": "BOUNDED_RANGE",
                        "suggestion": "Values in [0,1]; verify if feature is a rate/probability",
                        "priority": "Low",
                    }
                )
            elif 0 <= min_v and max_v <= 100:
                hints.append(
                    {
                        "column": col,
                        "hint_type": "PERCENTAGE_RANGE",
                        "suggestion": "Values in [0,100]; verify percentage semantics and scaling",
                        "priority": "Low",
                    }
                )

        # Groupable categorical hints based on cardinality analysis.
        card = self._cardinality_cache
        if card is None:
            card = pd.DataFrame()
        if not card.empty:
            groupable = card[card["cardinality_level"] == "GROUPABLE"]
            for rec in groupable.itertuples(index=False):
                if rec.rare_categories:
                    hints.append(
                        {
                            "column": rec.column,
                            "hint_type": "RARE_CATEGORY_GROUPING",
                            "suggestion": f"Merge rare categories into 'Other': {rec.rare_categories}",
                            "priority": "Medium",
                        }
                    )

        # Interaction hints.
        if self.target_col and self.target_col in self.df.columns:
            target_series = self.df[self.target_col]
            if pd.api.types.is_numeric_dtype(target_series):
                num_cols = self._numeric_cols(exclude_target=True, exclude_id=True)
                target_corrs: dict[str, float] = {}
                for col in num_cols:
                    valid = self.df[[col, self.target_col]].dropna()
                    if len(valid) < 3:
                        continue
                    corr = valid[col].corr(valid[self.target_col], method=self.correlation_method)
                    if pd.notna(corr):
                        target_corrs[col] = float(abs(corr))

                strong = [c for c, v in target_corrs.items() if v > 0.3]
                if len(strong) >= 2:
                    inter = self.df[strong].corr(method=self.correlation_method).abs()
                    for i, c1 in enumerate(strong):
                        for j in range(i + 1, len(strong)):
                            c2 = strong[j]
                            pair_corr = float(inter.loc[c1, c2])
                            if pair_corr < 0.5:
                                hints.append(
                                    {
                                        "column": f"{c1} * {c2}",
                                        "hint_type": "INTERACTION_FEATURE",
                                        "suggestion": (
                                            f"Create interaction {c1}*{c2}; both correlate with target "
                                            f"(>|0.3|) while inter-correlation is {pair_corr:.2f}"
                                        ),
                                        "priority": "Medium",
                                    }
                                )

        # Binary encoding candidates in object columns.
        binary_token_sets = [
            {"yes", "no"},
            {"true", "false"},
            {"0", "1"},
        ]
        for col in self._categorical_cols(exclude_target=False, exclude_id=True):
            uniq = {
                str(v).strip().lower()
                for v in self.df[col].dropna().unique().tolist()
            }
            if len(uniq) == 2 and any(uniq == tokens for tokens in binary_token_sets):
                hints.append(
                    {
                        "column": col,
                        "hint_type": "BINARY_ENCODING_CANDIDATE",
                        "suggestion": "Convert binary object values to int (0/1) early in pipeline",
                        "priority": "High",
                    }
                )

        # ID-like columns not declared as id_col.
        if n_rows > 0:
            for col in self.df.columns:
                if col == self.id_col:
                    continue
                if self.df[col].nunique(dropna=False) == n_rows:
                    hints.append(
                        {
                            "column": col,
                            "hint_type": "ID_LIKE_COLUMN",
                            "suggestion": "Cardinality equals row count; likely ID-like. Consider dropping or using as index only",
                            "priority": "Medium",
                        }
                    )

        result = pd.DataFrame(
            hints,
            columns=["column", "hint_type", "suggestion", "priority"],
        )

        self._print("\n=== Feature Engineering Hints ===")
        if result.empty:
            self._print("No feature-engineering hints generated.")
        else:
            self._print(result.to_string(index=False))
        self.fe_hints = result
        return result

    def generate_recommendation_report(self) -> pd.DataFrame:
        """Build the consolidated auditable recommendation report.

        Args:
            None.

        Returns:
            DataFrame containing all entries from ``self.decision_log``,
            sorted by severity priority.

        Raises:
            None.
        """
        if not self.decision_log:
            report = pd.DataFrame(
                columns=[
                    "action",
                    "columns",
                    "reason",
                    "severity",
                    "metric",
                    "executed",
                    "timestamp",
                ]
            )
            self.recommendation_report = report
            return report

        report = pd.DataFrame(self.decision_log)
        order = {
            "AUTO_REMOVE": 0,
            "RECOMMEND_REMOVE": 1,
            "RECOMMEND_REVIEW": 2,
            "RECOMMEND_GROUP": 3,
            "RECOMMEND_ENCODE": 4,
            "RECOMMEND_IMPUTE": 5,
            "INFO": 6,
        }
        report["_order"] = report["severity"].map(order).fillna(99)
        report = report.sort_values(["_order", "timestamp"]).drop(columns="_order")
        report = report.reset_index(drop=True)

        if self.verbose:
            counts = (
                report["severity"]
                .value_counts()
                .reindex(order.keys(), fill_value=0)
                .rename_axis("severity")
                .reset_index(name="count")
            )
            self._print("\n=== Recommendation Severity Summary ===")
            self._print(counts.to_string(index=False))

        self.recommendation_report = report
        return report

    def run_full_analysis(self) -> dict[str, pd.DataFrame]:
        """Run all cleaning recommendation analyses in a fixed pipeline order.

        Args:
            None.

        Returns:
            Dictionary keyed by method name with each result DataFrame.

        Raises:
            CleaningError: If a critical internal failure occurs.
        """
        try:
            outputs: dict[str, pd.DataFrame] = {}

            outputs["high_null_columns"] = self.analyze_high_null_columns()
            outputs["dominant_classes"] = self.analyze_dominant_classes()
            outputs["correlations"] = self.analyze_correlations()
            outputs["cardinality"] = self.analyze_cardinality()
            outputs["redundant_columns"] = self.detect_redundant_columns()
            outputs["outliers"] = self.analyze_outliers()
            outputs["fe_hints"] = self.generate_fe_hints()
            outputs["recommendation_report"] = self.generate_recommendation_report()

            # Store as instance attributes for downstream Part 2B consumption.
            self.results_ = outputs
            return outputs
        except Exception as exc:  # noqa: BLE001
            raise CleaningError(f"run_full_analysis failed: {exc}") from exc


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    np.random.seed(42)

    X, y = make_classification(
        n_samples=1200,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        weights=[0.85, 0.15],
        random_state=42,
    )

    cols = [f"feat_{i}" for i in range(X.shape[1])]
    df_demo = pd.DataFrame(X, columns=cols)
    df_demo["target"] = y

    # Inject intentional dirty patterns.
    df_demo["id"] = np.arange(len(df_demo))
    df_demo["feat_corr"] = df_demo["feat_0"] * 1.0
    df_demo["feat_corr_copy"] = df_demo["feat_corr"]
    df_demo["dominant_cat"] = np.where(np.random.rand(len(df_demo)) < 0.97, "A", "B")
    df_demo["binary_text"] = np.where(np.random.rand(len(df_demo)) < 0.5, "yes", "no")
    df_demo["high_card"] = [f"cat_{i % 120}" for i in range(len(df_demo))]

    # Null injection.
    null_mask_a = np.random.rand(len(df_demo)) < 0.65
    null_mask_b = np.random.rand(len(df_demo)) < 0.22
    df_demo.loc[null_mask_a, "feat_2"] = np.nan
    df_demo.loc[null_mask_b, "dominant_cat"] = np.nan

    # Duplicates.
    df_demo = pd.concat([df_demo, df_demo.iloc[:15]], ignore_index=True)

    # Minimal pipeline summary mock, compatible with DataExplorer output keys.
    pipeline_summary_demo = {
        "shape": tuple(df_demo.shape),
        "null_columns": df_demo.columns[df_demo.isnull().any()].tolist(),
        "high_null_columns": df_demo.columns[(df_demo.isnull().mean() > 0.3)].tolist(),
        "numeric_cols": df_demo.select_dtypes(include="number").columns.tolist(),
        "categorical_cols": df_demo.select_dtypes(include=["object", "category"]).columns.tolist(),
        "datetime_cols": [],
        "boolean_cols": [],
        "freetext_cols": [],
        "low_variance_cols": [],
        "outlier_cols": [],
        "non_normal_cols": [],
        "duplicate_rows": int(df_demo.duplicated().sum()),
        "duplicate_ids": None,
        "target_col": "target",
        "id_col": "id",
        "is_tidy": True,
        "is_time_series": False,
        "alerts": pd.DataFrame(),
    }

    cleaner = DataCleaner(
        df_demo,
        pipeline_summary=pipeline_summary_demo,
        target_col="target",
        id_col="id",
        dry_run=True,
        correlation_method="pearson",
        verbose=True,
    )

    outputs = cleaner.run_full_analysis()
    print("\n=== Output keys ===")
    for k, v in outputs.items():
        print(f"{k}: {type(v).__name__} ({len(v)} rows)")
