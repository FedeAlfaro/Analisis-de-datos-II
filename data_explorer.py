"""Data Explorer — Part 1/2 of a modular ML pipeline.

This module provides the ``DataExplorer`` class, a comprehensive exploratory
data analysis (EDA) toolkit designed to integrate cleanly with downstream
pipeline stages (data cleaning, feature engineering, model training).

Every public method returns structured objects (DataFrames, dicts, Series)
or generates deterministic visual/report artifacts that can be consumed
programmatically. When ``verbose=True`` (the default), human-readable
summaries are also printed to stdout.

Usage example
-------------
>>> import pandas as pd
>>> from data_explorer import DataExplorer
>>> df = pd.read_csv("my_dataset.csv")
>>> explorer = DataExplorer(df, target_col="label", id_col="id")
>>> results = explorer.run_full_eda()

Public API (high level)
-----------------------
Structural and quality diagnostics:
- ``check_tidy_format``
- ``get_structural_summary``
- ``analyze_nulls``
- ``analyze_duplicates``
- ``analyze_target``
- ``detect_low_variance``
- ``detect_outliers``
- ``generate_alert_summary``

Correlation and distribution plots:
- ``plot_correlation_heatmap``
- ``plot_target_correlations``
- ``plot_normality``
- ``plot_scatter``
- ``plot_scatter_vs_target``

Reporting and integration:
- ``generate_sweetviz_report``
- ``get_pipeline_summary``
- ``run_full_eda``

Integration with the broader ML pipeline
-----------------------------------------
The dict returned by ``run_full_eda()`` is intended to be passed directly to
the next stage of the pipeline.  Keys correspond to method names and values
are the structured outputs of each analysis step.  Downstream modules can,
for example, use the null analysis DataFrame to decide on imputation
strategies or the outlier DataFrame to apply capping/removal.
"""

from __future__ import annotations

import re
import traceback
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# Custom warning
# ---------------------------------------------------------------------------

class DataQualityWarning(UserWarning):
    """Custom warning for data quality issues detected during EDA."""


# ---------------------------------------------------------------------------
# Helper utilities (module-private)
# ---------------------------------------------------------------------------

def _cramers_v(series_a: pd.Series, series_b: pd.Series) -> float:
    """Compute Cramér's V statistic for two categorical Series."""
    confusion = pd.crosstab(series_a, series_b)
    n = confusion.sum().sum()
    if n == 0:
        return 0.0
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    r, k = confusion.shape
    phi2 = chi2 / n
    r_corr = r - (r - 1) ** 2 / (n - 1) if n > 1 else r
    k_corr = k - (k - 1) ** 2 / (n - 1) if n > 1 else k
    denom = min(r_corr - 1, k_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max(0, phi2 / denom)))


def _is_datelike_string(name: str) -> bool:
    """Return True if *name* looks like a date or year."""
    name_str = str(name)
    if re.fullmatch(r"\d{4}(-\d{2}){0,2}", name_str):
        return True
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", name_str):
        return True
    return False


def _detect_temporal_groups(columns: list[str]) -> list[str]:
    """Detect column-name groups that encode the same variable across time.

    For example ``['sales_2020', 'sales_2021', 'sales_2022']``.
    Returns a list of human-readable descriptions of each detected group.
    """
    pattern = re.compile(r"^(.+?)[\s_](\d{4})$")
    groups: dict[str, list[str]] = {}
    for col in columns:
        m = pattern.match(str(col))
        if m:
            prefix = m.group(1)
            groups.setdefault(prefix, []).append(str(col))
    issues: list[str] = []
    for prefix, cols in groups.items():
        if len(cols) >= 2:
            issues.append(
                f"Columns appear to encode '{prefix}' across time: {cols}"
            )
    return issues


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DataExplorer:
    """Comprehensive EDA toolkit for a single DataFrame.

    Args:
        df: Input dataset.
        id_col: Optional identifier column name.
        target_col: Optional target variable name.
        date_cols: Optional list of known date columns.
        verbose: If ``True``, print detailed output.
        outlier_method: ``'iqr'`` or ``'zscore'``.
        outlier_threshold: Multiplier for IQR or Z threshold.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        id_col: str | None = None,
        target_col: str | None = None,
        date_cols: list[str] | None = None,
        verbose: bool = True,
        outlier_method: str = "iqr",
        outlier_threshold: float | None = None,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        self.df = df.copy()
        self.id_col = id_col
        self.target_col = target_col
        self.date_cols = date_cols or []
        self.verbose = verbose

        outlier_method = outlier_method.lower()
        if outlier_method not in ("iqr", "zscore"):
            raise ValueError("outlier_method must be 'iqr' or 'zscore'")
        self.outlier_method = outlier_method

        if outlier_threshold is None:
            self.outlier_threshold = 1.5 if outlier_method == "iqr" else 3.0
        else:
            self.outlier_threshold = outlier_threshold

        # Internal store for alerts raised during analysis.
        self._alerts: list[dict[str, str]] = []

        # Caches used by downstream integration summaries.
        self._normality_results: pd.DataFrame | None = None

        # Parse known date columns.
        for col in self.date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _warn(self, message: str, *, severity: str = "WARNING",
              category: str = "general", affected: str = "") -> None:
        """Issue a ``DataQualityWarning`` and record it internally."""
        warnings.warn(message, DataQualityWarning, stacklevel=3)
        self._alerts.append({
            "severity": severity,
            "category": category,
            "message": message,
            "affected_columns": affected,
        })

    def _print(self, *args: Any, **kwargs: Any) -> None:
        """Print only when verbose mode is on."""
        if self.verbose:
            print(*args, **kwargs)

    def _numeric_cols(self, exclude_id: bool = True) -> list[str]:
        """Return numeric column names, optionally excluding the id column."""
        cols = self.df.select_dtypes(include="number").columns.tolist()
        if exclude_id and self.id_col and self.id_col in cols:
            cols.remove(self.id_col)
        return cols

    def _categorical_cols(
        self, cardinality_threshold: int = 50, exclude_id: bool = True
    ) -> list[str]:
        """Return categorical-like column names."""
        cols: list[str] = []
        for col in self.df.columns:
            if exclude_id and col == self.id_col:
                continue
            dtype = self.df[col].dtype
            if dtype.name == "category" or (
                dtype == object
                and self.df[col].nunique() <= cardinality_threshold
            ):
                cols.append(col)
        return cols

    def _get_grid_dims(self, n: int) -> tuple[int, int]:
        """Compute subplot grid dimensions close to a 16:9 aspect ratio.

        Args:
            n: Number of subplot cells required.

        Returns:
            A tuple ``(n_rows, n_cols)`` with at least one column and with
            columns capped at 4 for readability.

        Raises:
            ValueError: If ``n`` is not a positive integer.
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")

        best_rows, best_cols = 1, 1
        best_score = float("inf")

        max_cols = min(4, n)
        target_ratio = 16 / 9
        for n_cols in range(1, max_cols + 1):
            n_rows = int(np.ceil(n / n_cols))
            ratio = n_cols / max(1, n_rows)
            empty_slots = n_rows * n_cols - n
            score = abs(ratio - target_ratio) + 0.12 * empty_slots
            if score < best_score:
                best_score = score
                best_rows, best_cols = n_rows, n_cols

        return best_rows, best_cols

    def _get_figsize(
        self,
        n_rows: int,
        n_cols: int,
        cell_size: float = 4.0,
    ) -> tuple[float, float]:
        """Scale figure size from grid dimensions.

        Args:
            n_rows: Number of rows in the subplot grid.
            n_cols: Number of columns in the subplot grid.
            cell_size: Base inches per subplot cell.

        Returns:
            Figure size as ``(width, height)`` in inches.

        Raises:
            ValueError: If grid dimensions or cell size are non-positive.
        """
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("n_rows and n_cols must be positive")
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        return float(n_cols * cell_size), float(n_rows * cell_size)

    # ------------------------------------------------------------------
    # Tidy format detection
    # ------------------------------------------------------------------

    def check_tidy_format(self) -> dict[str, Any]:
        """Detect common deviations from tidy (long) format.

        Returns:
            A dict with keys ``is_tidy`` (bool), ``issues`` (list[str]),
            and ``suggestions`` (list[str]).
        """
        issues: list[str] = []
        suggestions: list[str] = []

        # 1. Numeric / date-like column names → probably pivoted.
        datelike = [c for c in self.df.columns if _is_datelike_string(str(c))]
        numeric_names = [
            c for c in self.df.columns
            if re.fullmatch(r"\d+\.?\d*", str(c)) and c not in datelike
        ]
        if datelike:
            issues.append(
                f"Column names look like dates: {datelike}"
            )
            suggestions.append(
                "Consider melting these columns into a long format with a "
                "'date' column."
            )
        if numeric_names:
            issues.append(
                f"Column names are purely numeric: {numeric_names}"
            )
            suggestions.append(
                "Consider melting numeric-named columns into long format."
            )

        # 2. Duplicate column names.
        dup_cols = self.df.columns[self.df.columns.duplicated()].tolist()
        if dup_cols:
            issues.append(f"Duplicate column names detected: {dup_cols}")
            suggestions.append("Rename or drop duplicate column names.")

        # 3. Meaningful (non-default) index.
        if not isinstance(self.df.index, pd.RangeIndex):
            issues.append(
                "The DataFrame has a non-default index which may contain "
                "information that should be a column."
            )
            suggestions.append(
                "Consider resetting the index with df.reset_index()."
            )

        # 4. Temporal column-name groups.
        temporal = _detect_temporal_groups(
            [str(c) for c in self.df.columns]
        )
        for msg in temporal:
            issues.append(msg)
            suggestions.append(
                "Melt temporal columns into rows with a 'year'/'period' column."
            )

        # Emit warnings.
        for issue in issues:
            self._warn(issue, severity="WARNING", category="tidy_format",
                       affected="")

        is_tidy = len(issues) == 0
        result = {"is_tidy": is_tidy, "issues": issues,
                  "suggestions": suggestions}

        self._print("\n=== Tidy Format Check ===")
        if is_tidy:
            self._print("Dataset appears to be in tidy format.")
        else:
            for i, iss in enumerate(issues):
                self._print(f"  Issue {i + 1}: {iss}")
                self._print(f"    Suggestion: {suggestions[i]}")
        return result

    # ------------------------------------------------------------------
    # Structural summary
    # ------------------------------------------------------------------

    def get_structural_summary(
        self, cardinality_threshold: int = 50
    ) -> pd.DataFrame:
        """Report high-level structural information about the dataset.

        Args:
            cardinality_threshold: Maximum unique values for an object column
                to be classified as categorical (vs. free text).

        Returns:
            A summary DataFrame with one metric per row.
        """
        df = self.df
        n_rows, n_cols = df.shape
        mem_bytes = df.memory_usage(deep=True).sum()
        mem_mb = mem_bytes / (1024 ** 2)

        numeric = df.select_dtypes(include="number").columns.tolist()
        boolean = df.select_dtypes(include="bool").columns.tolist()
        datetime = df.select_dtypes(include="datetime").columns.tolist()
        categorical: list[str] = []
        freetext: list[str] = []
        unknown: list[str] = []

        for col in df.columns:
            if col in numeric or col in boolean or col in datetime:
                continue
            if df[col].dtype.name == "category":
                categorical.append(col)
            elif df[col].dtype == object:
                nunique = df[col].nunique()
                n_nonnull = df[col].count()
                if n_nonnull > 0 and nunique / n_nonnull > 0.5:
                    freetext.append(col)
                elif nunique <= cardinality_threshold:
                    categorical.append(col)
                else:
                    freetext.append(col)
            else:
                unknown.append(col)

        is_large_rows = n_rows > 500_000
        is_large_cols = n_cols > 50

        rows: list[dict[str, Any]] = [
            {"metric": "Total rows", "value": n_rows},
            {"metric": "Total columns", "value": n_cols},
            {"metric": "Memory usage", "value": f"{mem_mb:.2f} MB"},
            {"metric": "Numeric columns", "value": len(numeric)},
            {"metric": "Categorical columns", "value": len(categorical)},
            {"metric": "Boolean columns", "value": len(boolean)},
            {"metric": "Datetime columns", "value": len(datetime)},
            {"metric": "Free-text columns", "value": len(freetext)},
            {"metric": "Unknown/mixed type columns", "value": len(unknown)},
            {"metric": "Large dataset (rows >500k)",
             "value": is_large_rows},
            {"metric": "Wide dataset (cols >50)",
             "value": is_large_cols},
        ]

        summary = pd.DataFrame(rows)

        self._print("\n=== Structural Summary ===")
        self._print(summary.to_string(index=False))
        return summary

    # ------------------------------------------------------------------
    # Null analysis
    # ------------------------------------------------------------------

    def analyze_nulls(self) -> pd.DataFrame:
        """Analyze missing values across all columns.

        Returns:
            A DataFrame with ``null_count`` and ``null_pct`` per column,
            sorted by ``null_pct`` descending.
        """
        df = self.df
        null_count = df.isnull().sum()
        null_pct = (null_count / len(df) * 100) if len(df) > 0 else null_count

        result = pd.DataFrame({
            "null_count": null_count,
            "null_pct": null_pct,
        }).sort_values("null_pct", ascending=False)

        # Graduated warnings.
        for col in result.index:
            pct = result.loc[col, "null_pct"]
            if pct > 90:
                self._warn(
                    f"Column '{col}' has {pct:.1f}% nulls (>90%)",
                    severity="CRITICAL", category="nulls", affected=col,
                )
            elif pct > 60:
                self._warn(
                    f"Column '{col}' has {pct:.1f}% nulls (>60%)",
                    severity="WARNING", category="nulls", affected=col,
                )
            elif pct > 30:
                self._warn(
                    f"Column '{col}' has {pct:.1f}% nulls (>30%)",
                    severity="INFO", category="nulls", affected=col,
                )

        # Entirely null rows.
        if len(df) > 0:
            fully_null_rows = int(df.isnull().all(axis=1).sum())
            if fully_null_rows > 0:
                self._warn(
                    f"{fully_null_rows} rows are entirely null",
                    severity="WARNING", category="nulls", affected="",
                )

        self._print("\n=== Null Analysis ===")
        self._print(result.to_string())
        return result

    # ------------------------------------------------------------------
    # Duplicate analysis
    # ------------------------------------------------------------------

    def analyze_duplicates(self) -> dict[str, Any]:
        """Detect duplicate rows and duplicate ID values.

        Returns:
            A dict with ``full_duplicates`` (int), ``id_duplicates``
            (int | None), and ``duplicated_ids`` (list | None).
        """
        df = self.df
        full_dups = int(df.duplicated().sum())
        id_dups: int | None = None
        dup_ids: list[Any] | None = None

        if self.id_col and self.id_col in df.columns:
            mask = df[self.id_col].duplicated(keep=False)
            id_dups = int(df[self.id_col].duplicated().sum())
            if id_dups > 0:
                dup_ids = df.loc[mask, self.id_col].unique().tolist()

        if full_dups > 0:
            self._warn(
                f"{full_dups} fully duplicate rows detected",
                severity="WARNING", category="duplicates", affected="",
            )
        if id_dups and id_dups > 0:
            self._warn(
                f"{id_dups} duplicate ID values in '{self.id_col}'",
                severity="CRITICAL", category="duplicates",
                affected=self.id_col or "",
            )

        result: dict[str, Any] = {
            "full_duplicates": full_dups,
            "id_duplicates": id_dups,
            "duplicated_ids": dup_ids,
        }

        self._print("\n=== Duplicate Analysis ===")
        self._print(f"  Full duplicate rows: {full_dups}")
        if self.id_col:
            self._print(f"  Duplicate IDs: {id_dups}")
        return result

    # ------------------------------------------------------------------
    # Target variable analysis
    # ------------------------------------------------------------------

    def analyze_target(self) -> dict[str, Any] | None:
        """Analyze the target variable (if provided).

        Returns:
            A summary dict, or ``None`` if no target column is set.
        """
        if not self.target_col or self.target_col not in self.df.columns:
            self._print("\n=== Target Analysis ===")
            self._print("  No target column specified; skipping.")
            return None

        series = self.df[self.target_col].dropna()
        result: dict[str, Any] = {"column": self.target_col}

        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
            # Treat as numerical target.
            result["type"] = "numerical"
            result["mean"] = float(series.mean())
            result["std"] = float(series.std())
            result["skew"] = float(series.skew())
            result["kurtosis"] = float(series.kurtosis())
        else:
            # Treat as categorical target.
            result["type"] = "categorical"
            counts = series.value_counts()
            pcts = series.value_counts(normalize=True) * 100
            result["class_counts"] = counts.to_dict()
            result["class_percentages"] = pcts.to_dict()

            min_pct = pcts.min()
            if min_pct < 5:
                self._warn(
                    f"Severe class imbalance: minority class is {min_pct:.1f}%",
                    severity="CRITICAL", category="target",
                    affected=self.target_col,
                )
            elif min_pct < 10:
                self._warn(
                    f"Class imbalance detected: minority class is {min_pct:.1f}%",
                    severity="WARNING", category="target",
                    affected=self.target_col,
                )

        # Check for high null rate in target.
        null_pct = self.df[self.target_col].isnull().mean() * 100
        if null_pct > 50:
            self._warn(
                f"Target column '{self.target_col}' has {null_pct:.1f}% nulls",
                severity="CRITICAL", category="target",
                affected=self.target_col,
            )

        self._print("\n=== Target Variable Analysis ===")
        for k, v in result.items():
            self._print(f"  {k}: {v}")
        return result

    # ------------------------------------------------------------------
    # Near-zero variance detection
    # ------------------------------------------------------------------

    def detect_low_variance(self, threshold: float = 0.01) -> list[str]:
        """Identify columns with near-zero or zero variance.

        Args:
            threshold: Variance threshold for numeric columns.

        Returns:
            A list of column names with low/zero variance.
        """
        low_var: list[str] = []

        for col in self._numeric_cols(exclude_id=True):
            var = self.df[col].var()
            if var is not None and var < threshold:
                low_var.append(col)

        for col in self._categorical_cols(exclude_id=True):
            counts = self.df[col].value_counts(normalize=True)
            if len(counts) > 0 and counts.iloc[0] > 0.99:
                low_var.append(col)

        if low_var:
            self._warn(
                f"Low/zero variance columns: {low_var}",
                severity="WARNING", category="low_variance",
                affected=", ".join(low_var),
            )

        # Check if target has zero variance.
        if self.target_col and self.target_col in self.df.columns:
            target_var = self.df[self.target_col].var()
            if (
                pd.api.types.is_numeric_dtype(self.df[self.target_col])
                and target_var is not None
                and target_var == 0
            ):
                self._warn(
                    f"Target column '{self.target_col}' has zero variance",
                    severity="CRITICAL", category="low_variance",
                    affected=self.target_col,
                )

        self._print("\n=== Low Variance Detection ===")
        if low_var:
            self._print(f"  Columns with low variance: {low_var}")
        else:
            self._print("  No low-variance columns detected.")
        return low_var

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------

    def detect_outliers(self) -> pd.DataFrame:
        """Detect outliers in numeric columns using the configured method.

        Returns:
            A DataFrame with outlier statistics per column.
        """
        records: list[dict[str, Any]] = []
        numeric = self._numeric_cols(exclude_id=True)

        for col in numeric:
            s = self.df[col].dropna()
            if len(s) == 0:
                continue

            if self.outlier_method == "iqr":
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.outlier_threshold * iqr
                upper = q3 + self.outlier_threshold * iqr
            else:  # zscore
                mean = s.mean()
                std = s.std()
                lower = mean - self.outlier_threshold * std
                upper = mean + self.outlier_threshold * std

            n_outliers = int(((s < lower) | (s > upper)).sum())
            pct = n_outliers / len(s) * 100

            records.append({
                "column": col,
                "n_outliers": n_outliers,
                "pct_outliers": round(pct, 2),
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
            })

        result = pd.DataFrame(records)
        if not result.empty:
            result = result.set_index("column")

        # Warn for columns with >5% outliers.
        warn_threshold = 10
        info_threshold = 5
        for rec in records:
            if rec["pct_outliers"] > warn_threshold:
                self._warn(
                    f"Column '{rec['column']}' has {rec['pct_outliers']}% "
                    f"outliers (>{warn_threshold}%)",
                    severity="WARNING", category="outliers",
                    affected=rec["column"],
                )
            elif rec["pct_outliers"] > info_threshold:
                self._warn(
                    f"Column '{rec['column']}' has {rec['pct_outliers']}% "
                    f"outliers (>{info_threshold}%)",
                    severity="INFO", category="outliers",
                    affected=rec["column"],
                )

        self._print("\n=== Outlier Detection ===")
        self._print(f"  Method: {self.outlier_method}, "
                     f"threshold: {self.outlier_threshold}")
        if not result.empty:
            self._print(result.to_string())
        else:
            self._print("  No numeric columns to analyze.")
        return result

    # ------------------------------------------------------------------
    # Correlation heatmap
    # ------------------------------------------------------------------

    def plot_correlation_heatmap(
        self,
        method: str = "pearson",
        max_cols_per_plot: int = 20,
        figsize: tuple[int, int] = (14, 10),
    ) -> None:
        """Plot correlation / association heatmaps.

        Args:
            method: ``'pearson'`` or ``'spearman'`` for numeric columns.
            max_cols_per_plot: Maximum columns per subplot before splitting.
            figsize: Figure size for each plot.
        """
        numeric = self._numeric_cols(exclude_id=True)
        categorical = self._categorical_cols(exclude_id=True)

        # --- Numeric correlations ---
        if len(numeric) < 2:
            self._warn(
                "Too few numeric columns for a meaningful correlation matrix",
                severity="INFO", category="correlation", affected="",
            )
        else:
            n_groups = max(1, -(-len(numeric) // max_cols_per_plot))  # ceil
            for g in range(n_groups):
                start = g * max_cols_per_plot
                end = start + max_cols_per_plot
                subset = numeric[start:end]
                corr = self.df[subset].corr(method=method)
                title = f"Numeric correlations ({method})"
                if n_groups > 1:
                    title += f" — Group {g + 1} of {n_groups}"
                plt.figure(figsize=figsize)
                sns.heatmap(
                    corr,
                    annot=True,
                    fmt=".2f",
                    cmap="RdBu_r",
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                )
                plt.title(title)
                plt.tight_layout()
                plt.show()

        # --- Categorical associations (Cramér's V) ---
        if len(categorical) >= 2:
            n_groups = max(
                1, -(-len(categorical) // max_cols_per_plot)
            )
            for g in range(n_groups):
                start = g * max_cols_per_plot
                end = start + max_cols_per_plot
                subset = categorical[start:end]
                matrix = pd.DataFrame(
                    np.zeros((len(subset), len(subset))),
                    index=subset,
                    columns=subset,
                )
                for i, c1 in enumerate(subset):
                    for j, c2 in enumerate(subset):
                        if i <= j:
                            v = (
                                1.0 if c1 == c2
                                else _cramers_v(
                                    self.df[c1].fillna("__NA__"),
                                    self.df[c2].fillna("__NA__"),
                                )
                            )
                            matrix.loc[c1, c2] = v
                            matrix.loc[c2, c1] = v
                title = "Categorical associations (Cramér's V)"
                if n_groups > 1:
                    title += f" — Group {g + 1} of {n_groups}"
                plt.figure(figsize=figsize)
                sns.heatmap(
                    matrix.astype(float),
                    annot=True,
                    fmt=".2f",
                    cmap="YlOrRd",
                    vmin=0,
                    vmax=1,
                    square=True,
                )
                plt.title(title)
                plt.tight_layout()
                plt.show()

    # ------------------------------------------------------------------
    # Correlation with target
    # ------------------------------------------------------------------

    def plot_target_correlations(
        self, top_n: int = 20
    ) -> pd.Series | None:
        """Plot correlations of features with the target variable.

        Args:
            top_n: Maximum number of features to display.

        Returns:
            A Series of correlation/association values, or ``None`` if no
            target is set.
        """
        if not self.target_col or self.target_col not in self.df.columns:
            self._print("\n=== Target Correlations ===")
            self._print("  No target column specified; skipping.")
            return None

        target = self.df[self.target_col]
        numeric = self._numeric_cols(exclude_id=True)

        if pd.api.types.is_numeric_dtype(target) and target.nunique() > 20:
            # Numeric target → Pearson/Spearman correlations.
            corrs: dict[str, float] = {}
            for col in numeric:
                if col == self.target_col:
                    continue
                valid = self.df[[col, self.target_col]].dropna()
                if len(valid) < 3:
                    continue
                corrs[col] = float(valid[col].corr(valid[self.target_col]))
            series = pd.Series(corrs).reindex(
                pd.Series(corrs).abs().sort_values(ascending=False).index
            ).head(top_n)
        else:
            # Categorical target → Cramér's V for categoricals,
            # or point-biserial for numeric features with binary target.
            assocs: dict[str, float] = {}
            is_binary = target.dropna().nunique() == 2

            if is_binary:
                # Encode target as 0/1 for point-biserial.
                classes = target.dropna().unique()
                target_binary = target.map(
                    {classes[0]: 0, classes[1]: 1}
                )
                for col in numeric:
                    if col == self.target_col:
                        continue
                    valid = pd.DataFrame(
                        {"feat": self.df[col], "target": target_binary}
                    ).dropna()
                    if len(valid) < 3:
                        continue
                    assocs[col] = float(
                        valid["feat"].corr(valid["target"])
                    )
            else:
                # Multi-class → Cramér's V for categorical features.
                for col in self._categorical_cols(exclude_id=True):
                    if col == self.target_col:
                        continue
                    assocs[col] = _cramers_v(
                        self.df[col].fillna("__NA__"),
                        target.fillna("__NA__"),
                    )

            series = pd.Series(assocs).reindex(
                pd.Series(assocs).abs().sort_values(ascending=False).index
            ).head(top_n)

        if len(series) > 0:
            plt.figure(figsize=(10, max(4, len(series) * 0.35)))
            series.sort_values().plot.barh()
            plt.title(f"Feature association with '{self.target_col}'")
            plt.xlabel("Correlation / Association")
            plt.tight_layout()
            plt.show()

        self._print("\n=== Target Correlations ===")
        self._print(series.to_string())
        return series

    # ------------------------------------------------------------------
    # Normality analysis
    # ------------------------------------------------------------------

    def plot_normality(
        self,
        max_cols: int = 24,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Evaluate and visualize normality for numeric features.

        Args:
            max_cols: Maximum number of numeric columns to plot/test.
            alpha: Significance level used to classify normal vs non-normal.

        Returns:
            DataFrame with columns ``variable``, ``test_used``, ``statistic``,
            ``p_value``, ``is_normal``, ``skewness``, and ``kurtosis``.

        Raises:
            ValueError: If ``max_cols`` is less than 1 or ``alpha`` invalid.
        """
        if max_cols < 1:
            raise ValueError("max_cols must be >= 1")
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be in (0, 1)")

        try:
            numeric = self._numeric_cols(exclude_id=True)
            if not numeric:
                self._warn(
                    "No numeric columns available for normality analysis",
                    severity="INFO",
                    category="normality",
                    affected="",
                )
                empty = pd.DataFrame(
                    columns=[
                        "variable",
                        "test_used",
                        "statistic",
                        "p_value",
                        "is_normal",
                        "skewness",
                        "kurtosis",
                    ]
                )
                self._normality_results = empty
                return empty

            selected = list(numeric)
            if len(selected) > max_cols:
                variances = self.df[selected].var(numeric_only=True)
                keep = variances.sort_values(ascending=False).head(max_cols).index
                keep_set = set(keep)
                excluded = [c for c in selected if c not in keep_set]
                selected = list(keep)
                self._warn(
                    "Normality analysis limited by max_cols. Excluded columns: "
                    f"{excluded}",
                    severity="INFO",
                    category="normality",
                    affected=", ".join(excluded),
                )

            # Keep figures readable by plotting in chunks.
            chunk_size = 8
            results: list[dict[str, Any]] = []

            for start in range(0, len(selected), chunk_size):
                cols_chunk = selected[start:start + chunk_size]
                n_vars = len(cols_chunk)
                n_rows, n_cols = self._get_grid_dims(n_vars)

                fig_rows = n_rows * 2
                fig_size = self._get_figsize(fig_rows, n_cols, cell_size=3.8)
                fig, axes = plt.subplots(fig_rows, n_cols, figsize=fig_size)
                axes_arr = np.array(axes, dtype=object)
                if axes_arr.ndim == 1:
                    axes_arr = axes_arr.reshape(fig_rows, n_cols)

                for idx, col in enumerate(cols_chunk):
                    row = idx // n_cols
                    col_pos = idx % n_cols
                    ax_hist = axes_arr[row, col_pos]
                    ax_qq = axes_arr[row + n_rows, col_pos]

                    s = self.df[col].dropna()
                    if len(s) < 8:
                        ax_hist.text(
                            0.5,
                            0.5,
                            "Insufficient data",
                            ha="center",
                            va="center",
                            transform=ax_hist.transAxes,
                        )
                        ax_qq.text(
                            0.5,
                            0.5,
                            "Insufficient data",
                            ha="center",
                            va="center",
                            transform=ax_qq.transAxes,
                        )
                        for ax in (ax_hist, ax_qq):
                            ax.set_xticks([])
                            ax.set_yticks([])
                        continue

                    if len(s) <= 5000:
                        test_name = "Shapiro"
                        stat, p_val = stats.shapiro(s)
                    else:
                        test_name = "D'Agostino"
                        stat, p_val = stats.normaltest(s)

                    mu = float(s.mean())
                    sigma = float(s.std())
                    sigma = max(sigma, 1e-12)

                    sns.histplot(
                        s,
                        kde=True,
                        stat="density",
                        bins="auto",
                        color="#4C78A8",
                        edgecolor="white",
                        ax=ax_hist,
                    )
                    x_line = np.linspace(float(s.min()), float(s.max()), 200)
                    normal_pdf = stats.norm.pdf(x_line, loc=mu, scale=sigma)
                    ax_hist.plot(
                        x_line,
                        normal_pdf,
                        linestyle="--",
                        linewidth=2,
                        color="#E45756",
                        label="Normal fit",
                    )
                    ax_hist.legend(loc="best", fontsize=8)

                    stats.probplot(s, dist="norm", plot=ax_qq)

                    is_normal = bool(p_val >= alpha)
                    title_color = "green" if is_normal else "red"
                    title = f"{col} | {test_name} p={p_val:.3g}"
                    ax_hist.set_title(title, color=title_color, fontsize=10)
                    ax_qq.set_title(
                        f"Q-Q: {col} | {test_name} p={p_val:.3g}",
                        color=title_color,
                        fontsize=10,
                    )

                    skewness = float(s.skew())
                    kurtosis = float(s.kurtosis())
                    results.append(
                        {
                            "variable": col,
                            "test_used": test_name,
                            "statistic": float(stat),
                            "p_value": float(p_val),
                            "is_normal": is_normal,
                            "skewness": skewness,
                            "kurtosis": kurtosis,
                        }
                    )

                # Hide unused axes in both rows.
                total_slots = n_rows * n_cols
                for idx in range(n_vars, total_slots):
                    row = idx // n_cols
                    col_pos = idx % n_cols
                    axes_arr[row, col_pos].axis("off")
                    axes_arr[row + n_rows, col_pos].axis("off")

                all_axes = axes_arr.ravel().tolist()
                if any(len(str(c)) > 8 for c in cols_chunk):
                    for ax in all_axes:
                        for tick in ax.get_xticklabels():
                            tick.set_rotation(35)
                            tick.set_ha("right")

                group_no = start // chunk_size + 1
                n_groups = int(np.ceil(len(selected) / chunk_size))
                fig.suptitle(
                    f"Normality diagnostics — Group {group_no} of {n_groups}",
                    fontsize=12,
                )
                plt.tight_layout()
                plt.show()

            result_df = pd.DataFrame(results)
            if not result_df.empty:
                failed = result_df.loc[~result_df["is_normal"], "variable"].tolist()
                if failed:
                    self._warn(
                        "Columns failing normality: "
                        f"{failed}. This may affect model assumptions.",
                        severity="WARNING",
                        category="normality",
                        affected=", ".join(failed),
                    )
            self._normality_results = result_df
            return result_df
        except Exception as exc:  # noqa: BLE001
            self._warn(
                f"plot_normality failed and was skipped: {exc}",
                severity="WARNING",
                category="normality",
                affected="",
            )
            return pd.DataFrame(
                columns=[
                    "variable",
                    "test_used",
                    "statistic",
                    "p_value",
                    "is_normal",
                    "skewness",
                    "kurtosis",
                ]
            )

    # ------------------------------------------------------------------
    # Scatter matrix analysis
    # ------------------------------------------------------------------

    def plot_scatter(
        self,
        max_features: int = 8,
        sample_size: int = 2000,
        alpha: float = 0.4,
    ) -> None:
        """Build pairplot-style scatter matrices with matplotlib.

        Args:
            max_features: Maximum number of features considered.
            sample_size: Maximum rows used for plotting.
            alpha: Base marker transparency.

        Returns:
            None.

        Raises:
            ValueError: If parameters are out of valid ranges.
        """
        if max_features < 1:
            raise ValueError("max_features must be >= 1")
        if sample_size < 10:
            raise ValueError("sample_size must be >= 10")
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must be in (0, 1]")

        try:
            numeric = self._numeric_cols(exclude_id=True)
            if self.target_col and self.target_col in numeric:
                numeric = [c for c in numeric if c != self.target_col]

            if len(numeric) < 2:
                self._warn(
                    "Fewer than 2 numeric columns available for scatter matrix",
                    severity="WARNING",
                    category="scatter",
                    affected=", ".join(numeric),
                )
                return

            target_exists = self.target_col and self.target_col in self.df.columns
            target_is_numeric = False
            if target_exists:
                target_series = self.df[self.target_col]
                target_is_numeric = (
                    pd.api.types.is_numeric_dtype(target_series)
                    and target_series.dropna().nunique() > 20
                )

            if target_exists and target_is_numeric:
                corr_scores: dict[str, float] = {}
                for col in numeric:
                    valid = self.df[[col, self.target_col]].dropna()
                    if len(valid) < 3:
                        continue
                    corr = valid[col].corr(valid[self.target_col])
                    corr_scores[col] = float(abs(corr)) if pd.notna(corr) else 0.0
                ranked = sorted(corr_scores, key=corr_scores.get, reverse=True)
                features = ranked[:max_features]
            else:
                variances = self.df[numeric].var(numeric_only=True)
                features = variances.sort_values(ascending=False).head(
                    max_features
                ).index.tolist()

            if len(features) < 2:
                self._warn(
                    "Could not select at least 2 numeric features for scatter matrix",
                    severity="WARNING",
                    category="scatter",
                    affected=", ".join(features),
                )
                return

            plot_df = self.df.copy()
            sampled = False
            if len(plot_df) > sample_size:
                plot_df = plot_df.sample(sample_size, random_state=42)
                sampled = True

            n_pts = len(plot_df)
            if n_pts <= 500:
                marker_size = 24
                alpha_eff = max(alpha, 0.6)
            elif n_pts <= 2000:
                marker_size = 12
                alpha_eff = alpha
            else:
                marker_size = 7
                alpha_eff = min(alpha, 0.3)

            groups: list[list[str]] = []
            if max_features > 5 and len(features) > 5:
                window = 5
                step = 3
                for i in range(0, len(features), step):
                    group = features[i:i + window]
                    if len(group) >= 2:
                        groups.append(group)
                    if i + window >= len(features):
                        break
            else:
                groups = [features]

            is_cat_target_small = False
            if target_exists and not target_is_numeric:
                is_cat_target_small = (
                    plot_df[self.target_col].dropna().nunique() <= 10
                )

            for g_idx, group in enumerate(groups, start=1):
                n = len(group)
                fig, axes = plt.subplots(n, n, figsize=self._get_figsize(n, n, 3.0))
                axes_arr = np.array(axes, dtype=object)
                if axes_arr.ndim == 1:
                    axes_arr = axes_arr.reshape(n, n)

                color_values = None
                cat_map: dict[Any, int] | None = None
                if target_exists and target_is_numeric:
                    color_values = plot_df[self.target_col]
                elif target_exists and is_cat_target_small:
                    classes = sorted(plot_df[self.target_col].dropna().unique())
                    cat_map = {cls: i for i, cls in enumerate(classes)}
                    color_values = plot_df[self.target_col].map(cat_map)

                for i, y_col in enumerate(group):
                    for j, x_col in enumerate(group):
                        ax = axes_arr[i, j]
                        if i == j:
                            vals = plot_df[x_col].dropna().to_numpy()
                            if len(vals) > 1 and np.std(vals) > 0:
                                kde = stats.gaussian_kde(vals)
                                x_grid = np.linspace(vals.min(), vals.max(), 200)
                                ax.plot(x_grid, kde(x_grid), color="#4C78A8")
                            else:
                                ax.hist(vals, bins=20, color="#4C78A8")
                        else:
                            if target_exists and target_is_numeric:
                                sc = ax.scatter(
                                    plot_df[x_col],
                                    plot_df[y_col],
                                    c=color_values,
                                    cmap="viridis",
                                    s=marker_size,
                                    alpha=alpha_eff,
                                    linewidths=0,
                                )
                            elif target_exists and is_cat_target_small and cat_map:
                                ax.scatter(
                                    plot_df[x_col],
                                    plot_df[y_col],
                                    c=color_values,
                                    cmap="tab10",
                                    s=marker_size,
                                    alpha=alpha_eff,
                                    linewidths=0,
                                )
                            else:
                                ax.scatter(
                                    plot_df[x_col],
                                    plot_df[y_col],
                                    color="#4C78A8",
                                    s=marker_size,
                                    alpha=alpha_eff,
                                    linewidths=0,
                                )

                        if i == n - 1:
                            ax.set_xlabel(x_col)
                        else:
                            ax.set_xlabel("")
                        if j == 0:
                            ax.set_ylabel(y_col)
                        else:
                            ax.set_ylabel("")

                title = "Scatter Matrix"
                if len(groups) > 1:
                    title += f" — Group {g_idx} of {len(groups)}"
                if sampled:
                    title += " (sampled)"
                fig.suptitle(title, fontsize=13)

                if target_exists and target_is_numeric:
                    cbar = fig.colorbar(sc, ax=axes_arr.ravel().tolist(),
                                        fraction=0.02, pad=0.01)
                    cbar.set_label(str(self.target_col))
                elif target_exists and is_cat_target_small and cat_map:
                    handles = [
                        plt.Line2D(
                            [0],
                            [0],
                            marker="o",
                            linestyle="",
                            label=str(cls),
                            markerfacecolor=plt.get_cmap("tab10")(idx),
                            markeredgecolor="none",
                            markersize=7,
                        )
                        for cls, idx in cat_map.items()
                    ]
                    fig.legend(handles=handles, title=str(self.target_col),
                               loc="upper right")

                if any(len(str(name)) > 8 for name in group):
                    for ax in axes_arr.ravel().tolist():
                        for tick in ax.get_xticklabels():
                            tick.set_rotation(35)
                            tick.set_ha("right")

                plt.tight_layout()
                plt.show()
        except Exception as exc:  # noqa: BLE001
            self._warn(
                f"plot_scatter failed and was skipped: {exc}",
                severity="WARNING",
                category="scatter",
                affected="",
            )

    # ------------------------------------------------------------------
    # Scatter vs target
    # ------------------------------------------------------------------

    def plot_scatter_vs_target(
        self,
        top_n: int = 12,
        sample_size: int = 3000,
    ) -> None:
        """Plot individual feature-vs-target scatter panels.

        Args:
            top_n: Number of top associated numeric features to visualize.
            sample_size: Maximum rows used for plotting.

        Returns:
            None.

        Raises:
            ValueError: If ``top_n`` or ``sample_size`` are invalid.
        """
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if sample_size < 10:
            raise ValueError("sample_size must be >= 10")

        try:
            if not self.target_col or self.target_col not in self.df.columns:
                self._print("\n=== Scatter vs Target ===")
                self._print("  No target column specified; skipping.")
                return

            target = self.df[self.target_col]
            numeric_features = self._numeric_cols(exclude_id=True)
            numeric_features = [
                c for c in numeric_features if c != self.target_col
            ]

            if not numeric_features:
                self._warn(
                    "No numeric features available for scatter-vs-target plot",
                    severity="INFO",
                    category="scatter_vs_target",
                    affected="",
                )
                return

            target_is_numeric = (
                pd.api.types.is_numeric_dtype(target)
                and target.dropna().nunique() > 20
            )

            scores: dict[str, float] = {}
            if target_is_numeric:
                for col in numeric_features:
                    valid = self.df[[col, self.target_col]].dropna()
                    if len(valid) < 3:
                        continue
                    corr = valid[col].corr(valid[self.target_col])
                    scores[col] = float(abs(corr)) if pd.notna(corr) else 0.0
            else:
                variances = self.df[numeric_features].var(numeric_only=True)
                scores = variances.to_dict()

            selected = sorted(scores, key=scores.get, reverse=True)[:top_n]
            if not selected:
                self._warn(
                    "Could not compute feature scores for scatter-vs-target",
                    severity="INFO",
                    category="scatter_vs_target",
                    affected="",
                )
                return

            plot_df = self.df[selected + [self.target_col]].copy()
            if len(plot_df) > sample_size:
                plot_df = plot_df.sample(sample_size, random_state=42)

            n_rows, n_cols = self._get_grid_dims(len(selected))
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=self._get_figsize(n_rows, n_cols, cell_size=4.1),
            )
            axes_arr = np.array(axes, dtype=object)
            if axes_arr.ndim == 0:
                axes_arr = axes_arr.reshape(1)
            axes_flat = axes_arr.ravel().tolist()

            class_map: dict[Any, int] = {}
            if not target_is_numeric:
                classes = sorted(plot_df[self.target_col].dropna().unique())
                class_map = {c: i for i, c in enumerate(classes)}

            for i, feat in enumerate(selected):
                ax = axes_flat[i]

                valid = plot_df[[feat, self.target_col]].dropna()
                if valid.empty:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                            ha="center", va="center")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                x = valid[feat].to_numpy(dtype=float)

                if target_is_numeric:
                    y = valid[self.target_col].to_numpy(dtype=float)
                    ax.scatter(x, y, s=13, alpha=0.45, color="#4C78A8")

                    if len(x) > 2 and np.std(x) > 0:
                        slope, intercept = np.polyfit(x, y, deg=1)
                        x_line = np.linspace(np.min(x), np.max(x), 200)
                        y_line = slope * x_line + intercept
                        y_hat = slope * x + intercept
                        resid = y - y_hat
                        s_err = np.sqrt(np.sum(resid ** 2) / max(len(x) - 2, 1))
                        x_center = np.mean(x)
                        denom = np.sum((x - x_center) ** 2)
                        if denom > 0:
                            conf = 1.96 * s_err * np.sqrt(
                                1 / len(x) + (x_line - x_center) ** 2 / denom
                            )
                            ax.fill_between(
                                x_line,
                                y_line - conf,
                                y_line + conf,
                                color="#72B7B2",
                                alpha=0.2,
                            )
                        ax.plot(x_line, y_line, color="#E45756", linewidth=2)
                    corr_val = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
                    ax.set_ylabel(str(self.target_col))
                else:
                    y_enc = valid[self.target_col].map(class_map).to_numpy(float)
                    jitter = np.random.default_rng(42).normal(0, 0.08, size=len(y_enc))
                    y = y_enc + jitter
                    ax.scatter(
                        x,
                        y,
                        c=y_enc,
                        cmap="tab10",
                        s=13,
                        alpha=0.5,
                    )
                    corr_val = (
                        np.corrcoef(x, y_enc)[0, 1] if len(x) > 1 else np.nan
                    )
                    ax.set_yticks(list(class_map.values()))
                    ax.set_yticklabels([str(c) for c in class_map.keys()])
                    ax.set_ylabel(str(self.target_col))

                ax.set_xlabel(feat)
                title_corr = f"{corr_val:.2f}" if pd.notna(corr_val) else "nan"
                ax.set_title(f"{feat} | r = {title_corr}")

                if len(str(feat)) > 8:
                    for tick in ax.get_xticklabels():
                        tick.set_rotation(35)
                        tick.set_ha("right")

            for j in range(len(selected), len(axes_flat)):
                axes_flat[j].axis("off")

            plt.tight_layout()
            plt.show()
        except Exception as exc:  # noqa: BLE001
            self._warn(
                f"plot_scatter_vs_target failed and was skipped: {exc}",
                severity="WARNING",
                category="scatter_vs_target",
                affected="",
            )

    # ------------------------------------------------------------------
    # Sweetviz report
    # ------------------------------------------------------------------

    def generate_sweetviz_report(
        self,
        output_path: str = "eda_report.html",
        compare_by_target: bool = False,
        max_features: int = 50,
        sample_size: int | None = None,
    ) -> None:
        """Generate a Sweetviz HTML report safely.

        Args:
            output_path: Destination HTML file path.
            compare_by_target: Whether to run intra-target comparison mode.
            max_features: Maximum number of columns used in the report.
            sample_size: Optional explicit row sample size.

        Returns:
            None.

        Raises:
            ValueError: If ``max_features`` or ``sample_size`` are invalid.
        """
        if max_features < 1:
            raise ValueError("max_features must be >= 1")
        if sample_size is not None and sample_size < 100:
            raise ValueError("sample_size must be >= 100 when provided")

        df_report = self.df.copy()

        if df_report.shape[1] > max_features:
            must_keep = {
                c
                for c in [self.id_col, self.target_col]
                if c and c in df_report.columns
            }
            remaining = [c for c in df_report.columns if c not in must_keep]
            n_to_sample = max_features - len(must_keep)
            n_to_sample = max(n_to_sample, 0)

            sampled_cols = []
            if n_to_sample > 0 and remaining:
                rng = np.random.default_rng(42)
                sampled_cols = rng.choice(
                    remaining,
                    size=min(n_to_sample, len(remaining)),
                    replace=False,
                ).tolist()

            keep_cols = [
                c for c in df_report.columns if c in must_keep or c in sampled_cols
            ]
            excluded = [c for c in df_report.columns if c not in keep_cols]
            df_report = df_report[keep_cols]
            self._warn(
                "Sweetviz limited by max_features. Excluded columns: "
                f"{excluded}",
                severity="INFO",
                category="sweetviz",
                affected=", ".join(excluded),
            )

        auto_sample = sample_size is None and len(df_report) > 100_000
        if sample_size is not None or auto_sample:
            n_rows = sample_size if sample_size is not None else 50_000
            n_rows = min(n_rows, len(df_report))
            if n_rows < len(df_report):
                df_report = df_report.sample(n_rows, random_state=42)
                self._warn(
                    f"Sweetviz report generated on a {n_rows}-row sample.",
                    severity="INFO",
                    category="sweetviz",
                    affected="",
                )

        try:
            import sweetviz as sv
        except ImportError:
            self._print(
                "Sweetviz is not installed. Install with: pip install sweetviz"
            )
            return

        try:
            report = None
            target_ok = (
                self.target_col
                and self.target_col in df_report.columns
            )

            if not target_ok:
                report = sv.analyze(df_report)
            else:
                target = df_report[self.target_col]
                target_is_continuous = (
                    pd.api.types.is_numeric_dtype(target)
                    and target.dropna().nunique() > 20
                )

                if compare_by_target:
                    if target_is_continuous:
                        self._warn(
                            "compare_by_target=True requires a binary/"
                            "categorical target. Falling back to analyze "
                            "with target feature.",
                            severity="WARNING",
                            category="sweetviz",
                            affected=self.target_col,
                        )
                        report = sv.analyze(df_report, target_feat=self.target_col)
                    else:
                        value = target.dropna().iloc[0] if target.notna().any() else None
                        if value is None:
                            report = sv.analyze(df_report, target_feat=self.target_col)
                        else:
                            report = sv.compare_intra(
                                df_report,
                                df_report[self.target_col] == value,
                                self.target_col,
                            )
                else:
                    report = sv.analyze(df_report, target_feat=self.target_col)

            resolved = str(Path(output_path).expanduser().resolve())
            report.show_html(filepath=resolved, open_browser=False)
            self._print(f"Sweetviz report saved to: {resolved}")
        except Exception as exc:  # noqa: BLE001
            self._print(f"Sweetviz generation failed: {exc}")
            self._print(traceback.format_exc())

    # ------------------------------------------------------------------
    # Integration summary
    # ------------------------------------------------------------------

    def get_pipeline_summary(self) -> dict[str, Any]:
        """Create a standardized summary dict for downstream modules.

        Args:
            None.

        Returns:
            Dictionary with shape, schema slices, quality indicators,
            alert summary, and metadata keys expected by the next module.

        Raises:
            None.
        """
        try:
            df = self.df
            n_rows, n_cols = df.shape

            null_columns = df.columns[df.isnull().any()].tolist()
            high_null_columns = df.columns[(df.isnull().mean() > 0.30)].tolist()

            numeric_cols = self._numeric_cols(exclude_id=False)
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

            low_variance_cols = self.detect_low_variance()

            outlier_df = self.detect_outliers()
            outlier_cols: list[str] = []
            if not outlier_df.empty and "pct_outliers" in outlier_df.columns:
                outlier_cols = outlier_df.index[
                    outlier_df["pct_outliers"] > 5
                ].tolist()

            if self._normality_results is None:
                self._normality_results = self.plot_normality()
            non_normal_cols: list[str] = []
            if (
                self._normality_results is not None
                and not self._normality_results.empty
                and "is_normal" in self._normality_results.columns
            ):
                non_normal_cols = self._normality_results.loc[
                    ~self._normality_results["is_normal"], "variable"
                ].tolist()

            duplicate_info = self.analyze_duplicates()
            duplicate_rows = int(duplicate_info.get("full_duplicates", 0))
            duplicate_ids = duplicate_info.get("duplicated_ids")

            tidy_result = self.check_tidy_format()
            is_tidy = bool(tidy_result.get("is_tidy", False))

            alerts = self.generate_alert_summary()

            return {
                "shape": (n_rows, n_cols),
                "null_columns": null_columns,
                "high_null_columns": high_null_columns,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "datetime_cols": datetime_cols,
                "boolean_cols": boolean_cols,
                "freetext_cols": freetext_cols,
                "low_variance_cols": low_variance_cols,
                "outlier_cols": outlier_cols,
                "non_normal_cols": non_normal_cols,
                "duplicate_rows": duplicate_rows,
                "duplicate_ids": duplicate_ids,
                "target_col": self.target_col,
                "id_col": self.id_col,
                "is_tidy": is_tidy,
                "alerts": alerts,
            }
        except Exception as exc:  # noqa: BLE001
            self._warn(
                f"get_pipeline_summary failed and returned fallback values: {exc}",
                severity="WARNING",
                category="pipeline_summary",
                affected="",
            )
            return {
                "shape": tuple(self.df.shape),
                "null_columns": [],
                "high_null_columns": [],
                "numeric_cols": [],
                "categorical_cols": [],
                "datetime_cols": [],
                "boolean_cols": [],
                "freetext_cols": [],
                "low_variance_cols": [],
                "outlier_cols": [],
                "non_normal_cols": [],
                "duplicate_rows": 0,
                "duplicate_ids": None,
                "target_col": self.target_col,
                "id_col": self.id_col,
                "is_tidy": False,
                "alerts": self.generate_alert_summary(),
            }

    # ------------------------------------------------------------------
    # Executive alert summary
    # ------------------------------------------------------------------

    def generate_alert_summary(self) -> pd.DataFrame:
        """Compile all warnings issued during the analysis.

        Returns:
            A DataFrame with columns ``severity``, ``category``,
            ``message``, and ``affected_columns``.
        """
        if not self._alerts:
            self._print("\n=== Alert Summary ===")
            self._print("  No issues detected. ✓")
            return pd.DataFrame(
                columns=["severity", "category", "message",
                         "affected_columns"]
            )

        summary = pd.DataFrame(self._alerts)

        # Sort by severity priority.
        severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
        summary["_order"] = summary["severity"].map(severity_order)
        summary = summary.sort_values("_order").drop(columns="_order")
        summary = summary.reset_index(drop=True)

        self._print("\n=== Alert Summary ===")
        self._print(summary.to_string(index=False))
        return summary

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------

    def run_full_eda(self) -> dict[str, Any]:
        """Execute all analysis steps and return consolidated results.

        Args:
            None.

        Returns:
            A dict keyed by method name with the output of each step.

        Raises:
            None.
        """
        results: dict[str, Any] = {}

        results["tidy_format"] = self.check_tidy_format()
        results["structural_summary"] = self.get_structural_summary()
        results["null_analysis"] = self.analyze_nulls()
        results["duplicate_analysis"] = self.analyze_duplicates()
        results["target_analysis"] = self.analyze_target()
        results["low_variance"] = self.detect_low_variance()
        results["outlier_detection"] = self.detect_outliers()

        try:
            results["normality"] = self.plot_normality()
        except Exception:  # noqa: BLE001
            results["normality"] = pd.DataFrame()
            self._print("  (Skipped normality analysis due to an error)")

        # Plotting — wrapped in try/except to avoid breaking headless envs.
        try:
            self.plot_correlation_heatmap()
        except Exception:  # noqa: BLE001
            self._print("  (Skipped correlation heatmap — display unavailable)")

        try:
            results["target_correlations"] = self.plot_target_correlations()
        except Exception:  # noqa: BLE001
            results["target_correlations"] = None
            self._print(
                "  (Skipped target correlations — display unavailable)"
            )

        try:
            self.plot_scatter()
            results["scatter_matrix"] = "completed"
        except Exception:  # noqa: BLE001
            results["scatter_matrix"] = "skipped"
            self._print("  (Skipped scatter matrix due to an error)")

        if self.target_col and self.target_col in self.df.columns:
            try:
                self.plot_scatter_vs_target()
                results["scatter_vs_target"] = "completed"
            except Exception:  # noqa: BLE001
                results["scatter_vs_target"] = "skipped"
                self._print("  (Skipped scatter-vs-target due to an error)")
        else:
            results["scatter_vs_target"] = None

        try:
            self.generate_sweetviz_report()
            results["sweetviz_report"] = "completed"
        except Exception:  # noqa: BLE001
            results["sweetviz_report"] = "skipped"
            self._print("  (Skipped Sweetviz report due to an error)")

        results["alert_summary"] = self.generate_alert_summary()
        try:
            results["pipeline_summary"] = self.get_pipeline_summary()
        except Exception:  # noqa: BLE001
            results["pipeline_summary"] = {}
            self._print("  (Could not generate pipeline summary)")
        return results


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # Generate a synthetic classification dataset.
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[0.85, 0.15],
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    df["id"] = range(len(df))

    # Inject some nulls for demonstration.
    rng = np.random.default_rng(42)
    for col in ["feature_0", "feature_3"]:
        mask = rng.random(len(df)) < 0.35
        df.loc[mask, col] = np.nan

    # Run the full EDA.
    explorer = DataExplorer(
        df,
        id_col="id",
        target_col="target",
        verbose=True,
        outlier_method="iqr",
    )
    results = explorer.run_full_eda()

    print("\n\n=== Keys returned by run_full_eda() ===")
    for key in results:
        print(f"  {key}: {type(results[key]).__name__}")
