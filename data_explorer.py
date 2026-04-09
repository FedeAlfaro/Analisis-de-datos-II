"""Data Explorer — Part 1 of a modular ML pipeline.

This module provides the ``DataExplorer`` class, a comprehensive exploratory
data analysis (EDA) toolkit designed to integrate cleanly with downstream
pipeline stages (data cleaning, feature engineering, model training).

Every public method **returns** structured objects (DataFrames, dicts, Series)
so that results can be consumed programmatically.  When ``verbose=True``
(the default), human-readable summaries are also printed to stdout.

Usage example
-------------
>>> import pandas as pd
>>> from data_explorer import DataExplorer
>>> df = pd.read_csv("my_dataset.csv")
>>> explorer = DataExplorer(df, target_col="label", id_col="id")
>>> results = explorer.run_full_eda()

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
import warnings
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
        for rec in records:
            if rec["pct_outliers"] > 10:
                self._warn(
                    f"Column '{rec['column']}' has {rec['pct_outliers']}% "
                    f"outliers (>{10}%)",
                    severity="WARNING", category="outliers",
                    affected=rec["column"],
                )
            elif rec["pct_outliers"] > 5:
                self._warn(
                    f"Column '{rec['column']}' has {rec['pct_outliers']}% "
                    f"outliers (>{5}%)",
                    severity="INFO", category="outliers",
                    affected=rec["column"],
                )

        self._print("\n=== Outlier Detection ===")
        self._print(f"  Method: {self.outlier_method}, "
                     f"threshold: {self.outlier_threshold}")
        if not result.empty:
            self._print(result.to_string())
        else:
            self._print("  No numeric columns to analyse.")
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

        Returns:
            A dict keyed by method name with the output of each step.
        """
        results: dict[str, Any] = {}

        results["tidy_format"] = self.check_tidy_format()
        results["structural_summary"] = self.get_structural_summary()
        results["null_analysis"] = self.analyze_nulls()
        results["duplicate_analysis"] = self.analyze_duplicates()
        results["target_analysis"] = self.analyze_target()
        results["low_variance"] = self.detect_low_variance()
        results["outlier_detection"] = self.detect_outliers()

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

        results["alert_summary"] = self.generate_alert_summary()
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
