"""Módulo de regresión comparativa y selección de modelos.

Este módulo ofrece una implementación estandarizada para preparar datos,
comparar múltiples modelos de regresión, seleccionar el mejor en función de
métricas configurables y entrenar un modelo final robusto con validación cruzada.

El módulo es tolerante a errores y maneja importaciones opcionales de estimadores
avanzados como XGBoost, LightGBM, CatBoost, Mars y RuleFit.

Uso básico
-----------
>>> from data_reggresion import DataRegresion
>>> cleaner = DataRegresion(df, target_col='y', id_col='id')
>>> cleaner.split_data(train_size=0.6, val_size=0.2, test_size=0.2)
>>> report = cleaner.compare_models(metrics=['rmse', 'r2'])
>>> cleaner.train_best_model(metric='rmse')
>>> cleaner.export_model_artifact('best_model.pkl')
"""

from __future__ import annotations

import importlib
import json
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


class RegressorImportWarning(UserWarning):
    """Advertencia para regresores opcionales no disponibles."""


def _optional_import(module_name: str, class_name: str) -> Any | None:
    """Import a class optionally, retornando None si no está disponible."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        warnings.warn(
            f"No se encontró {class_name} en {module_name}; se omitirá.",
            RegressorImportWarning,
            stacklevel=3,
        )
        return None


@dataclass
class ModelResult:
    model_name: str
    mse: float
    mae: float
    rmse: float
    r2: float
    aic: float | None
    bic: float | None
    metrics: dict[str, float]


class DataRegresion:
    """Clase para comparar y seleccionar modelos de regresión.

    Args:
        df: DataFrame de entrada.
        target_col: Nombre de la variable objetivo.
        id_col: Nombre de la columna identificadora opcional.
        verbose: Si ``True`` imprime progreso y resúmenes.

    """

    DEFAULT_MODELS = {
        "linear_regression": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "decision_tree": DecisionTreeRegressor,
        "knn": KNeighborsRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "random_forest": RandomForestRegressor,
        "bagging": BaggingRegressor,
        "bayesian_ridge": BayesianRidge,
    }

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        id_col: str | None = None,
        verbose: bool = True,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas DataFrame")
        if target_col not in df.columns:
            raise ValueError("target_col debe existir en el DataFrame")

        self.df_original = df.copy(deep=True)
        self.df = df.copy(deep=True)
        self.target_col = target_col
        self.id_col = id_col
        self.verbose = verbose

        self.available_models: dict[str, Any] = self._load_models()
        self.split_seed = 42
        self.preprocessor: ColumnTransformer | None = None
        self.best_model_pipeline: Pipeline | None = None
        self.best_model_name: str | None = None
        self.model_metadata: dict[str, Any] = {}
        self.comparison_report: pd.DataFrame | None = None
        self.train_sets: dict[str, Any] = {}

    def _print(self, *args: Any, **kwargs: Any) -> None:
        if self.verbose:
            print(*args, **kwargs)

    def _load_models(self) -> dict[str, Any]:
        models = self.DEFAULT_MODELS.copy()
        lightgbm_cls = _optional_import("lightgbm", "LGBMRegressor")
        xgb_cls = _optional_import("xgboost", "XGBRegressor")
        catboost_cls = _optional_import("catboost", "CatBoostRegressor")
        earth_cls = _optional_import("pyearth", "Earth")
        rulefit_cls = _optional_import("skrule", "RuleFit") or _optional_import("rulefit", "RuleFit")

        if lightgbm_cls is not None:
            models["lightgbm"] = lightgbm_cls
        if xgb_cls is not None:
            models["xgboost"] = xgb_cls
        if catboost_cls is not None:
            models["catboost"] = catboost_cls
        if earth_cls is not None:
            models["mars"] = earth_cls
        if rulefit_cls is not None:
            models["rulefit"] = rulefit_cls

        return models

    def _safe_model_instance(self, name: str) -> Any:
        cls = self.available_models[name]
        if name == "lasso":
            return cls(alpha=0.001, max_iter=10000)
        if name == "ridge":
            return cls(alpha=1.0)
        if name == "decision_tree":
            return cls(max_depth=6, random_state=self.split_seed)
        if name == "knn":
            return cls(n_neighbors=5)
        if name == "gradient_boosting":
            return cls(n_estimators=100, learning_rate=0.1, random_state=self.split_seed)
        if name == "random_forest":
            return cls(n_estimators=100, max_depth=8, random_state=self.split_seed, n_jobs=-1)
        if name == "bagging":
            return cls(n_estimators=10, random_state=self.split_seed, n_jobs=-1)
        if name == "bayesian_ridge":
            return cls()
        if name == "lightgbm":
            return cls(n_estimators=100, learning_rate=0.1, random_state=self.split_seed)
        if name == "xgboost":
            return cls(n_estimators=100, learning_rate=0.1, random_state=self.split_seed, verbosity=0)
        if name == "catboost":
            return cls(iterations=100, learning_rate=0.1, verbose=0, random_seed=self.split_seed)
        if name == "mars":
            return cls(max_degree=2)
        if name == "rulefit":
            return cls(tree_size=4, sample_fract='default', max_rules=2000)
        if name == "linear_regression":
            return cls()
        raise ValueError(f"Modelo desconocido {name}")

    def _numeric_columns(self) -> list[str]:
        return [
            c for c in self.df.select_dtypes(include="number").columns.tolist()
            if c != self.target_col and c != self.id_col
        ]

    def _categorical_columns(self) -> list[str]:
        return [
            c for c in self.df.select_dtypes(include=["object", "category"]).columns.tolist()
            if c != self.target_col and c != self.id_col
        ]

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_cols = self._numeric_columns()
        categorical_cols = self._categorical_columns()

        numeric_transformers: list[tuple[str, Pipeline, list[str]]] = []
        if numeric_cols:
            low_card_cols = [
                col for col in numeric_cols if self.df[col].nunique(dropna=True) <= 10
            ]
            high_card_cols = [
                col for col in numeric_cols if self.df[col].nunique(dropna=True) > 10
            ]

            if low_card_cols:
                numeric_transformers.append(
                    (
                        "num_median",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", MinMaxScaler()),
                            ]
                        ),
                        low_card_cols,
                    )
                )
            if high_card_cols:
                numeric_transformers.append(
                    (
                        "num_mean",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="mean")),
                                ("scaler", MinMaxScaler()),
                            ]
                        ),
                        high_card_cols,
                    )
                )
        
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        transformer = ColumnTransformer(
            transformers=[*numeric_transformers, ("cat", categorical_pipeline, categorical_cols)],
            remainder="drop",
        )
        self.preprocessor = transformer
        return transformer

    def _make_stratify_bins(self, y: pd.Series, bins: int = 10) -> pd.Series:
        if y.dtype.kind in "bifc" and y.nunique() > bins:
            try:
                return pd.qcut(y, q=bins, duplicates="drop")
            except ValueError:
                return pd.cut(y, bins=bins, duplicates="drop")
        return y.astype(str)

    def split_data(
        self,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
    ) -> None:
        """Divide el conjunto en entrenamiento, validación y prueba.

        Args:
            train_size: Proporción de entrenamiento.
            val_size: Proporción de validación.
            test_size: Proporción de prueba.

        Returns:
            None.

        Raises:
            ValueError: Si las proporciones no suman a 1.
        """
        total = train_size + val_size + test_size
        if not np.isclose(total, 1.0):
            raise ValueError("train_size + val_size + test_size debe ser 1.0")

        y = self.df[self.target_col]
        X = self.df.drop(columns=[self.target_col])

        if y.dtype.kind in "bifc" and y.nunique() > 1:
            stratify = self._make_stratify_bins(y)
            X_temp, X_test, y_temp, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=self.split_seed,
                stratify=stratify,
            )
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=self.split_seed,
            )

        val_fraction = val_size / (train_size + val_size)
        if y_temp.dtype.kind in "bifc" and y_temp.nunique() > 1:
            stratify_temp = self._make_stratify_bins(y_temp)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=val_fraction,
                random_state=self.split_seed,
                stratify=stratify_temp,
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=val_fraction,
                random_state=self.split_seed,
            )

        self.train_sets = {
            "train": (X_train.reset_index(drop=True), y_train.reset_index(drop=True)),
            "validation": (X_val.reset_index(drop=True), y_val.reset_index(drop=True)),
            "test": (X_test.reset_index(drop=True), y_test.reset_index(drop=True)),
        }
        self._print("Datos divididos con estratificación." )
        self._print(
            f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

    def _regression_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model: Any,
    ) -> dict[str, float | None]:
        n = len(y_true)
        mse = float(mean_squared_error(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        aic, bic = self._compute_aic_bic(y_true, y_pred, model)
        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "aic": aic,
            "bic": bic,
        }

    def _compute_aic_bic(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model: Any,
    ) -> tuple[float | None, float | None]:
        n = len(y_true)
        residuals = y_true.to_numpy(dtype=float) - y_pred
        rss = float(np.sum(residuals ** 2))
        if rss <= 0 or n <= 0:
            return None, None
        sigma2 = rss / n
        if sigma2 <= 0:
            return None, None
        log_lik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
        k = self._effective_parameter_count(model)
        if k is None:
            return None, None
        aic = 2 * k - 2 * log_lik
        bic = np.log(n) * k - 2 * log_lik
        return float(aic), float(bic)

    def _effective_parameter_count(self, model: Any) -> int | None:
        if hasattr(model, "coef_"):
            coef = getattr(model, "coef_")
            return int(np.prod(coef.shape) + 1)
        if hasattr(model, "n_features_in_"):
            return int(getattr(model, "n_features_in_") + 1)
        return None

    def compare_models(
        self,
        metrics: list[str] | str = ["rmse"],
        regression_models: list[str] | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Comparar modelos de regresión en el conjunto de validación.

        Args:
            metrics: Métrica o lista de métricas para ordenar la comparación.
                Soportadas: ``mse``, ``mae``, ``rmse``, ``r2``, ``aic``, ``bic``.
            regression_models: Lista de nombres de modelos a evaluar.
            verbose: Si ``True`` imprime resultados.

        Returns:
            DataFrame con métricas de cada modelo.
        """
        if self.train_sets == {}:
            raise RuntimeError("Debe llamar a split_data() antes de comparar modelos")

        if isinstance(metrics, str):
            metrics = [metrics]
        metrics = [m.lower() for m in metrics]
        supported = {"mse", "mae", "rmse", "r2", "aic", "bic"}
        if not all(m in supported for m in metrics):
            raise ValueError(f"Métricas soportadas: {sorted(supported)}")

        models_to_use = regression_models or list(self.available_models.keys())
        models_to_use = [m for m in models_to_use if m in self.available_models]

        if not models_to_use:
            raise RuntimeError("No hay modelos disponibles para comparar")

        self._build_preprocessor()
        X_train, y_train = self.train_sets["train"]
        X_val, y_val = self.train_sets["validation"]

        rows: list[dict[str, Any]] = []
        for name in models_to_use:
            try:
                estimator = self._safe_model_instance(name)
                pipeline = Pipeline(
                    steps=[("preprocessor", self.preprocessor), ("model", estimator)]
                )
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_val)
                scores = self._regression_metrics(y_val, y_pred, estimator)
                row = {"model": name, **scores}
                row["status"] = "ok"
                rows.append(row)
                self._print(f"Modelo {name}: RMSE={scores['rmse']:.4f}, R2={scores['r2']:.4f}")
            except Exception as exc:
                rows.append(
                    {
                        "model": name,
                        "mse": np.nan,
                        "mae": np.nan,
                        "rmse": np.nan,
                        "r2": np.nan,
                        "aic": None,
                        "bic": None,
                        "status": f"failed: {exc}",
                    }
                )
                warnings.warn(f"Fallo al entrenar {name}: {exc}")

        report = pd.DataFrame(rows)
        sort_key = metrics[0]
        ascending = sort_key != "r2"
        if sort_key in report.columns:
            report = report.sort_values(by=sort_key, ascending=ascending)

        self.comparison_report = report.reset_index(drop=True)
        if verbose:
            self._print("\n=== Comparación de modelos ===")
            self._print(self.comparison_report.to_string(index=False))

        return self.comparison_report

    def _param_grid_for_model(self, name: str) -> dict[str, list[Any]]:
        grids = {
            "linear_regression": {},
            "ridge": {"model__alpha": [0.1, 1.0, 10.0]},
            "lasso": {"model__alpha": [0.0001, 0.001, 0.01]},
            "decision_tree": {"model__max_depth": [4, 6, 8], "model__min_samples_split": [2, 5]},
            "knn": {"model__n_neighbors": [3, 5, 7]},
            "gradient_boosting": {"model__n_estimators": [50, 100], "model__learning_rate": [0.05, 0.1]},
            "random_forest": {"model__n_estimators": [50, 100], "model__max_depth": [6, 8]},
            "bagging": {"model__n_estimators": [10, 20]},
            "bayesian_ridge": {},
            "lightgbm": {"model__n_estimators": [50, 100], "model__learning_rate": [0.05, 0.1]},
            "xgboost": {"model__n_estimators": [50, 100], "model__learning_rate": [0.05, 0.1]},
            "catboost": {"model__iterations": [50, 100], "model__learning_rate": [0.05, 0.1]},
            "mars": {"model__max_degree": [1, 2]},
            "rulefit": {},
        }
        return grids.get(name, {})

    def train_best_model(
        self,
        metric: str = "rmse",
        cv: int = 5,
        scoring: str | None = None,
    ) -> None:
        """Selecciona y entrena el mejor modelo con validación cruzada.

        Args:
            metric: Métrica usada para seleccionar el mejor modelo.
            cv: Número de folds para cross-validation.
            scoring: Nombre de la métrica de sklearn para optimización.

        Returns:
            None.

        Raises:
            RuntimeError: Si no hay reportes de comparación disponibles.
        """
        if self.comparison_report is None or self.comparison_report.empty:
            raise RuntimeError("Debe llamar a compare_models() antes de train_best_model()")

        metric = metric.lower()
        supported = {"mse", "mae", "rmse", "r2", "aic", "bic"}
        if metric not in supported:
            raise ValueError(f"Métrica no soportada: {metric}")

        if scoring is None:
            scoring = "neg_mean_squared_error" if metric in {"mse", "rmse"} else (
                "neg_mean_absolute_error" if metric == "mae" else "r2"
            )

        ascending = metric != "r2"
        report = self.comparison_report.copy()
        report = report[report["status"] == "ok"]
        if report.empty:
            raise RuntimeError("No hay modelos válidos para elegir")

        best_row = report.sort_values(by=metric, ascending=ascending).iloc[0]
        best_name = best_row["model"]
        self.best_model_name = best_name

        X_train, y_train = self.train_sets["train"]
        X_val, y_val = self.train_sets["validation"]
        X_full = pd.concat([X_train, X_val], ignore_index=True)
        y_full = pd.concat([y_train, y_val], ignore_index=True)

        estimator = self._safe_model_instance(best_name)
        pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("model", estimator)]
        )

        param_grid = self._param_grid_for_model(best_name)
        if param_grid:
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                scoring=scoring,
                cv=KFold(n_splits=cv, shuffle=True, random_state=self.split_seed),
                n_jobs=-1,
                error_score="raise",
            )
            search.fit(X_full, y_full)
            self.best_model_pipeline = search.best_estimator_
            self.model_metadata["best_params"] = search.best_params_
        else:
            pipeline.fit(X_full, y_full)
            self.best_model_pipeline = pipeline
            self.model_metadata["best_params"] = {}

        self.model_metadata["selected_model"] = best_name
        self.model_metadata["metric_selected"] = metric
        self.model_metadata["scoring"] = scoring
        self.model_metadata["cv_folds"] = cv

        X_test, y_test = self.train_sets["test"]
        predictions = self.best_model_pipeline.predict(X_test)
        scores = self._regression_metrics(y_test, predictions, self.best_model_pipeline.named_steps["model"])

        self.model_metadata["final_test_metrics"] = scores
        self.model_metadata["test_size"] = len(X_test)
        self.model_metadata["train_size"] = len(X_full)
        self.model_metadata["id_col"] = self.id_col
        self.model_metadata["numerical_columns"] = self._numeric_columns()
        self.model_metadata["categorical_columns"] = self._categorical_columns()
        self.model_metadata["python_version"] = platform.python_version()
        self.model_metadata["numpy_version"] = np.__version__
        import sklearn

        self.model_metadata["sklearn_version"] = sklearn.__version__
        self.model_metadata["model_version"] = "1.0"
        self.model_metadata["feature_processing"] = {
            "numeric_imputation": "mean/median + minmax scaling",
            "categorical_imputation": "mode + one-hot encoding",
        }

        self.model_metadata["predictions"] = predictions.tolist()
        self.model_metadata["predictions_index"] = self.train_sets["test"][0].index.tolist()

        self._print(f"Mejor modelo entrenado: {best_name}")
        self._print(f"Métricas de test final: {scores}")

    def export_model_artifact(self, output_path: str) -> None:
        """Guarda el modelo final y metadata en disco.

        Args:
            output_path: Ruta de salida para el archivo (extensión .json o .pkl).

        Returns:
            None.

        Raises:
            RuntimeError: Si aún no se entrenó el mejor modelo.
        """
        if self.best_model_pipeline is None:
            raise RuntimeError("No hay modelo entrenado para exportar")

        path = Path(output_path)
        if path.suffix.lower() == ".json":
            artifact = {
                "best_model_name": self.best_model_name,
                "metadata": self.model_metadata,
                "preprocessor": str(self.preprocessor),
                "model": str(self.best_model_pipeline.named_steps["model"]),
            }
            path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
        else:
            import pickle

            with path.open("wb") as handle:
                pickle.dump(
                    {
                        "best_model_name": self.best_model_name,
                        "metadata": self.model_metadata,
                        "pipeline": self.best_model_pipeline,
                    },
                    handle,
                )
        self._print(f"Artifact guardado en {path.resolve()}")

    def final_summary(self) -> dict[str, Any]:
        """Retorna metadata del modelo final y resultados.

        Args:
            None.

        Returns:
            Diccionario con la información del modelo entrenado.
        """
        return {
            "best_model_name": self.best_model_name,
            "metadata": self.model_metadata,
            "comparison_report": self.comparison_report.to_dict(orient="records")
            if self.comparison_report is not None
            else [],
        }


if __name__ == "__main__":
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=600,
        n_features=8,
        n_informative=6,
        noise=0.7,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
    df["category"] = np.random.choice(["A", "B", "C"], size=len(df))
    df["target"] = y
    df.loc[df.sample(frac=0.15, random_state=42).index, "num_0"] = np.nan
    df.loc[df.sample(frac=0.1, random_state=24).index, "category"] = np.nan
    df["id"] = np.arange(len(df))

    dr = DataRegresion(df, target_col="target", id_col="id")
    dr.split_data(train_size=0.6, val_size=0.2, test_size=0.2)
    dr.compare_models(metrics=["rmse", "r2"])
    dr.train_best_model(metric="rmse", cv=3)
    dr.export_model_artifact("regression_artifact.json")
    print(dr.final_summary())
