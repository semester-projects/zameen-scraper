import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from rich.progress import Progress, BarColumn, TextColumn
from typing import Dict, Tuple, Any


def train_all_models(
    clean_data_path: str = "assets/cleaned_data.csv",
    model_dir: str = "assets/save_model",
    test_size: float = 0.2,
    random_state: int = 42,
    custom_params: Dict[str, Any] = None
) -> Tuple[
    Dict[str, TransformedTargetRegressor],
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series
]:
    """
    Fits multiple regression pipelines wrapped in Scikit-Learn standard scaled
    target regressors. Serializes and outputs metrics for comparative
    leaderboards.
    """
    if not os.path.exists(clean_data_path):
        raise FileNotFoundError(
            f"Cleaned dataset not found at {clean_data_path}"
        )

    cleaned_listings_df = pd.read_csv(clean_data_path)

    essential_target_columns = ['Price', 'Area', 'Bedrooms', 'Bathrooms']
    for column_name in essential_target_columns:
        if column_name not in cleaned_listings_df.columns:
            raise ValueError(
                f"Dataset is missing required feature field: {column_name}"
            )

    target_prices_vector = cleaned_listings_df['Price']
    feature_matrix = cleaned_listings_df.drop(columns=['Price'])

    (
        train_features,
        test_features,
        train_target_prices,
        test_target_prices
    ) = train_test_split(
        feature_matrix,
        target_prices_vector,
        test_size=test_size,
        random_state=random_state
    )

    model_hyperparams = {
        'linear_regression': {},
        'decision_tree': {'max_depth': 10, 'random_state': random_state},
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 2,
            'random_state': random_state
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': random_state
        },
        'xgboost': {
            'n_estimators': 50,
            'learning_rate': 0.05,
            'max_depth': 3,
            'random_state': random_state
        },
        'catboost': {
            'iterations': 100,
            'learning_rate': 0.1,
            'verbose': 0,
            'random_seed': random_state
        }
    }

    if custom_params:
        model_hyperparams.update(custom_params)

    unfitted_pipelines = {
        'Linear Regression': TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                (
                    'model',
                    LinearRegression(
                        **model_hyperparams['linear_regression']
                    )
                )
            ]),
            transformer=StandardScaler()
        ),
        'Decision Tree': TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                (
                    'model',
                    DecisionTreeRegressor(
                        **model_hyperparams['decision_tree']
                    )
                )
            ]),
            transformer=StandardScaler()
        ),
        'Random Forest': TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                (
                    'model',
                    RandomForestRegressor(
                        **model_hyperparams['random_forest']
                    )
                )
            ]),
            transformer=StandardScaler()
        ),
        'Gradient Boosting': TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                (
                    'model',
                    GradientBoostingRegressor(
                        **model_hyperparams['gradient_boosting']
                    )
                )
            ]),
            transformer=StandardScaler()
        ),
        'XGBoost': TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                ('model', XGBRegressor(**model_hyperparams['xgboost']))
            ]),
            transformer=StandardScaler()
        ),
        'CatBoost': TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                ('model', CatBoostRegressor(**model_hyperparams['catboost']))
            ]),
            transformer=StandardScaler()
        )
    }

    fit_estimators_dictionary = {}
    os.makedirs(model_dir, exist_ok=True)

    with Progress(
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
    ) as progress:
        task = progress.add_task(
            "Training models...",
            total=len(unfitted_pipelines)
        )

        for name, transformer_reg in unfitted_pipelines.items():
            progress.update(task, description=f"Training {name}...")
            transformer_reg.fit(train_features, train_target_prices)
            fit_estimators_dictionary[name] = transformer_reg

            sanitized_name = name.lower().replace(" ", "_")
            serialized_path = os.path.join(model_dir, f"{sanitized_name}.pkl")
            with open(serialized_path, 'wb') as f:
                pickle.dump(transformer_reg, f)

            progress.advance(task)

    return (
        fit_estimators_dictionary,
        train_features,
        test_features,
        train_target_prices,
        test_target_prices
    )
