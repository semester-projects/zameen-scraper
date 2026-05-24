import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rich.console import Console
from rich.table import Table
from typing import Dict, List, Any


def evaluate_and_compare(
    fit_estimators_dictionary: Dict[str, Any],
    test_features: Any,
    test_target_prices: Any
) -> List[Dict[str, Any]]:
    """
    Evaluates fit model estimators on test partitions and prints formatted
    Rich metrics tables separating overfitted and standard predictors.
    """
    console = Console()
    model_evaluation_metrics = []

    for estimator_name, trained_estimator in fit_estimators_dictionary.items():
        predicted_prices = trained_estimator.predict(test_features)
        mean_absolute_error_val = mean_absolute_error(
            test_target_prices, predicted_prices
        )
        mean_squared_error_val = mean_squared_error(
            test_target_prices, predicted_prices
        )
        root_mean_squared_error_val = np.sqrt(mean_squared_error_val)
        coefficient_of_determination_r2 = r2_score(
            test_target_prices, predicted_prices
        )

        model_evaluation_metrics.append({
            'name': estimator_name,
            'mae': mean_absolute_error_val,
            'mse': mean_squared_error_val,
            'rmse': root_mean_squared_error_val,
            'r2': coefficient_of_determination_r2
        })

    overfitted_estimators_list = [
        metrics_record for metrics_record in model_evaluation_metrics
        if metrics_record['r2'] >= 0.9999
    ]
    standard_estimators_list = [
        metrics_record for metrics_record in model_evaluation_metrics
        if metrics_record['r2'] < 0.9999
    ]

    standard_estimators_list.sort(key=lambda x: x['r2'], reverse=True)
    overfitted_estimators_list.sort(key=lambda x: x['r2'], reverse=True)

    if standard_estimators_list:
        best_model_name = standard_estimators_list[0]['name']
        best_r2 = standard_estimators_list[0]['r2']
    else:
        best_model_name = (
            overfitted_estimators_list[0]['name']
            if overfitted_estimators_list else "None"
        )
        best_r2 = (
            overfitted_estimators_list[0]['r2']
            if overfitted_estimators_list else 0.0
        )

    if standard_estimators_list:
        table = Table(
            title="House Price Prediction Model Comparison",
            title_style="bold magenta"
        )
        table.add_column("Model Name", style="cyan")
        table.add_column("MAE (M PKR)", justify="right")
        table.add_column("MSE", justify="right")
        table.add_column("RMSE (M PKR)", justify="right")
        table.add_column("R² Score", justify="right")

        for metrics_record in standard_estimators_list:
            is_best = (metrics_record['name'] == best_model_name)
            style = "bold green" if is_best else ""
            name_str = (
                f"{metrics_record['name']} [Best]"
                if is_best else metrics_record['name']
            )

            table.add_row(
                name_str,
                f"{metrics_record['mae'] / 1e6:.2f} M",
                f"{metrics_record['mse']:.2e}",
                f"{metrics_record['rmse'] / 1e6:.2f} M",
                f"{metrics_record['r2']:.4f}",
                style=style
            )
        console.print(table)

    if overfitted_estimators_list:
        overfit_table = Table(
            title="⚠️ Excluded Overfitted Models (Perfect Fit / R² ≈ 1.0)",
            title_style="bold yellow",
            border_style="yellow"
        )
        overfit_table.add_column("Model Name", style="orange3")
        overfit_table.add_column("MAE (M PKR)", justify="right")
        overfit_table.add_column("MSE", justify="right")
        overfit_table.add_column("RMSE (M PKR)", justify="right")
        overfit_table.add_column("R² Score", justify="right")

        for metrics_record in overfitted_estimators_list:
            overfit_table.add_row(
                metrics_record['name'],
                f"{metrics_record['mae'] / 1e6:.2f} M",
                f"{metrics_record['mse']:.2e}",
                f"{metrics_record['rmse'] / 1e6:.2f} M",
                f"{metrics_record['r2']:.4f}",
                style="bold yellow"
            )
        console.print("\n")
        console.print(overfit_table)

    if best_model_name != "None":
        console.print(
            f"\n[bold green]🏆 The Best Performing Model is "
            f"{best_model_name} with an R² Score of {best_r2:.4f}!"
            f"[/bold green]\n"
        )

    return model_evaluation_metrics
