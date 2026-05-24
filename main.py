import argparse
import sys
import os
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from scraper.run_spider import execute_spider
from preprocessing.clean_data import preprocess_pipeline
from models.train_models import run_models_pipeline

warnings.filterwarnings("ignore")


def run_prediction(
    prediction_data_file_path: str,
    model_name: str,
    raw_data_path: str,
    clean_data_path: str,
    model_dir: str
):
    """
    Loads a saved serialized model, reads new raw property listings data,
    aligns feature categories, and generates a formatted comparison report.
    """
    console = Console()
    import pickle
    import pandas as pd
    from preprocessing.parsers import parse_price, parse_area

    if not os.path.exists(prediction_data_file_path):
        console.print(
            f"[bold red]✘ Error: Prediction dataset not found at "
            f"{prediction_data_file_path}[/bold red]"
        )
        sys.exit(1)

    sanitized_model_name = model_name.lower().replace(" ", "_")
    serialized_model_file_path = os.path.join(
        model_dir, f"{sanitized_model_name}.pkl"
    )
    if not os.path.exists(serialized_model_file_path):
        console.print(
            f"[bold red]✘ Error: Saved model {model_name} not found at "
            f"{serialized_model_file_path}[/bold red]"
        )
        console.print(
            "[yellow]Please run the training phase first using "
            "--train or --all.[/yellow]"
        )
        sys.exit(1)

    # Load model
    with open(serialized_model_file_path, 'rb') as f:
        model = pickle.load(f)

    # Load predict data
    raw_prediction_dataframe = pd.read_csv(prediction_data_file_path)
    processed_prediction_dataframe = raw_prediction_dataframe.copy()

    # Store original prices for comparison table if present
    original_prices_series = None
    if 'Price' in processed_prediction_dataframe.columns:
        original_prices_series = processed_prediction_dataframe['Price'].copy()
        processed_prediction_dataframe['Price'] = (
            processed_prediction_dataframe['Price'].apply(parse_price)
        )

    # 1. Clean and standardize Area
    if 'Area' in processed_prediction_dataframe.columns:
        processed_prediction_dataframe['Area'] = (
            processed_prediction_dataframe['Area'].apply(parse_area)
        )

    # 2. Impute missing values using training dataset as reference
    if os.path.exists(clean_data_path):
        reference_clean_dataframe = pd.read_csv(clean_data_path)
    else:
        reference_clean_dataframe = None

    for feature_column_name in processed_prediction_dataframe.columns:
        if feature_column_name == 'Price':
            continue
        if processed_prediction_dataframe[feature_column_name].isnull().any():
            is_numeric = (
                processed_prediction_dataframe[feature_column_name].dtype
                in [float, int]
            )
            if (
                reference_clean_dataframe is not None
                and feature_column_name in reference_clean_dataframe.columns
            ):
                median_val = (
                    reference_clean_dataframe[feature_column_name].median()
                )
                mode_val = (
                    reference_clean_dataframe[feature_column_name].mode()[0]
                )
                imputed_column_fill_value = (
                    median_val if is_numeric else mode_val
                )
            else:
                df_col = processed_prediction_dataframe[feature_column_name]
                median_val = df_col.median()
                mode_val = df_col.mode()[0]
                imputed_column_fill_value = (
                    median_val if is_numeric else mode_val
                )
            processed_prediction_dataframe[feature_column_name] = (
                processed_prediction_dataframe[feature_column_name]
                .fillna(imputed_column_fill_value)
            )

    # 3. Target encode Location using training data mapping
    if 'Location' in processed_prediction_dataframe.columns:
        if os.path.exists(raw_data_path):
            raw_training_dataframe = pd.read_csv(raw_data_path)
            raw_training_dataframe['Price_num'] = (
                raw_training_dataframe['Price'].apply(parse_price)
            )
            location_target_mean_prices = (
                raw_training_dataframe.groupby('Location')['Price_num'].mean()
            )
            global_mean_property_price = (
                raw_training_dataframe['Price_num'].mean()
            )
        else:
            location_target_mean_prices = pd.Series()
            global_mean_property_price = 0.0

        processed_prediction_dataframe['Location'] = (
            processed_prediction_dataframe['Location']
            .map(location_target_mean_prices)
            .fillna(global_mean_property_price)
        )

    # 4. One-Hot encoding of Property type and City
    if reference_clean_dataframe is not None:
        training_feature_names = [
            col for col in reference_clean_dataframe.columns
            if col != 'Price'
        ]
    else:
        training_feature_names = []

    columns_to_one_hot_encode = [
        col for col in ['Property type', 'City']
        if col in processed_prediction_dataframe.columns
    ]
    if columns_to_one_hot_encode:
        processed_prediction_dataframe = pd.get_dummies(
            processed_prediction_dataframe,
            columns=columns_to_one_hot_encode,
            dtype=float
        )
    aligned_prediction_features = (
        processed_prediction_dataframe
        .drop(columns=['Price'], errors='ignore')
    )

    # 5. Align features: Add missing columns, drop extra, keep order
    if training_feature_names:
        for feature_column_name in training_feature_names:
            if feature_column_name not in aligned_prediction_features.columns:
                aligned_prediction_features[feature_column_name] = 0.0
        aligned_prediction_features = (
            aligned_prediction_features[training_feature_names]
        )

    # Run prediction
    predicted_property_prices = model.predict(aligned_prediction_features)

    # Create Table to present predictions
    table = Table(
        title=f"🏡 Property Price Predictions using {model_name}",
        title_style="bold cyan"
    )
    table.add_column("Index", justify="center")
    table.add_column("Location (Raw)", style="cyan")
    table.add_column("Area", justify="right")
    table.add_column("Rooms (Bed/Bath)", justify="center")
    if original_prices_series is not None:
        table.add_column("Scraped Price", justify="right")
    table.add_column(
        "Predicted Price (PKR)", style="bold green", justify="right"
    )
    table.add_column("Deviation", justify="right")

    for i in range(len(raw_prediction_dataframe)):
        loc = raw_prediction_dataframe.iloc[i].get('Location', 'Unknown')
        area = raw_prediction_dataframe.iloc[i].get('Area', 'N/A')
        beds = raw_prediction_dataframe.iloc[i].get('Bedrooms', '?')
        baths = raw_prediction_dataframe.iloc[i].get('Bathrooms', '?')

        predicted_price_pkr = predicted_property_prices[i]
        formatted_price_string = f"{predicted_price_pkr:,.2f}"

        prediction_table_row_data = [
            str(i + 1),
            str(loc),
            str(area),
            f"{beds} bed / {baths} bath"
        ]

        if original_prices_series is not None:
            raw_price = original_prices_series.iloc[i]
            prediction_table_row_data.append(str(raw_price))

            try:
                num_actual = parse_price(raw_price)
                price_deviation_percentage = (
                    ((predicted_price_pkr - num_actual) / num_actual) * 100
                )
                price_deviation_string = f"{price_deviation_percentage:+.1f}%"

                if abs(price_deviation_percentage) < 15:
                    dev_style = "green"
                elif abs(price_deviation_percentage) < 30:
                    dev_style = "yellow"
                else:
                    dev_style = "red"

                formatted_deviation_text = (
                    f"[{dev_style}]{price_deviation_string}[/{dev_style}]"
                )
            except Exception:
                formatted_deviation_text = "-"
        else:
            formatted_deviation_text = "-"

        prediction_table_row_data.append(formatted_price_string)
        prediction_table_row_data.append(formatted_deviation_text)

        table.add_row(*prediction_table_row_data)

    console.print(table)


def main():
    console = Console()

    cli_argument_parser = argparse.ArgumentParser(
        description="Zameen.com Property Price Prediction System CLI"
    )
    cli_argument_parser.add_argument('--scrape', action='store_true')
    cli_argument_parser.add_argument('--preprocess', action='store_true')
    cli_argument_parser.add_argument('--train', action='store_true')
    cli_argument_parser.add_argument('--all', action='store_true')
    cli_argument_parser.add_argument(
        '--predict', type=str, default=None,
        help="Path to property CSV file for running price predictions"
    )
    cli_argument_parser.add_argument(
        '--model-name', type=str, default='random_forest',
        choices=[
            'linear_regression', 'decision_tree', 'random_forest',
            'gradient_boosting', 'xgboost', 'catboost'
        ],
        help="Model to use for predictions"
    )
    cli_argument_parser.add_argument('--limit', type=int, default=10)
    cli_argument_parser.add_argument(
        '--city', type=str, default='Islamabad',
        choices=['Islamabad', 'Karachi', 'Lahore', 'Rawalpindi']
    )
    cli_argument_parser.add_argument(
        '--raw-data-path', type=str, default='assets/raw_data.csv'
    )
    cli_argument_parser.add_argument(
        '--clean-data-path', type=str, default='assets/cleaned_data.csv'
    )
    cli_argument_parser.add_argument(
        '--model-dir', type=str, default='assets/save_model'
    )

    parsed_cli_arguments = cli_argument_parser.parse_args()

    if not (
        parsed_cli_arguments.scrape
        or parsed_cli_arguments.preprocess
        or parsed_cli_arguments.train
        or parsed_cli_arguments.all
        or parsed_cli_arguments.predict
    ):
        cli_argument_parser.print_help()
        sys.exit(1)

    console.print(Panel(
        Text(
            f"ZAMEEN.COM HOUSE PRICE PREDICTION SYSTEM\n"
            f"{parsed_cli_arguments.city.upper()} PIPELINE",
            style="bold white",
            justify="center"
        ),
        style="bold magenta",
        expand=False
    ))

    if parsed_cli_arguments.predict:
        console.print(
            f"\n[bold cyan]⚡ Running Inference Engine "
            f"(Model: {parsed_cli_arguments.model_name})[/bold cyan]"
        )
        try:
            run_prediction(
                prediction_data_file_path=parsed_cli_arguments.predict,
                model_name=parsed_cli_arguments.model_name,
                raw_data_path=parsed_cli_arguments.raw_data_path,
                clean_data_path=parsed_cli_arguments.clean_data_path,
                model_dir=parsed_cli_arguments.model_dir
            )
        except Exception as e:
            console.print(
                f"[bold red]✘ Inference Engine failed: {str(e)}[/bold red]\n"
            )
            sys.exit(1)
        sys.exit(0)

    if parsed_cli_arguments.scrape or parsed_cli_arguments.all:
        console.print(
            f"\n[bold cyan]⚡ Phase 1: Executing Scraper Pipeline "
            f"for {parsed_cli_arguments.city}[/bold cyan]"
        )
        try:
            execute_spider(
                limit=parsed_cli_arguments.limit,
                raw_data_path=parsed_cli_arguments.raw_data_path,
                city=parsed_cli_arguments.city
            )
            console.print(
                "[bold green]✔ Scraper Phase completed successfully."
                "[/bold green]\n"
            )
        except Exception as e:
            console.print(
                f"[bold red]✘ Scraper Phase failed: {str(e)}[/bold red]\n"
            )
            sys.exit(1)

    if parsed_cli_arguments.preprocess or parsed_cli_arguments.all:
        console.print(
            "\n[bold yellow]⚡ Phase 2: Executing Preprocessing "
            "Pipeline[/bold yellow]"
        )
        try:
            is_preprocessing_successful = preprocess_pipeline(
                raw_data_file_path=parsed_cli_arguments.raw_data_path,
                cleaned_data_file_path=parsed_cli_arguments.clean_data_path
            )
            if is_preprocessing_successful:
                console.print(
                    "[bold green]✔ Preprocessing Phase completed "
                    "successfully.[/bold green]\n"
                )
            else:
                console.print(
                    "[bold red]✘ Preprocessing Phase failed.[/bold red]\n"
                )
                sys.exit(1)
        except Exception as e:
            console.print(
                f"[bold red]✘ Preprocessing Phase failed with error: "
                f"{str(e)}[/bold red]\n"
            )
            sys.exit(1)

    if parsed_cli_arguments.train or parsed_cli_arguments.all:
        console.print(
            "\n[bold magenta]⚡ Phase 3: Executing Model Training "
            "and Evaluation[/bold magenta]"
        )
        try:
            run_models_pipeline(
                cleaned_dataset_file_path=parsed_cli_arguments.clean_data_path,
                model_serialization_directory=parsed_cli_arguments.model_dir
            )
            console.print(
                "[bold green]✔ Model Training Phase completed "
                "successfully.[/bold green]\n"
            )
        except Exception as e:
            console.print(
                f"[bold red]✘ Model Training Phase failed with error: "
                f"{str(e)}[/bold red]\n"
            )
            sys.exit(1)

    console.print(Panel(
        Text("✨ PIPELINE COMPLETE ✨", style="bold green", justify="center"),
        style="bold green",
        expand=False
    ))


if __name__ == '__main__':
    main()
