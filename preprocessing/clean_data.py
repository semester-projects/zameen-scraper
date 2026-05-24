import os
import pandas as pd
from rich.console import Console
from preprocessing.parsers import parse_price, parse_area
from preprocessing.cleaner import clean_missing_and_duplicates
from preprocessing.encoder import encode_categorical, encode_location


def preprocess_pipeline(
    raw_data_file_path: str = "assets/raw_data.csv",
    cleaned_data_file_path: str = "assets/cleaned_data.csv"
) -> bool:
    """
    Orchestrates the entire property data preprocessing pipeline:
    Reads raw crawled CSV data, cleans, standardizes price/area features,
    imputes missing data, encodes categorical/location variables,
    and serializes the output.
    """
    console = Console()

    if not os.path.exists(raw_data_file_path):
        console.print(
            f"[bold red]Error: Raw data file {raw_data_file_path} "
            f"not found![/bold red]"
        )
        return False

    property_listings_dataframe = pd.read_csv(raw_data_file_path)
    console.print(
        f"[green]Loaded raw dataset of shape "
        f"{property_listings_dataframe.shape}[/green]"
    )

    property_listings_dataframe['Price'] = (
        property_listings_dataframe['Price'].apply(parse_price)
    )
    property_listings_dataframe['Area'] = (
        property_listings_dataframe['Area'].apply(parse_area)
    )
    console.print(
        "[green]Standardized Price and Area to numeric scales[/green]"
    )

    property_listings_dataframe = clean_missing_and_duplicates(
        property_listings_dataframe
    )
    console.print(
        f"[green]Removed duplicates and imputed missing features. "
        f"Shape: {property_listings_dataframe.shape}[/green]"
    )

    property_listings_dataframe = encode_location(
        property_listings_dataframe
    )
    console.print(
        "[green]Target encoded high-cardinality Location feature[/green]"
    )

    property_listings_dataframe = encode_categorical(
        property_listings_dataframe, ['Property type', 'City']
    )
    console.print(
        "[green]One-Hot encoded Property Type and City "
        "categorical predictors[/green]"
    )

    property_listings_dataframe = property_listings_dataframe.round(2)
    console.print(
        "[green]Formatted dataset float variables to 2 "
        "decimal points[/green]"
    )

    os.makedirs(
        os.path.dirname(os.path.abspath(cleaned_data_file_path)),
        exist_ok=True
    )
    property_listings_dataframe.to_csv(
        cleaned_data_file_path, index=False
    )
    console.print(
        f"[bold green]Preprocessed output written to "
        f"{cleaned_data_file_path}[/bold green]"
    )

    return True
