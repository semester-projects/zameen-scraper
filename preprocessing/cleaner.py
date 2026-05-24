import pandas as pd
import numpy as np


def clean_missing_and_duplicates(
    raw_listings_dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Removes duplicated properties and handles missing feature imputations using
    calculated median values or realistic fallbacks.
    """
    cleaned_listings_dataframe = raw_listings_dataframe.copy()
    cleaned_listings_dataframe = cleaned_listings_dataframe.drop_duplicates()

    target_numeric_columns = [
        'Bedrooms', 'Bathrooms', 'Built in year', 'Parking space',
        'Servant Quarters', 'Store rooms', 'Kitchens', 'Drawing Rooms'
    ]

    for column_name in target_numeric_columns:
        if column_name in cleaned_listings_dataframe.columns:
            cleaned_listings_dataframe[column_name] = pd.to_numeric(
                cleaned_listings_dataframe[column_name], errors='coerce'
            )

    if 'Built in year' in cleaned_listings_dataframe.columns:
        invalid_years_mask = (
            (cleaned_listings_dataframe['Built in year'] < 1900) |
            (cleaned_listings_dataframe['Built in year'] > 2026)
        )
        cleaned_listings_dataframe.loc[
            invalid_years_mask, 'Built in year'
        ] = np.nan

    default_impute_medians = {
        'Bedrooms': 3.0,
        'Bathrooms': 3.0,
        'Built in year': 2018.0,
        'Parking space': 1.0,
        'Servant Quarters': 0.0,
        'Store rooms': 0.0,
        'Kitchens': 1.0,
        'Drawing Rooms': 1.0
    }

    for column_name, default_impute_value in default_impute_medians.items():
        if column_name in cleaned_listings_dataframe.columns:
            calculated_median_value = (
                cleaned_listings_dataframe[column_name].median()
            )
            final_fill_value = (
                calculated_median_value
                if not pd.isna(calculated_median_value)
                else default_impute_value
            )
            cleaned_listings_dataframe[column_name] = (
                cleaned_listings_dataframe[column_name]
                .fillna(final_fill_value)
            )

    return cleaned_listings_dataframe
