import pandas as pd
from typing import List


def encode_categorical(
    property_dataframe: pd.DataFrame,
    categorical_columns: List[str]
) -> pd.DataFrame:
    """
    Applies one-hot dummy encoding to specify categorical features, converting
    text labels into aligned binary indicators.
    """
    present_categorical_columns = [
        col for col in categorical_columns
        if col in property_dataframe.columns
    ]
    if not present_categorical_columns:
        return property_dataframe.copy()
    return pd.get_dummies(
        property_dataframe,
        columns=present_categorical_columns,
        drop_first=True,
        dtype=float
    )


def encode_location(
    property_dataframe: pd.DataFrame,
    location_column_name: str = 'Location',
    target_price_column_name: str = 'Price'
) -> pd.DataFrame:
    """
    Applies target mean-value encoding on high-cardinality location fields,
    using historical prices or falling back to category code indices.
    """
    encoded_property_dataframe = property_dataframe.copy()
    if location_column_name in encoded_property_dataframe.columns:
        if target_price_column_name in encoded_property_dataframe.columns:
            location_target_mean_prices = (
                encoded_property_dataframe
                .groupby(location_column_name)[target_price_column_name]
                .mean()
            )
            global_mean_property_price = (
                encoded_property_dataframe[target_price_column_name]
                .mean()
            )
            encoded_property_dataframe[location_column_name] = (
                encoded_property_dataframe[location_column_name]
                .map(location_target_mean_prices)
                .fillna(global_mean_property_price)
            )
        else:
            encoded_property_dataframe[location_column_name] = (
                encoded_property_dataframe[location_column_name]
                .astype('category')
                .cat.codes
            )
    return encoded_property_dataframe
