from models.trainer import train_all_models
from models.evaluator import evaluate_and_compare
from typing import Dict, List, Any


def run_models_pipeline(
    cleaned_dataset_file_path: str = "assets/cleaned_data.csv",
    model_serialization_directory: str = "assets/save_model",
    test_size: float = 0.2,
    random_state: int = 42,
    custom_hyperparameters: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Coordinates training and comparative metrics evaluations.
    """
    (
        fit_estimators_dictionary,
        train_features,
        test_features,
        train_target_prices,
        test_target_prices
    ) = train_all_models(
        clean_data_path=cleaned_dataset_file_path,
        model_dir=model_serialization_directory,
        test_size=test_size,
        random_state=random_state,
        custom_params=custom_hyperparameters
    )
    model_evaluation_metrics = evaluate_and_compare(
        fit_estimators_dictionary=fit_estimators_dictionary,
        test_features=test_features,
        test_target_prices=test_target_prices
    )
    return model_evaluation_metrics
