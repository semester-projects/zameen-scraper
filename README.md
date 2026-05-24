# 🏡 Zameen.com Property Price Prediction System

![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Scrapy Crawler](https://img.shields.io/badge/Scrapy-Crawler-green.svg?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML_Pipeline-orange.svg?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-Predictors-red.svg?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-Predictors-yellow.svg?style=for-the-badge)
![Rich Terminal](https://img.shields.io/badge/Rich-Terminal_UI-blueviolet.svg?style=for-the-badge)

An end-to-end Machine Learning property intelligence system that scrapes, preprocesses, and fits multiple model estimators to predict house prices across major Pakistani cities. Styled with a premium Rich Terminal UI featuring progress bars, overfitting model isolation, and target-unscaled price prediction engines.

---

## ⚡ Setup and Installation

### 1. Prerequisite Packages
Ensure you have Python 3.8+ installed on your system. 

### 2. Environment Activation (Recommended)
Create and activate a virtual environment to isolate project dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies
Run the following pip command to install all scraper libraries, machine learning frameworks, terminal visualizer pipelines, and plotting toolkits:

```bash
pip install scrapy beautifulsoup4 pandas numpy scikit-learn xgboost catboost rich matplotlib seaborn
```

---

## 🚀 Command Line Interface (CLI) Manual

The system is fully controlled via the `main.py` CLI orchestrator. It manages scraping, cleaning, categorical target encoding, model evaluation, and running real-time price predictions.

### 📋 Full Argument Reference Table

| Argument | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `--scrape` | Flag | `False` | Executes the scraper pipeline for the selected city. |
| `--preprocess` | Flag | `False` | Standardizes raw prices/areas, filters outliers, target encodes locations, and creates categorical dummy features. |
| `--train` | Flag | `False` | Splits data, trains 6 standard/boosting estimators, isolates overfitting, and serializes estimators. |
| `--all` | Flag | `False` | Executes scraper, preprocessing, and training phases sequentially. |
| `--predict <path_to_csv>` | String | `None` | Path to a CSV file to execute the real-time Price Prediction inference engine. |
| `--model-name <name>` | Choice | `random_forest` | Saved model to load for running predictions. Choices: `linear_regression`, `decision_tree`, `random_forest`, `gradient_boosting`, `xgboost`, `catboost`. |
| `--limit <int>` | Integer | `10` | Maximum number of property listings to scrape. Set to `-1` to run an unlimited crawl. |
| `--city <str>` | Choice | `Islamabad` | Target city to scrape. Choices: `Islamabad`, `Karachi`, `Lahore`, `Rawalpindi`. |
| `--raw-data-path <str>` | String | `assets/raw_data.csv` | Output file path for raw property CSV exports. |
| `--clean-data-path <str>` | String | `assets/cleaned_data.csv`| Output file path for preprocessed, numeric, and rounded datasets. |
| `--model-dir <str>` | String | `assets/save_model` | Target folder directory for saving and loading serialized `.pkl` estimators. |

---

## 💡 Example Usage Scenarios

### 1. Execute the Complete Pipeline End-to-End
Run the crawler, parse fields, target encode high-cardinality locations, train all estimators, filter overfitted models, and save them in one command:

```bash
# Run for Islamabad with default limit of 10
python main.py --all

# Run for Lahore with a target limit of 1000 properties
python main.py --all --city Lahore --limit 1000
```

Alternatively, run the automated pipeline runner script:
```bash
chmod +x run.sh
./run.sh
```

### 2. Scraping Phase Only
Execute a customized crawl targeting particular property counts in a specific city:
```bash
python main.py --scrape --city Karachi --limit 150
```

### 3. Preprocessing Phase Only
Run numeric standardizations and feature engineer location mappings:
```bash
python main.py --preprocess --raw-data-path assets/raw_data.csv --clean-data-path assets/cleaned_data.csv
```

### 4. Training and Evaluation Phase Only
Fit all 6 estimators (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, CatBoost), isolate overfitting, and display clean metrics:
```bash
python main.py --train --clean-data-path assets/cleaned_data.csv --model-dir assets/save_model
```

### 5. Run Price Predictions (Inference Engine)
Predict house prices for a batch of properties in a CSV file using your best saved model. High-cardinality locations are mapped back to their training target encoded means, and final predicted prices are automatically unscaled back to standard PKR Rupees:
```bash
# Predict using CatBoost
python main.py --predict assets/raw_data.csv --model-name catboost

# Predict using Random Forest
python main.py --predict assets/raw_data.csv --model-name random_forest
```

---

## 📊 Model Output Formatting

- **Robust Target Scaling**: All estimators are wrapped in Scikit-Learn `TransformedTargetRegressor` pipelines with standard scaling, ensuring predictions are transparently unscaled back into real currency values.
- **Normalized Table Metrics**: Creepy large floats are formatted elegantly: MAE and RMSE are displayed in **Millions of PKR (M)** (e.g. `14.02 M`), while MSE is formatted in standard **Scientific Notation** (e.g. `2.11e+14`).
- **Overfitting Model Isolation**: Any estimator yielding an $R^2 \ge 0.9999$ (e.g., Linear Regression on tiny datasets) is automatically stripped from the leaderboard and printed in a dedicated yellow warning panel.
