# NYC Taxi Fare Prediction - Big Data ML Project

## Project Overview

This project implements a complete Big Data machine learning pipeline for predicting NYC taxi fares using PySpark and distributed computing. The project demonstrates end-to-end ML workflow from data acquisition to model deployment and visualization.

**Dataset:** NYC Taxi Trip Records (AWS Open Data Registry)  
**Size:** >1GB (Jan-Mar 2023)  
**Problem Type:** Regression (Fare Amount Prediction)  
**Framework:** PySpark 3.4.0 + MLlib

## Business Objective

Predict taxi fare amounts to enable:
- Dynamic pricing optimization
- Revenue forecasting
- Demand pattern analysis
- Operational efficiency improvements

## Technical Architecture

### Medallion Architecture (Bronze → Silver → Gold)

1. **Bronze Layer:** Raw data ingestion with RDD-based parallelization
2. **Silver Layer:** Cleaned and validated data with quality checks
3. **Gold Layer:** Feature-engineered data ready for ML

### Machine Learning Pipeline

- **5 Regression Algorithms:**
  1. Linear Regression (baseline)
  2. Decision Tree Regressor
  3. Random Forest Regressor
  4. Gradient Boosted Trees
  5. Generalized Linear Regression

- **Hyperparameter Tuning:** TrainValidationSplit with Grid Search
- **Evaluation Metrics:** RMSE, MAE, R²
- **Scalability Analysis:** Strong/weak scaling experiments

## Project Structure

```
nyc-taxi-ml-project/
├── README.md
├── FINAL_REPORT_DRAFT.md      # Draft of the final academic report
├── config/
│   └── spark_config.yaml      # Spark configuration
├── data/
│   ├── raw/                   # Raw Parquet files
│   ├── bronze/                # Ingested Parquet (partitioned)
│   ├── silver/                # Cleaned Parquet
│   └── gold/                  # Feature-engineered Parquet
├── notebooks/
│   ├── 01_Data_Ingestion_Bronze_Layer.ipynb
│   ├── 02_EDA_Silver_Layer.ipynb
│   ├── 03_Feature_Engineering_Gold_Layer.ipynb
│   ├── 04_Model_Training.ipynb
│   ├── 05_Hyperparameter_Tuning.ipynb
│   ├── 06_Scalability_Analysis.ipynb
│   └── 07_Tableau_Data_Preparation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data acquisition & schema utilities
│   ├── feature_engineering.py # Feature transformation logic
│   ├── model_trainer.py       # Model training & tuning logic
│   └── evaluator.py           # Evaluation metrics & reporting
├── models/                    # Saved ML models
└── results/                   # CSV exports for Tableau & Reports
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- Java JDK 11 (Required for PySpark)
- 8GB+ RAM recommended

### Installation

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Workflow Execution

Run the notebooks in the following order:

1. **01_Data_Ingestion_Bronze_Layer.ipynb**
   - Downloads raw data
   - Creates Bronze layer (Parquet)
   - Validates schema

2. **02_EDA_Silver_Layer.ipynb**
   - Performs Exploratory Data Analysis
   - Cleans data (Silver layer)
   - Visualizes distributions

3. **03_Feature_Engineering_Gold_Layer.ipynb**
   - Creates temporal, spatial, and trip features
   - Assembles feature vectors (Gold layer)

4. **04_Model_Training.ipynb**
   - Trains 5 baseline models
   - Compares performance

5. **05_Hyperparameter_Tuning.ipynb**
   - Optimizes best models (Random Forest, GBT)
   - Performs feature importance analysis

6. **06_Scalability_Analysis.ipynb**
   - Analyzes training time vs data size
   - Optimizes partition counts

7. **07_Tableau_Data_Preparation.ipynb**
   - Exports predictions and stats for dashboarding

## Tableau Dashboards

The project includes data preparation for 4 Tableau dashboards:
1. **Model Performance:** Actual vs Predicted analysis
2. **Business Insights:** Hourly and location-based patterns
3. **Feature Analysis:** Impact of distance, time, and location
4. **Scalability:** System performance metrics

## Academic Integrity

**Tools Used:**
- Cascade AI: Code structure guidance, debugging assistance

**Declaration:**
All code implementations are original and written using PySpark official documentation. The Medallion architecture and ML pipeline design reflect best practices in Big Data engineering.

---

**Last Updated:** March 1, 2026
# NYC-Taxi-Fare-Prediction
