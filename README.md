# TermWise

This project aims to predict the success of bank marketing campaigns for term deposits using machine learning techniques. It includes data preprocessing, feature engineering, model training, and evaluation, all integrated with MLflow for experiment tracking and model management.

## Project Structure

The project consists of the following main components:

1. `main.py`: The entry point of the application.
2. `feature_engineering.py`: Contains functions for feature engineering.
3. `feature_selection.py`: Implements feature selection techniques.
4. `preprocess.py`: Defines the preprocessing pipeline.
5. `model_building.py`: Handles model training, evaluation, and selection.
6. `predict.py`: Provides functionality for making predictions using the trained model.

## Detailed Component Descriptions

### 1. Main Script (`main.py`)

This script orchestrates the entire machine learning pipeline:

- Sets up the MLflow experiment
- Loads and preprocesses the data
- Performs feature engineering and selection
- Splits the data into training and test sets
- Trains and evaluates multiple models
- Selects the best model
- Logs results and artifacts to MLflow

### 2. Feature Engineering (`feature_engineering.py`)

This module applies various feature engineering techniques to the dataset:

- Handles missing values
- Creates age groups and calculates years until retirement
- Categorizes jobs and creates employment status features
- Generates loan-related features
- Maps education levels and creates a higher education indicator
- Calculates average duration per contact and marital financial responsibility

### 3. Feature Selection (`feature_selection.py`)

This script performs feature selection to improve model performance:

- Calculates correlation matrix for numerical features
- Generates a correlation heatmap (saved as an artifact)
- Identifies and removes highly correlated features
- Performs chi-squared tests for categorical features
- Removes features with high p-values

### 4. Preprocessing (`preprocess.py`)

Defines the preprocessing pipeline using scikit-learn's ColumnTransformer:

- Applies StandardScaler to numerical features
- Applies OneHotEncoder to categorical features

### 5. Model Building (`model_building.py`)

This module handles the core machine learning tasks:

- Defines models to be trained (Logistic Regression and XGBoost)
- Sets up hyperparameter grids for each model
- Performs grid search with cross-validation
- Evaluates models on the test set
- Generates and saves ROC curves and confusion matrices
- Logs model parameters, metrics, and artifacts to MLflow

### 6. Prediction Script (`predict.py`)

Provides functionality for making predictions using the trained model:

- Loads the best model from MLflow
- Preprocesses input data
- Makes predictions

## MLflow Integration

MLflow is used throughout the project for experiment tracking and model management:

- Experiment Tracking: Each run is logged as an MLflow experiment, capturing parameters, metrics, and artifacts.
- Model Logging: The best model is logged using `mlflow.sklearn.log_model`, allowing for easy retrieval and deployment.
- Artifact Logging: Important artifacts like ROC curves, confusion matrices, and correlation heatmaps are logged for each run.
- Model Registry: The final model is registered in the MLflow Model Registry for version control and deployment.

## How to Use

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```
   python main.py
   ```

3. To make predictions using the trained model:
   ```
   python predict.py
   ```

4. View the MLflow UI to analyze experiments:
   ```
   mlflow ui
   ```

## Requirements

- Python 3.10+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- mlflow
- xgboost
- sweetviz
- scipy

## Future Improvements

- Implement more advanced feature engineering techniques specific to banking domain
- Explore neural networks to build classification model
- Enhance the prediction script to handle batch predictions
- Develop a web interface and a FastAPI endpoint for easy interaction with the model
- Containerize the FastAPI application to deploy in cloud applications to maintain auto scalability
- Implement auto retrainability in the event of data drift by leveraging MLFlow logging

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

