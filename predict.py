import pandas as pd
import numpy as np
import mlflow
from scripts.feature_engineering import feature_engineering

def load_model():
    model_uri = "runs:/70b578fbfaf94f35a293bfbe0cb6a228/final_model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    return loaded_model

def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    df = feature_engineering(df)
    return df

def predict(input_data):
    model = load_model()
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    
    return int(prediction[0])

if __name__ == "__main__":
    # Example input data (replace with actual input method)
    sample_input = {
        "age": 30,
        "job": "management",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 1000,
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "poutcome": "unknown",
        "emp_var_rate": 1.1,
        "cons_price_idx": 93.994,
        "cons_conf_idx": -36.4,
        "euribor3m": 4.857,
        "nr_employed": 5191.0
    }

    result = predict(sample_input)
    if result == 0:
        print("Client did not subscribe to a term deposit")
    else:
        print("Client did subscribe to a term deposit")