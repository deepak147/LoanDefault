import pandas as pd
import numpy as np

def load_preprocess(file_path):

    df = pd.read_csv(file_path)
    df = df.drop_duplicates()

    columns = ["job", "marital", "education", "housing", "loan"]
    for column in columns:
        df[column] = df[column].apply(lambda x: np.nan if x == "unknown" else x)
        df[column].fillna(df[column].mode()[0], inplace=True)

    return df