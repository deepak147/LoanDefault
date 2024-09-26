from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_preprocessor(df):
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.drop(
        "y"
    )
    categorical_features = df.select_dtypes(include=["object"]).columns

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_transformer, numerical_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor