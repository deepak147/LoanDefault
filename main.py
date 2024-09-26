import mlflow
import mlflow.sklearn
import pandas as pd

from scripts.feature_engineering import feature_engineering
from scripts.feature_selection import feature_selection
from scripts.model_building import train_and_evaluate_models, select_best_model
from scripts.preprocess import create_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def main():

    mlflow.set_experiment("Bank Marketing Campaign")

    with mlflow.start_run(run_name="Full Pipeline"):
        file_path = input("Enter file name: ")
        df = pd.read_csv(file_path)
        fe_df = feature_engineering(df)
        fs_df, correlation_heatmap = feature_selection(fe_df)
        mlflow.log_artifact(correlation_heatmap)
        preprocessor = create_preprocessor(fs_df)

        X = fs_df.drop("y", axis=1)
        y = fs_df["y"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        results, models = train_and_evaluate_models(
            X_train, X_test, y_train, y_test, preprocessor
        )

        best_model_name = select_best_model(results)
        best_model = models[best_model_name]

        final_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", best_model)]
        )
        final_pipeline.fit(X, y)

        mlflow.log_artifact("term_deposit_classification.ipynb")
        mlflow.log_params({"best_model": best_model_name})
        mlflow.log_params(results[best_model_name]["Best Parameters"])
        results_metrics = {k: v for k, v in list(results[best_model_name].items())[1:]}
        mlflow.log_metrics(results_metrics)
        mlflow.sklearn.log_model(final_pipeline, "final_model")


if __name__ == "__main__":

    main()
