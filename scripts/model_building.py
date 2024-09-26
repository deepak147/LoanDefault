import matplotlib.pyplot as plt
import mlflow
import seaborn as sns

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):

    models = {
        "Logistic Regression": LogisticRegression(),
        "XGBoost": XGBClassifier(),
    }

    param_grids = {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__solver": ["liblinear", "saga"],
        },
        "XGBoost": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.1, 0.2],
            "classifier__subsample": [0.6, 0.8, 1.0],
        },
    }

    results = {}
    for name, model in models.items():
        with mlflow.start_run(run_name=f"Train {name}", nested=True):
            try:
                pipeline = Pipeline(
                    steps=[("preprocessor", preprocessor), ("classifier", model)]
                )

                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grids[name],
                    cv=5,
                    scoring="f1_weighted",
                    verbose=1,
                )

                grid_search.fit(X_train, y_train)
                pipeline.fit(X_train, y_train)

                best_pipeline = pipeline
                y_pred = best_pipeline.predict(X_test)
                y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

                results[name] = {
                    "Best Parameters": grid_search.best_params_,
                    "Test Accuracy": accuracy_score(y_test, y_pred),
                    "Test Precision": precision_score(
                        y_test, y_pred, average="weighted"
                    ),
                    "Test Recall": recall_score(y_test, y_pred, average="weighted"),
                    "Test F1-score": f1_score(y_test, y_pred, average="weighted"),
                    "Test ROC-AUC": roc_auc_score(y_test, y_pred_proba),
                    "CV Mean F1-score": grid_search.best_score_,
                }

                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                plt.figure(figsize=(10, 8))
                plt.plot(
                    fpr,
                    tpr,
                    label=f'ROC Curve (area = {results[name]["Test ROC-AUC"]:.4f})',
                )
                plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve - {name}")
                roc_auc_plot = f"roc_auc_{name}.png"
                plt.savefig(roc_auc_plot)
                plt.close()

                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Confusion Matrix - {name}")
                plt.ylabel("Actual")
                plt.xlabel("Predicted")
                cm_heatmap = f"confusion_matrix_{name}.png"
                plt.savefig(cm_heatmap)
                plt.close()

                mlflow.log_artifact(cm_heatmap)
                mlflow.log_artifact(roc_auc_plot)
                mlflow.log_params(results[name]["Best Parameters"])
                results_metrics = {k: v for k, v in list(results[name].items())[1:]}
                mlflow.log_metrics(results_metrics)
                mlflow.sklearn.log_model(best_pipeline, f"{name}_model")

            except Exception as e:
                print(e)

    return results, models


def select_best_model(results):
    vote_count = {model: 0 for model in results.keys()}

    for metric in ["Test Precision", "Test Recall", "Test F1-score", "Test ROC-AUC"]:
        best_model = max(results, key=lambda x: results[x][metric])
        vote_count[best_model] += 1

    best_model = max(vote_count, key=vote_count.get)

    return best_model
