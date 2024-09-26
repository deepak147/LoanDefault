import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy.stats import chi2_contingency


def feature_selection(df):

    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.drop(
        "y"
    )
    correlation_matrix = df[numerical_features].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap of Numerical Features")
    correlation_heatmap = "Correlation heatmap.png"
    plt.savefig(correlation_heatmap)
    plt.close()

    correlation_threshold = 0.8
    high_correlation_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                colname = correlation_matrix.columns[i]
                high_correlation_features.add(colname)

    categorical_features = df.select_dtypes(include=["object"]).columns
    high_pvalue_features = set()
    pvalue_threshold = 0.05
    for feature in categorical_features:
        if feature != "y":
            chi2, p_value, dof, expected = chi2_contingency(
                pd.crosstab(df[feature], df["y"])
            )
            if p_value > pvalue_threshold:
                high_pvalue_features.add(feature)

    features_to_drop = high_correlation_features.union(high_pvalue_features)
    df_cleaned = df.drop(columns=features_to_drop)

    return df_cleaned, correlation_heatmap