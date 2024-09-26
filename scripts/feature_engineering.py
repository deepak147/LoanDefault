import pandas as pd
import numpy as np

def feature_engineering(df):

    df = df.drop_duplicates()

    columns = ["job", "marital", "education", "housing", "loan"]
    for column in columns:
        df[column] = df[column].apply(lambda x: np.nan if x == "unknown" else x)
        df[column].fillna(df[column].mode()[0], inplace=True)

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, np.inf],
        labels=["0-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        right=False,
    )
    df["years_until_retirement"] = 65 - df["age"]

    job_categories = {
        "admin.": "white-collar",
        "blue-collar": "blue-collar",
        "entrepreneur": "white-collar",
        "housemaid": "blue-collar",
        "management": "white-collar",
        "retired": "retired",
        "self-employed": "white-collar",
        "services": "blue-collar",
        "student": "student",
        "technician": "white-collar",
        "unemployed": "unemployed",
    }
    df["job_category"] = df["job"].map(job_categories)

    df["is_employed"] = df["job"].apply(
        lambda x: 0 if x in ["unemployed", "student", "retired"] else 1
    )
    df["has_both_loans"] = ((df["housing"] == "yes") & (df["loan"] == "yes")).astype(
        int
    )
    df["credit_risk_score"] = (
        df["default"].map({"yes": 2, "unknown": 1, "no": 0})
        + df["housing"].map({"yes": 1, "no": 0})
        + df["loan"].map({"yes": 1, "no": 0})
    )

    education_level = {
        "illiterate": 0,
        "basic.4y": 1,
        "basic.6y": 2,
        "basic.9y": 3,
        "high.school": 4,
        "professional.course": 5,
        "university.degree": 6,
    }
    df["education"] = df["education"].map(education_level)
    df["higher_education"] = df["education"].isin([5, 6]).astype(int)

    df["avg_duration_per_contact"] = df.groupby("contact")["duration"].transform("mean")
    df["marital_financial_responsibility"] = df.apply(
        lambda x: (
            1
            if x["marital"] in ["married", "divorced"]
            and (x["housing"] == "yes" or x["loan"] == "yes")
            else 0
        ),
        axis=1,
    )

    return df