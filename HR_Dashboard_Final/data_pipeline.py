import os
import json
import sqlite3
import pandas as pd
import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

DATASET_PATH = "dataset.csv"
DB_PATH = "employees.db"
MODEL_PATH = "model.joblib"
FEATURES_PATH = "features.json"
METRICS_PATH = "model_metrics.json"
TARGET_COL = "Attrition"


def generate_mock_dataset():
    print("Generating synthetic IBM-like HR dataset...")
    np.random.seed(42)
    n = 1470
    
    data = {
        'Age': np.random.randint(18, 60, size=n),
        'Attrition': np.random.choice(['Yes', 'No'], size=n, p=[0.16, 0.84]),
        'BusinessTravel': np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], size=n),
        'DailyRate': np.random.randint(100, 1500, size=n),
        'Department': np.random.choice(['Sales', 'Research & Development', 'Human Resources'], size=n),
        'DistanceFromHome': np.random.randint(1, 30, size=n),
        'Education': np.random.randint(1, 5, size=n),
        'EducationField': np.random.choice(['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'], size=n),
        'EmployeeCount': [1]*n,
        'EmployeeNumber': range(1, n+1),
        'EnvironmentSatisfaction': np.random.randint(1, 5, size=n),
        'Gender': np.random.choice(['Male', 'Female'], size=n),
        'HourlyRate': np.random.randint(30, 100, size=n),
        'JobInvolvement': np.random.randint(1, 5, size=n),
        'JobLevel': np.random.randint(1, 6, size=n),
        'JobRole': np.random.choice(['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'], size=n),
        'JobSatisfaction': np.random.randint(1, 5, size=n),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], size=n),
        'MonthlyIncome': np.random.randint(2000, 20000, size=n),
        'MonthlyRate': np.random.randint(2000, 26999, size=n),
        'NumCompaniesWorked': np.random.randint(0, 10, size=n),
        'Over18': ['Y']*n,
        'OverTime': np.random.choice(['Yes', 'No'], size=n),
        'PercentSalaryHike': np.random.randint(11, 26, size=n),
        'PerformanceRating': np.random.choice([3, 4], size=n),
        'RelationshipSatisfaction': np.random.randint(1, 5, size=n),
        'StandardHours': [80]*n,
        'StockOptionLevel': np.random.randint(0, 4, size=n),
        'TotalWorkingYears': np.random.randint(0, 40, size=n),
        'TrainingTimesLastYear': np.random.randint(0, 7, size=n),
        'WorkLifeBalance': np.random.randint(1, 5, size=n),
        'YearsAtCompany': np.random.randint(0, 20, size=n), # roughly
        'YearsInCurrentRole': np.random.randint(0, 15, size=n),
        'YearsSinceLastPromotion': np.random.randint(0, 10, size=n),
        'YearsWithCurrManager': np.random.randint(0, 10, size=n),
    }
    
    df = pd.DataFrame(data)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Saved synthetic dataset to {DATASET_PATH}")
    return df


def download_dataset():
    if not os.path.exists(DATASET_PATH):
        generate_mock_dataset()


def setup_database(df):
    print("Setting up SQLite Database...")
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("employees", conn, if_exists="replace", index=False)
    
    # Initialize users table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_logged_in BOOLEAN DEFAULT 0
        )
    ''')
    
    # Clear and seed admins
    conn.execute('DELETE FROM users')
    admins = [
        ("shruti", "Shruti123", "Shruti"),
        ("aryan1", "Aryan123", "Aryan"),
        ("pranav", "Pranav123", "Pranav"),
        ("shreyan", "Shreyan123", "Shreyan"),
        ("aryan2", "Aryan123", "Aryan"),
        ("angad", "Angad123", "Angad")
    ]
    for username, pwd, name in admins:
        # We will store 'name' as well for display if needed, but username is enough. 
        # Using simple passwords as requested (not hashed, for demo purposes and explicitly matching the prompt)
        conn.execute('INSERT OR IGNORE INTO users (username, password, is_logged_in) VALUES (?, ?, 0)', (username, pwd))
        
    # Initialize prediction history
    conn.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            inputs TEXT,
            prediction TEXT,
            confidence REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Data loaded to dynamic SQLite schema successfully.")


def identify_features(df):
    print("Analyzing dataset features dynamically...")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

    drop_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            drop_cols.append(col)
        elif df[col].nunique() == len(df) and df[col].dtype in ['int64', 'object']:
            drop_cols.append(col)

    for extra in ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']:
        if extra in df.columns and extra not in drop_cols:
            drop_cols.append(extra)

    df_cleaned = df.drop(columns=drop_cols)

    y = df_cleaned[TARGET_COL]
    if y.dtype == 'object':
        y = y.map({'Yes': 1, 'No': 0})
        
    X = df_cleaned.drop(columns=[TARGET_COL])

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

    return X, y, numerical_cols, categorical_cols


def generate_feature_schema(X, numerical_cols, categorical_cols):
    print("Generating feature schema for dynamic UI...")
    schema = {}
    for col in numerical_cols:
        schema[col] = {
            "type": "numerical",
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean())
        }
    for col in categorical_cols:
        val_counts = X[col].value_counts()
        schema[col] = {
            "type": "categorical",
            "options": val_counts.index.tolist()
        }
    
    with open(FEATURES_PATH, "w") as f:
        json.dump(schema, f, indent=4)
    print("Feature schema saved.")


def train_models(X_train, y_train, X_test, y_test, numerical_cols, categorical_cols):
    print("Training models pipeline...")

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    best_model = None
    best_score = 0
    best_name = ""
    best_pipeline = None

    metrics = {"models": [], "feature_importances": {}}

    for name, model in models.items():
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        metrics["models"].append({
            "name": name,
            "accuracy": float(acc),
            "f1_score": float(f1)
        })
        
        print(f"Model: {name} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
        if f1 > best_score:
            best_score = f1
            best_name = name
            best_pipeline = clf

    # Extract feature importances from Best Model if Random Forest
    if "Random Forest" in best_name:
        rf = best_pipeline.named_steps['classifier']
        # get feature names properly built by columntransformer
        cat_features = best_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols).tolist()
        feature_names = numerical_cols + cat_features
        importances = rf.feature_importances_
        # sort and get top 15
        feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:15]
        metrics["feature_importances"] = {k: float(v) for k, v in feat_imp}
    elif "Logistic Regression" in best_name:
        lr = best_pipeline.named_steps['classifier']
        cat_features = best_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols).tolist()
        feature_names = numerical_cols + cat_features
        importances = np.abs(lr.coef_[0])
        feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:15]
        metrics["feature_importances"] = {k: float(v) for k, v in feat_imp}

    # Save metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Selected Best Model: {best_name} with F1-score {best_score:.4f}")
    
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def main():
    download_dataset()
    df = pd.read_csv(DATASET_PATH)
    setup_database(df)
    
    X, y, numerical_cols, categorical_cols = identify_features(df)
    generate_feature_schema(X, numerical_cols, categorical_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_models(X_train, y_train, X_test, y_test, numerical_cols, categorical_cols)
    print("Data Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
