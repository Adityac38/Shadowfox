import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
MODEL_FILE = "bike_price_model.pkl"
ENCODER_FILE = "bike_label_encoders.pkl"
DATA_FILE = "bike_data.csv"
def train_bike_model():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Place your bike CSV dataset in project root.")
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()
    if "Bike_Name" in df.columns:
        df = df.drop(["Bike_Name"], axis=1)
    categorical_cols = ["Owner", "Seller_Type", "Fuel_Type"]
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    if "Selling_Price" not in df.columns:
        raise KeyError("Dataset must contain 'Selling_Price' column.")
    X = df.drop(["Selling_Price"], axis=1)
    y = df["Selling_Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=250, random_state=42)
    model.fit(X_train, y_train)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODER_FILE, "wb") as f:
        pickle.dump(label_encoders, f)
    return model, label_encoders
def _safe_encode(value, le):
    try:
        classes = list(le.classes_)
        if str(value) in classes:
            return int(np.where(le.classes_ == str(value))[0][0])
        else:
            return 0
    except Exception:
        return 0
def predict_bike_price(features: dict) -> float:
    """
    features keys: Present_Price, Kms_Driven, Owner, Year, Fuel_Type, Seller_Type
    returns float (₹ Lakh)
    """
    if not os.path.exists(MODEL_FILE) or not os.path.exists(ENCODER_FILE):
        if os.path.exists(DATA_FILE):
            train_bike_model()
        else:
            pres = float(features.get("Present_Price", 0))
            kms = float(features.get("Kms_Driven", 0))
            pred = max(0.01, pres * 0.65 - (kms / 100000.0) * pres * 0.08)
            return round(pred, 2)

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_FILE, "rb") as f:
        label_encoders = pickle.load(f)
    x = features.copy()
    for col, le in label_encoders.items():
        if col in x:
            x[col] = _safe_encode(x[col], le)
    df_in = pd.DataFrame([x])
    model_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else df_in.columns
    for c in model_features:
        if c not in df_in.columns:
            df_in[c] = 0
    df_in = df_in[model_features]
    pred = model.predict(df_in)[0]
    return round(float(pred), 2)
if __name__ == "__main__":
    train_bike_model()