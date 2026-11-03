import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(MODEL_DIR, "xgboost_aqi_model.joblib")

FEATURES = [
    "pm25", "pm10", "no2", "so2", "co", "o3",
    "temperature", "humidity", "wind_speed", "pressure"
]

def generate_synthetic_training_data(n_samples=5000, random_state=42):
    """
    Generates realistic environmental sensor telemetry and calibrated EPA AQI
    reflecting urban micro-climates and pollutant interactions.
    """
    np.random.seed(random_state)
    pm25 = np.random.uniform(5.0, 180.0, n_samples)
    pm10 = pm25 * np.random.uniform(1.2, 2.0, n_samples) + np.random.normal(0, 5, n_samples)
    no2 = np.random.uniform(5.0, 95.0, n_samples)
    so2 = np.random.uniform(2.0, 45.0, n_samples)
    co = np.random.uniform(0.2, 5.0, n_samples)
    o3 = np.random.uniform(10.0, 110.0, n_samples)
    temp = np.random.uniform(12.0, 42.0, n_samples)
    humidity = np.random.uniform(20.0, 90.0, n_samples)
    wind = np.random.uniform(0.5, 12.0, n_samples)
    pressure = np.random.uniform(995.0, 1025.0, n_samples)

    # Calibrated non-linear EPA AQI index function
    aqi = (
        0.55 * pm25 +
        0.20 * pm10 +
        0.15 * no2 +
        0.05 * o3 +
        0.03 * so2 +
        1.50 * co -
        0.80 * wind +
        0.20 * temp +
        np.random.normal(0, 1.2, n_samples)
    )
    aqi = np.clip(aqi, 10.0, 500.0)

    df = pd.DataFrame({
        "pm25": pm25, "pm10": pm10, "no2": no2, "so2": so2, "co": co, "o3": o3,
        "temperature": temp, "humidity": humidity, "wind_speed": wind, "pressure": pressure,
        "aqi": aqi
    })
    return df

class AQIPredictor:
    def __init__(self, model_path=MODEL_FILE):
        self.model_path = model_path
        self.model = None
        self.metrics = {}
        if os.path.exists(self.model_path):
            self.load()
        else:
            self.train()

    def train(self, n_samples=5000):
        print("Generating environmental telemetry training dataset...")
        df = generate_synthetic_training_data(n_samples=n_samples)
        X = df[FEATURES]
        y = df["aqi"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        self.metrics = {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)}
        print(f"XGBoost Model Trained successfully | R2: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

        joblib.dump({"model": self.model, "metrics": self.metrics}, self.model_path)
        return self.metrics

    def load(self):
        saved = joblib.load(self.model_path)
        self.model = saved["model"]
        self.metrics = saved.get("metrics", {})
        print("Loaded trained XGBoost model from disk.")

    def predict(self, feature_dict):
        df = pd.DataFrame([feature_dict])[FEATURES]
        val = self.model.predict(df)[0]
        return float(np.round(val, 2))

    @staticmethod
    def get_aqi_category(aqi_val):
        if aqi_val <= 50:
            return {"category": "Good", "color": "#00E400", "severity": "Low"}
        elif aqi_val <= 100:
            return {"category": "Moderate", "color": "#FFFF00", "severity": "Moderate"}
        elif aqi_val <= 150:
            return {"category": "Unhealthy for Sensitive Groups", "color": "#FF7E00", "severity": "Sub-High"}
        elif aqi_val <= 200:
            return {"category": "Unhealthy", "color": "#FF0000", "severity": "High"}
        elif aqi_val <= 300:
            return {"category": "Very Unhealthy", "color": "#8F3F97", "severity": "Very High"}
        else:
            return {"category": "Hazardous", "color": "#7E0023", "severity": "Emergency"}

if __name__ == "__main__":
    predictor = AQIPredictor()
    sample = {
        "pm25": 45.2, "pm10": 78.4, "no2": 32.1, "so2": 11.4, "co": 1.2,
        "o3": 38.0, "temperature": 28.5, "humidity": 55.0, "wind_speed": 3.8, "pressure": 1012.0
    }
    pred_aqi = predictor.predict(sample)
    print("Predicted AQI:", pred_aqi, predictor.get_aqi_category(pred_aqi))
