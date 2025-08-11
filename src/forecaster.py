import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

class DemandForecaster:
    def __init__(self, horizon=48):
        self.horizon = horizon
        self.models = {}  # one model per horizon step
        self.feature_cols = None

    def fit(self, X: pd.DataFrame, Y_dict: dict, feature_cols, valid_frac=0.1):
        self.feature_cols = feature_cols
        n = len(X)
        n_valid = int(np.floor(valid_frac * n))
        X_train, X_val = X.iloc[:-n_valid], X.iloc[-n_valid:]
        metrics = {}
        for h in range(1, self.horizon+1):
            y = Y_dict[h]
            y_train, y_val = y.iloc[:-n_valid], y.iloc[-n_valid:]

            model = GradientBoostingRegressor(random_state=42)
            model.fit(X_train, y_train)
            self.models[h] = model

            # validation
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            metrics[h] = {'MAE': mae, 'RMSE': rmse}

        return metrics

    def predict(self, X_recent: pd.DataFrame):
        """
        Given the most recent feature row X_t, predict y_{t+h} for h=1..H using direct models.
        X_recent: single-row DataFrame with same feature columns.
        """
        assert self.feature_cols is not None, "Model not fitted"
        preds = []
        for h in range(1, self.horizon+1):
            m = self.models[h]
            preds.append(float(m.predict(X_recent[self.feature_cols])[0]))
        return np.array(preds)

    def save(self, path_dir: str):
        os.makedirs(path_dir, exist_ok=True)
        with open(os.path.join(path_dir, "forecaster.pkl"), "wb") as f:
            pickle.dump({'horizon': self.horizon,
                         'models': self.models,
                         'feature_cols': self.feature_cols}, f)

    @staticmethod
    def load(path_dir: str):
        with open(os.path.join(path_dir, "forecaster.pkl"), "rb") as f:
            d = pickle.load(f)
        obj = DemandForecaster(horizon=d['horizon'])
        obj.models = d['models']
        obj.feature_cols = d['feature_cols']
        return obj
