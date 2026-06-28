import joblib
import numpy as np
scaler = joblib.load('/data/quyhv/Geo_KAN_Project/models/global_scaler.pkl')
print(f"Scaler means max: {np.max(scaler.mean_)}")
print(f"Scaler var min: {np.min(scaler.var_)}")
