import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def correct_temperature(temp):
    """Convert Fahrenheit to Celsius if temp > 40"""
    return (temp - 32) * 5/9 if temp > 40 else temp

# Read and merge data
df_level1 = pd.read_csv('level_2/all_data_from_level_1.in')
df_level1.columns = ['BOP', 'Temp', 'Humidity']
df_level1['Temp_Corrected'] = df_level1['Temp'].apply(correct_temperature)

df_level2 = pd.read_csv('level_2/level_2_b.in')
df_level2.columns = ['BOP', 'Veg', 'Ins', 'Urb', 'Score']

df_merged = pd.merge(df_level2, df_level1[['BOP', 'Temp_Corrected', 'Humidity']], on='BOP', how='left')

known = df_merged[df_merged['Score'] != 'missing'].copy()
known['Score'] = known['Score'].astype(float)

# Add engineered features
known['Temp_Squared'] = known['Temp_Corrected'] ** 2
known['Temp_Cubic'] = known['Temp_Corrected'] ** 3
known['Temp_x_Humidity'] = known['Temp_Corrected'] * known['Humidity']
known['Temp_x_Veg'] = known['Temp_Corrected'] * known['Veg']
known['Veg_x_Ins'] = known['Veg'] * known['Ins']

print("="*70)
print("TESTING WITH POLYNOMIAL AND INTERACTION FEATURES")
print("="*70)

feature_sets = {
    'Temp only': ['Temp_Corrected'],
    'Temp + Temp²': ['Temp_Corrected', 'Temp_Squared'],
    'Temp + Temp² + Temp³': ['Temp_Corrected', 'Temp_Squared', 'Temp_Cubic'],
    'Temp + Temp² + Humidity': ['Temp_Corrected', 'Temp_Squared', 'Humidity'],
    'Temp + Temp² + Veg + Ins + Urb': ['Temp_Corrected', 'Temp_Squared', 'Veg', 'Ins', 'Urb'],
    'Temp + Temp² + Temp*Hum': ['Temp_Corrected', 'Temp_Squared', 'Temp_x_Humidity'],
    'Temp + Temp² + Temp*Veg': ['Temp_Corrected', 'Temp_Squared', 'Temp_x_Veg'],
    'All features + interactions': ['Temp_Corrected', 'Temp_Squared', 'Temp_Cubic', 'Humidity', 'Veg', 'Ins', 'Urb', 'Temp_x_Humidity', 'Temp_x_Veg', 'Veg_x_Ins'],
}

best_rmse = float('inf')
best_features = None

for name, features in feature_sets.items():
    X = known[features].dropna()
    y = known.loc[X.index, 'Score']

    model = ExtraTreesRegressor(n_estimators=1000, random_state=42, min_samples_leaf=1, min_samples_split=2)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()
    std = scores.std()

    print(f"{name:40s} RMSE: {rmse:.4f} (+/- {std:.4f})")

    if rmse < best_rmse:
        best_rmse = rmse
        best_features = features

print(f"\nBest features: {best_features} with RMSE: {best_rmse:.4f}")

# Fine-tune the best model
print("\n" + "="*70)
print("FINE-TUNING EXTRA TREES WITH BEST FEATURES")
print("="*70)

X = known[best_features].dropna()
y = known.loc[X.index, 'Score']

configs = [
    {'n_estimators': 1000, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
    {'n_estimators': 2000, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
    {'n_estimators': 1500, 'max_depth': 50, 'min_samples_split': 2, 'min_samples_leaf': 1},
    {'n_estimators': 1500, 'max_depth': None, 'min_samples_split': 3, 'min_samples_leaf': 1},
    {'n_estimators': 2000, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 2},
]

best_config_rmse = float('inf')
best_config = None

for config in configs:
    model = ExtraTreesRegressor(random_state=42, **config)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()
    std = scores.std()

    config_str = f"n={config['n_estimators']}, depth={config['max_depth']}, split={config['min_samples_split']}, leaf={config['min_samples_leaf']}"
    print(f"{config_str:60s} RMSE: {rmse:.4f} (+/- {std:.4f})")

    if rmse < best_config_rmse:
        best_config_rmse = rmse
        best_config = config

print(f"\nBest config: {best_config}")
print(f"Best RMSE: {best_config_rmse:.4f}")
