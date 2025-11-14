import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
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

# Test different feature combinations
print("="*70)
print("TESTING DIFFERENT FEATURE COMBINATIONS")
print("="*70)

feature_sets = {
    'All 5 features': ['Veg', 'Ins', 'Urb', 'Temp_Corrected', 'Humidity'],
    'Only Temp': ['Temp_Corrected'],
    'Temp + Humidity': ['Temp_Corrected', 'Humidity'],
    'Temp + Veg + Ins + Urb': ['Temp_Corrected', 'Veg', 'Ins', 'Urb'],
    'Temp + Humidity + Veg': ['Temp_Corrected', 'Humidity', 'Veg'],
}

best_rmse = float('inf')
best_config = None

for name, features in feature_sets.items():
    X = known[features].dropna()
    y = known.loc[X.index, 'Score']

    model = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=2, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()
    std = scores.std()

    print(f"{name:30s} RMSE: {rmse:.4f} (+/- {std:.4f})")

    if rmse < best_rmse:
        best_rmse = rmse
        best_config = (name, features)

print(f"\nBest: {best_config[0]} with RMSE: {best_rmse:.4f}")

# Test different models with best features
print("\n" + "="*70)
print(f"TESTING DIFFERENT MODELS WITH: {best_config[0]}")
print("="*70)

X = known[best_config[1]].dropna()
y = known.loc[X.index, 'Score']

models = {
    'Random Forest (n=100, depth=10)': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Random Forest (n=200, depth=15)': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
    'Random Forest (n=500, no limit)': RandomForestRegressor(n_estimators=500, random_state=42),
    'Random Forest (n=1000, no limit)': RandomForestRegressor(n_estimators=1000, random_state=42),
    'Extra Trees (n=500)': ExtraTreesRegressor(n_estimators=500, random_state=42),
    'Extra Trees (n=1000)': ExtraTreesRegressor(n_estimators=1000, random_state=42),
    'Gradient Boosting (n=200)': GradientBoostingRegressor(n_estimators=200, random_state=42),
    'Gradient Boosting (n=500)': GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, random_state=42),
    'Ridge (alpha=0.1)': Ridge(alpha=0.1),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0),
}

best_model_rmse = float('inf')
best_model_config = None

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()
    std = scores.std()
    print(f"{name:40s} RMSE: {rmse:.4f} (+/- {std:.4f})")

    if rmse < best_model_rmse:
        best_model_rmse = rmse
        best_model_config = (name, model)

print(f"\nBest model: {best_model_config[0]} with RMSE: {best_model_rmse:.4f}")
