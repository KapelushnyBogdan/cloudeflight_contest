import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Read the data
df = pd.read_csv('level_2/level_2_b.in')
df.columns = ['BOP', 'Veg', 'Ins', 'Urb', 'Score']

# Split into known and missing
known = df[df['Score'] != 'missing'].copy()
known['Score'] = known['Score'].astype(float)

print("=" * 60)
print("DATA ANALYSIS FOR LEVEL 2B")
print("=" * 60)
print(f"\nTotal samples: {len(df)}")
print(f"Known scores: {len(known)}")
print(f"Missing scores: {len(df) - len(known)}")

print("\nScore statistics:")
print(known['Score'].describe())

print("\nFeature correlations with Score:")
X = known[['Veg', 'Ins', 'Urb']]
y = known['Score']
for col in ['Veg', 'Ins', 'Urb']:
    corr = np.corrcoef(known[col], known['Score'])[0, 1]
    print(f"{col}: {corr:.4f}")

print("\n" + "=" * 60)
print("TESTING DIFFERENT MODELS")
print("=" * 60)

# Test different models with cross-validation
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1)': Ridge(alpha=1),
    'Ridge (alpha=10)': Ridge(alpha=10),
    'Lasso (alpha=0.1)': Lasso(alpha=0.1),
    'Random Forest (100, depth=10)': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Random Forest (200, depth=20)': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42),
    'Random Forest (500, no limit)': RandomForestRegressor(n_estimators=500, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    # Use negative RMSE for cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()
    std = scores.std()
    results[name] = (rmse, std)
    print(f"{name:40s} RMSE: {rmse:.4f} (+/- {std:.4f})")

print("\n" + "=" * 60)
print("TESTING WITH FEATURE ENGINEERING")
print("=" * 60)

# Test with polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

models_poly = {
    'Linear Regression (poly-2)': LinearRegression(),
    'Ridge (poly-2, alpha=1)': Ridge(alpha=1),
    'Ridge (poly-2, alpha=10)': Ridge(alpha=10),
}

for name, model in models_poly.items():
    scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()
    std = scores.std()
    results[name] = (rmse, std)
    print(f"{name:40s} RMSE: {rmse:.4f} (+/- {std:.4f})")

# Test with scaled features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "=" * 60)
print("TESTING WITH SCALED FEATURES")
print("=" * 60)

models_scaled = {
    'Linear Regression (scaled)': LinearRegression(),
    'Ridge (scaled, alpha=1)': Ridge(alpha=1),
}

for name, model in models_scaled.items():
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()
    std = scores.std()
    results[name] = (rmse, std)
    print(f"{name:40s} RMSE: {rmse:.4f} (+/- {std:.4f})")

print("\n" + "=" * 60)
print("BEST MODELS RANKED BY RMSE")
print("=" * 60)
sorted_results = sorted(results.items(), key=lambda x: x[1][0])
for i, (name, (rmse, std)) in enumerate(sorted_results[:10], 1):
    print(f"{i}. {name:40s} RMSE: {rmse:.4f} (+/- {std:.4f})")
