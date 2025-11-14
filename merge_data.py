import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Read level 1 data (temperature and humidity)
df_level1 = pd.read_csv('level_2/all_data_from_level_1.in')
df_level1.columns = ['BOP', 'Temp', 'Humidity']

# Read level 2 data
df_level2 = pd.read_csv('level_2/level_2_b.in')
df_level2.columns = ['BOP', 'Veg', 'Ins', 'Urb', 'Score']

# Merge the data on BOP
df_merged = pd.merge(df_level2, df_level1, on='BOP', how='left')

print("="*60)
print("MERGED DATA SAMPLE")
print("="*60)
print(df_merged.head(20))

# Check for temperature values that might be in Fahrenheit
print("\n" + "="*60)
print("TEMPERATURE ANALYSIS")
print("="*60)
print(f"Temperature min: {df_level1['Temp'].min()}")
print(f"Temperature max: {df_level1['Temp'].max()}")
print(f"Temperature mean: {df_level1['Temp'].mean():.2f}")

# Temperatures in °C should be around 0-40, in °F around 32-104
# Let's check if some values might be in Fahrenheit
print("\nPotential Fahrenheit values (> 40°C):")
fahrenheit_values = df_level1[df_level1['Temp'] > 40]
print(fahrenheit_values)

# Convert potential Fahrenheit to Celsius
df_level1_corrected = df_level1.copy()
df_level1_corrected['Temp_Corrected'] = df_level1_corrected['Temp'].apply(
    lambda x: (x - 32) * 5/9 if x > 40 else x
)

print("\n" + "="*60)
print("CORRECTED TEMPERATURE STATISTICS")
print("="*60)
print(f"Temperature min: {df_level1_corrected['Temp_Corrected'].min():.2f}")
print(f"Temperature max: {df_level1_corrected['Temp_Corrected'].max():.2f}")
print(f"Temperature mean: {df_level1_corrected['Temp_Corrected'].mean():.2f}")

# Merge with corrected temperature
df_merged = pd.merge(df_level2, df_level1_corrected[['BOP', 'Temp_Corrected', 'Humidity']], on='BOP', how='left')

# Filter known scores
known = df_merged[df_merged['Score'] != 'missing'].copy()
known['Score'] = known['Score'].astype(float)

print("\n" + "="*60)
print("CORRELATION ANALYSIS WITH ALL FEATURES")
print("="*60)
for col in ['Veg', 'Ins', 'Urb', 'Temp_Corrected', 'Humidity']:
    if col in known.columns:
        corr = np.corrcoef(known[col].dropna(), known['Score'][known[col].notna()])[0, 1]
        print(f"{col:20s}: {corr:.4f}")

# Test models with all features
print("\n" + "="*60)
print("TESTING MODELS WITH ALL FEATURES")
print("="*60)

X = known[['Veg', 'Ins', 'Urb', 'Temp_Corrected', 'Humidity']].dropna()
y = known.loc[X.index, 'Score']

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest (100)': RandomForestRegressor(n_estimators=100, random_state=42),
    'Random Forest (200)': RandomForestRegressor(n_estimators=200, random_state=42),
    'Random Forest (500)': RandomForestRegressor(n_estimators=500, random_state=42),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()
    std = scores.std()
    print(f"{name:30s} RMSE: {rmse:.4f} (+/- {std:.4f})")

# Train best model and show coefficients
print("\n" + "="*60)
print("LINEAR REGRESSION COEFFICIENTS")
print("="*60)
model = LinearRegression()
model.fit(X, y)
for i, col in enumerate(['Veg', 'Ins', 'Urb', 'Temp_Corrected', 'Humidity']):
    print(f"{col:20s}: {model.coef_[i]:.6f}")
print(f"{'Intercept':20s}: {model.intercept_:.6f}")
