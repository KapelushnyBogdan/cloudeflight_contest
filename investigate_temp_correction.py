import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Read data
df_level1 = pd.read_csv('level_2/all_data_from_level_1.in')
df_level1.columns = ['BOP', 'Temp', 'Humidity']

df_level2 = pd.read_csv('level_2/level_2_b.in')
df_level2.columns = ['BOP', 'Veg', 'Ins', 'Urb', 'Score']

print("="*70)
print("TESTING DIFFERENT TEMPERATURE CORRECTION THRESHOLDS")
print("="*70)

thresholds = [35, 38, 40, 42, 45, 50]

for threshold in thresholds:
    df_level1_temp = df_level1.copy()
    df_level1_temp['Temp_Corrected'] = df_level1_temp['Temp'].apply(
        lambda x: (x - 32) * 5/9 if x > threshold else x
    )

    df_merged = pd.merge(df_level2, df_level1_temp[['BOP', 'Temp_Corrected']], on='BOP', how='left')

    known = df_merged[df_merged['Score'] != 'missing'].copy()
    known['Score'] = known['Score'].astype(float)

    X = known[['Temp_Corrected']].dropna()
    y = known.loc[X.index, 'Score']

    model = ExtraTreesRegressor(n_estimators=1500, max_depth=None, min_samples_split=3, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()

    print(f"Threshold > {threshold}°C: RMSE = {rmse:.4f}")

# Let's also visualize the relationship
print("\n" + "="*70)
print("ANALYZING TEMP VS SCORE RELATIONSHIP")
print("="*70)

def correct_temp_40(x):
    return (x - 32) * 5/9 if x > 40 else x

df_level1['Temp_Corrected'] = df_level1['Temp'].apply(correct_temp_40)
df_merged = pd.merge(df_level2, df_level1[['BOP', 'Temp_Corrected']], on='BOP', how='left')
known = df_merged[df_merged['Score'] != 'missing'].copy()
known['Score'] = known['Score'].astype(float)

# Sort by temperature and show relationship
sorted_known = known.sort_values('Temp_Corrected')
print("\nSample of Temp vs Score (sorted by temp):")
print(sorted_known[['Temp_Corrected', 'Score']].head(20))

# Check if there are outliers
print("\n" + "="*70)
print("IDENTIFYING POTENTIAL OUTLIERS")
print("="*70)

X = known[['Temp_Corrected']].values
y = known['Score'].values

# Fit model
model = ExtraTreesRegressor(n_estimators=1500, random_state=42)
model.fit(X, y)
predictions = model.predict(X)
errors = np.abs(y - predictions)

# Find largest errors
error_df = known.copy()
error_df['Predicted'] = predictions
error_df['Error'] = errors

print("\nTop 20 largest prediction errors:")
print(error_df.nlargest(20, 'Error')[['BOP', 'Temp_Corrected', 'Score', 'Predicted', 'Error']])

print(f"\nMean absolute error: {np.mean(errors):.2f}")
print(f"Median absolute error: {np.median(errors):.2f}")
print(f"95th percentile error: {np.percentile(errors, 95):.2f}")
print(f"Max error: {np.max(errors):.2f}")
