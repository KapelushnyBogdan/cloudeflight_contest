import pandas as pd
import numpy as np

# Read sample data
df_sample = pd.read_csv('level_2/level_2_sample.in')
df_sample.columns = ['BOP', 'Veg', 'Ins', 'Urb', 'Score']

known = df_sample[df_sample['Score'] != 'missing'].copy()
known['Score'] = known['Score'].astype(float)

print("="*60)
print("TESTING DIFFERENT SIMPLE FORMULAS ON SAMPLE DATA")
print("="*60)

formulas = {
    'Veg + Ins': lambda row: row['Veg'] + row['Ins'],
    'Veg + Ins - Urb': lambda row: row['Veg'] + row['Ins'] - row['Urb'],
    'Veg * 1.2 + Ins': lambda row: row['Veg'] * 1.2 + row['Ins'],
    '100 - Urb + Ins': lambda row: 100 - row['Urb'] + row['Ins'],
}

for name, formula in formulas.items():
    print(f"\nFormula: {name}")
    errors = []
    for idx, row in known.iterrows():
        predicted = formula(row)
        error = abs(predicted - row['Score'])
        errors.append(error)
        print(f"  BOP {int(row['BOP'])}: Predicted={predicted:.2f}, Actual={row['Score']:.2f}, Error={error:.2f}")
    print(f"  Mean Absolute Error: {np.mean(errors):.2f}")

# Test the best formula on the missing values
print("\n" + "="*60)
print("TESTING Veg + Ins ON MISSING VALUES")
print("="*60)

missing = df_sample[df_sample['Score'] == 'missing'].copy()
for idx, row in missing.iterrows():
    predicted = row['Veg'] + row['Ins']
    print(f"BOP {int(row['BOP'])}: Veg={row['Veg']:.2f} + Ins={row['Ins']:.2f} = {predicted:.2f}")

print("\nExpected from sample output:")
print("BOP 3: 29.50")
print("BOP 4: 42.95")

# Hmm, that doesn't work either. Let me try the full level_2_b dataset
print("\n" + "="*60)
print("TESTING FORMULAS ON LEVEL_2_B FULL DATASET")
print("="*60)

df_b = pd.read_csv('level_2/level_2_b.in')
df_b.columns = ['BOP', 'Veg', 'Ins', 'Urb', 'Score']

known_b = df_b[df_b['Score'] != 'missing'].copy()
known_b['Score'] = known_b['Score'].astype(float)

for name, formula in formulas.items():
    predictions = known_b.apply(formula, axis=1)
    errors = np.abs(predictions - known_b['Score'])
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(errors)
    print(f"{name:30s} RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# Let's try to learn coefficients
print("\n" + "="*60)
print("FINDING BEST LINEAR COMBINATION")
print("="*60)

from sklearn.linear_model import LinearRegression

X = known_b[['Veg', 'Ins', 'Urb']]
y = known_b['Score']

model = LinearRegression()
model.fit(X, y)

print(f"Coefficients:")
print(f"  Vegetation: {model.coef_[0]:.4f}")
print(f"  Insects: {model.coef_[1]:.4f}")
print(f"  Urban Light: {model.coef_[2]:.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

predictions = model.predict(X)
rmse = np.sqrt(np.mean((predictions - y)**2))
print(f"\nRMSE on training data: {rmse:.4f}")

# Test on sample data
X_sample = known[['Veg', 'Ins', 'Urb']]
y_sample = known['Score']
pred_sample = model.predict(X_sample)
print(f"\nPredictions on sample:")
for i, (bop, pred, actual) in enumerate(zip(known['BOP'], pred_sample, y_sample)):
    print(f"  BOP {int(bop)}: Predicted={pred:.2f}, Actual={actual:.2f}, Error={abs(pred-actual):.2f}")
