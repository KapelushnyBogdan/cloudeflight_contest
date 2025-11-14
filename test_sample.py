import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Read sample data
df_sample = pd.read_csv('level_2/level_2_sample.in')
df_sample.columns = ['BOP', 'Veg', 'Ins', 'Urb', 'Score']

known = df_sample[df_sample['Score'] != 'missing'].copy()
known['Score'] = known['Score'].astype(float)
missing = df_sample[df_sample['Score'] == 'missing'].copy()

print("Sample Training Data:")
print(known)
print("\nSample Test Data:")
print(missing)

# Train model
X_train = known[['Veg', 'Ins', 'Urb']]
y_train = known['Score']

model = LinearRegression()
model.fit(X_train, y_train)

# Predict
X_test = missing[['Veg', 'Ins', 'Urb']]
predictions = model.predict(X_test)

print("\nPredictions:")
for bop, pred in zip(missing['BOP'], predictions):
    print(f"BOP {bop}: {pred:.2f}")

print("\nExpected (from sample output):")
print("BOP 3: 29.50")
print("BOP 4: 42.95")

print("\nDifference:")
print(f"BOP 3: {abs(predictions[0] - 29.50):.2f}")
print(f"BOP 4: {abs(predictions[1] - 42.95):.2f}")

# Let's try to figure out the relationship manually
print("\n" + "="*60)
print("MANUAL RELATIONSHIP ANALYSIS")
print("="*60)

# With only 2 training samples, let's see what patterns exist
print("\nTraining data details:")
for idx, row in known.iterrows():
    print(f"BOP={row['BOP']}: Veg={row['Veg']:.2f}, Ins={row['Ins']:.2f}, Urb={row['Urb']:.2f} -> Score={row['Score']:.2f}")

print("\nTest data details:")
for idx, row in missing.iterrows():
    print(f"BOP={row['BOP']}: Veg={row['Veg']:.2f}, Ins={row['Ins']:.2f}, Urb={row['Urb']:.2f}")

print("\nExpected outputs:")
print("BOP=3: 29.50")
print("BOP=4: 42.95")

# Let's try different combinations to reverse engineer
print("\n" + "="*60)
print("TRYING DIFFERENT FORMULAS")
print("="*60)

# Check if there's a simple relationship
for idx, row in known.iterrows():
    veg, ins, urb, score = row['Veg'], row['Ins'], row['Urb'], row['Score']
    print(f"\nBOP {int(row['BOP'])}:")
    print(f"  Veg + Ins - Urb = {veg + ins - urb:.2f}")
    print(f"  Veg - Urb = {veg - urb:.2f}")
    print(f"  Veg + Ins = {veg + ins:.2f}")
    print(f"  Actual Score = {score:.2f}")
