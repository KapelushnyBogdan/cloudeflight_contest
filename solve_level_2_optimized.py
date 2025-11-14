import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import warnings
warnings.filterwarnings('ignore')

def correct_temperature(temp, threshold=42):
    """Convert Fahrenheit to Celsius if temp > threshold"""
    return (temp - 32) * 5/9 if temp > threshold else temp

def predict_scores_with_temp(level_file, output_file, df_level1, training_data=None):
    """
    Predict missing Bird Love Scores using temperature data from level 1.

    Args:
        level_file: Path to level 2 input file
        output_file: Path to output file
        df_level1: DataFrame with BOP, Temp, Humidity from level 1
        training_data: Optional external training data for level_2_c
    """
    # Read level 2 data
    df = pd.read_csv(level_file)
    df.columns = ['BOP', 'Veg', 'Ins', 'Urb', 'Score']

    # Merge with level 1 temperature data
    df_merged = pd.merge(df, df_level1[['BOP', 'Temp_Corrected']], on='BOP', how='left')

    # Separate known and missing scores
    known = df_merged[df_merged['Score'] != 'missing'].copy()
    if len(known) > 0:
        known['Score'] = known['Score'].astype(float)

    # If we have external training data, use it
    if training_data is not None:
        known = pd.concat([known, training_data], ignore_index=True)

    missing = df_merged[df_merged['Score'] == 'missing'].copy()
    missing = missing.sort_values('BOP')  # Sort by BOP

    # Prepare features and target
    # Add polynomial features
    known['Temp_Squared'] = known['Temp_Corrected'] ** 2
    known['Temp_Cubic'] = known['Temp_Corrected'] ** 3

    X_train = known[['Temp_Corrected', 'Temp_Squared', 'Temp_Cubic']].values
    y_train = known['Score'].values

    # Train the model
    model = ExtraTreesRegressor(
        n_estimators=1500,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Prepare test features
    missing['Temp_Squared'] = missing['Temp_Corrected'] ** 2
    missing['Temp_Cubic'] = missing['Temp_Corrected'] ** 3

    X_test = missing[['Temp_Corrected', 'Temp_Squared', 'Temp_Cubic']].values
    predictions = model.predict(X_test)

    # Create output DataFrame
    output_df = pd.DataFrame({
        'BOP': missing['BOP'],
        'Bird Love Score [<3]': predictions
    })

    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"[OK] {level_file} -> {output_file}: {len(missing)} predictions using {len(known)} training samples")

# Load level 1 data and correct temperature
print("Loading and preprocessing Level 1 data...")
df_level1 = pd.read_csv('level_2/all_data_from_level_1.in')
df_level1.columns = ['BOP', 'Temp', 'Humidity']
df_level1['Temp_Corrected'] = df_level1['Temp'].apply(lambda x: correct_temperature(x, threshold=42))

print("\n" + "="*70)
print("GENERATING PREDICTIONS FOR ALL LEVEL 2 FILES")
print("="*70 + "\n")

# Process level_2_a
predict_scores_with_temp('level_2/level_2_a.in', 'level_2/level_2_a.out', df_level1)

# Process level_2_b
predict_scores_with_temp('level_2/level_2_b.in', 'level_2/level_2_b.out', df_level1)

# For level_2_c, collect training data from previous files
print("\nCollecting training data for level_2_c from level_2_a and level_2_b...")
training_dfs = []

for file in ['level_2/level_2_a.in', 'level_2/level_2_b.in']:
    df = pd.read_csv(file)
    df.columns = ['BOP', 'Veg', 'Ins', 'Urb', 'Score']

    # Merge with temperature data
    df_merged = pd.merge(df, df_level1[['BOP', 'Temp_Corrected']], on='BOP', how='left')

    known = df_merged[df_merged['Score'] != 'missing'].copy()
    if len(known) > 0:
        known['Score'] = known['Score'].astype(float)
        training_dfs.append(known[['Temp_Corrected', 'Score']])

all_training_data = pd.concat(training_dfs, ignore_index=True)
predict_scores_with_temp('level_2/level_2_c.in', 'level_2/level_2_c.out', df_level1, training_data=all_training_data)

print("\n" + "="*70)
print("DONE! All predictions generated successfully.")
print("="*70)
