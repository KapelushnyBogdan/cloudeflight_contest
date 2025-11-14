import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def predict_scores(input_file, output_file, training_data=None):
    # Read the CSV file
    df = pd.read_csv(input_file)
    # Assume columns are in order: BOP, Vegetation [%], Insects [g/m²], Urban Light [%], Bird Love Score [<3]
    df.columns = ['BOP', 'Veg', 'Ins', 'Urb', 'Score']

    # Filter known scores (not 'missing')
    known = df[df['Score'] != 'missing'].copy()
    if len(known) > 0:
        known['Score'] = known['Score'].astype(float)

    # If we have external training data, use it
    if training_data is not None:
        known = pd.concat([known, training_data], ignore_index=True)

    # Filter missing scores
    missing = df[df['Score'] == 'missing'].copy()
    missing = missing.sort_values('BOP')  # Sort by BOP as in sample

    # Features and target
    X = known[['Veg', 'Ins', 'Urb']]
    y = known['Score']

    # Train a RandomForestRegressor model (adjust parameters as needed for better RMSE)
    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=20)
    model.fit(X, y)

    # Predict for missing
    X_missing = missing[['Veg', 'Ins', 'Urb']]
    predictions = model.predict(X_missing)

    # Create output DataFrame
    output_df = pd.DataFrame({
        'BOP': missing['BOP'],
        'Bird Love Score [<3]': predictions
    })

    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"Processed {input_file} -> {output_file}: {len(missing)} predictions made using {len(known)} training samples")

# Process level_2_a
predict_scores('level_2/level_2_a.in', 'level_2/level_2_a.out')

# Process level_2_b
predict_scores('level_2/level_2_b.in', 'level_2/level_2_b.out')

# For level_2_c, we need to collect all known data from previous files
training_dfs = []
for file in ['level_2/level_2_a.in', 'level_2/level_2_b.in']:
    df = pd.read_csv(file)
    df.columns = ['BOP', 'Veg', 'Ins', 'Urb', 'Score']
    known = df[df['Score'] != 'missing'].copy()
    if len(known) > 0:
        known['Score'] = known['Score'].astype(float)
        training_dfs.append(known)

all_training_data = pd.concat(training_dfs, ignore_index=True)
predict_scores('level_2/level_2_c.in', 'level_2/level_2_c.out', training_data=all_training_data)
