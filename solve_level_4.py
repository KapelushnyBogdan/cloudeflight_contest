import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def correct_temperature(temp, threshold=42):
    return (temp - 32) * 5/9 if temp > threshold else temp

def is_palindrome(path):
    return path == path[::-1]

def extract_flock_features(paths, bop_temp):
    """Extract features from a flock's paths"""
    all_bops = set()
    for p in paths:
        all_bops.update(p)

    # Calculate temperatures
    temps = [bop_temp.get(bop, 20) for bop in all_bops]
    avg_temp = sum(temps) / len(temps) if temps else 20

    # Check patterns
    all_palindromes = all(is_palindrome(p) for p in paths)
    all_same_path = len(set(' '.join(p) for p in paths)) == 1
    all_circular = all(p[0] == p[-1] for p in paths if len(p) > 1)

    # Shared prefix
    shared_prefix = 0
    if not all_same_path and len(paths) > 1:
        min_len = min(len(p) for p in paths)
        for i in range(min_len):
            if len(set(p[i] for p in paths)) == 1:
                shared_prefix += 1
            else:
                break

    # Path lengths
    path_lengths = [len(p) for p in paths]

    # Common start/end
    starts = [p[0] for p in paths if len(p) > 0]
    ends = [p[-1] for p in paths if len(p) > 0]
    common_start_end = len(set(starts)) == 1 and len(set(ends)) == 1 and starts[0] == ends[0]

    return {
        'all_palindromes': int(all_palindromes),
        'all_same_path': int(all_same_path),
        'all_circular': int(all_circular),
        'shared_prefix': shared_prefix,
        'avg_temp': avg_temp,
        'num_bops': len(all_bops),
        'num_birds': len(paths),
        'avg_path_length': np.mean(path_lengths),
        'std_path_length': np.std(path_lengths) if len(path_lengths) > 1 else 0,
        'common_start_end': int(common_start_end),
        'bops': all_bops
    }

def solve_level_4():
    # Load temperature data
    df_temp = pd.read_csv('level_4/all_data_from_level_1.in')
    df_temp.columns = ['BOP', 'Temp', 'Humidity']
    df_temp['Temp_Corrected'] = df_temp['Temp'].apply(correct_temperature)
    bop_temp = dict(zip(df_temp['BOP'].astype(str), df_temp['Temp_Corrected']))

    # Read level 4 data
    df = pd.read_csv('level_4/level_4.in')

    # Group by flock ID
    flocks = defaultdict(lambda: {'paths': [], 'species': None})
    for _, row in df.iterrows():
        flock_id = str(row['Flock ID'])
        path = row['BOP Path'].split()
        species = row['Species']

        flocks[flock_id]['paths'].append(path)
        if species != 'missing':
            flocks[flock_id]['species'] = species

    # Extract features for all flocks
    flock_features = {}
    for flock_id, data in flocks.items():
        features = extract_flock_features(data['paths'], bop_temp)
        features['species'] = data['species']
        flock_features[flock_id] = features

    # Find Medieval Bluetit flocks (for Sticky Wolfthroat detection)
    bluetit_bops = set()
    for flock_id, features in flock_features.items():
        if features['species'] == 'Medieval Bluetit':
            bluetit_bops.update(features['bops'])

    # Add "is_subset_of_bluetit" feature
    for flock_id, features in flock_features.items():
        is_subset = features['bops'].issubset(bluetit_bops) if bluetit_bops else False
        features['is_subset_bluetit'] = int(is_subset)

    # Prepare training data (known species)
    train_flocks = {k: v for k, v in flock_features.items() if v['species'] != 'missing' and v['species'] is not None}
    test_flocks = {k: v for k, v in flock_features.items() if v['species'] == 'missing' or v['species'] is None}

    # Feature names
    feature_names = ['all_palindromes', 'all_same_path', 'all_circular', 'shared_prefix',
                     'avg_temp', 'num_bops', 'num_birds', 'avg_path_length', 'std_path_length',
                     'common_start_end', 'is_subset_bluetit']

    # Create training data
    X_train = []
    y_train = []
    for flock_id, features in train_flocks.items():
        X_train.append([features[f] for f in feature_names])
        y_train.append(features['species'])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    clf.fit(X_train, y_train)

    # Predict for test data
    predictions = {}
    for flock_id, features in test_flocks.items():
        X_test = np.array([[features[f] for f in feature_names]])
        pred = clf.predict(X_test)[0]
        predictions[flock_id] = pred

    # Write output
    with open('level_4/level_4.out', 'w') as f:
        f.write("Flock ID,Species\n")
        for flock_id in sorted(predictions.keys(), key=int):
            f.write(f"{flock_id},{predictions[flock_id]}\n")

    print(f"Predictions for {len(predictions)} flocks written to level_4/level_4.out")
    print(f"Training data: {len(train_flocks)} flocks")
    print(f"Test data: {len(test_flocks)} flocks")

    # Show some predictions
    print("\nSample predictions:")
    for i, (flock_id, species) in enumerate(sorted(predictions.items(), key=lambda x: int(x[0]))[:10]):
        print(f"  Flock {flock_id}: {species}")

solve_level_4()
