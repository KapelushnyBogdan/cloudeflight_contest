import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score

def correct_temperature(temp, threshold=42):
    return (temp - 32) * 5/9 if temp > threshold else temp

def is_palindrome(path):
    return path == path[::-1]

def analyze_flock(paths, bop_temp):
    """Extract comprehensive features from a flock"""
    all_bops = set()
    for p in paths:
        all_bops.update(p)
    
    # Temperature features
    temps = [bop_temp.get(bop, 20) for bop in all_bops]
    avg_temp = np.mean(temps) if temps else 20
    max_temp = max(temps) if temps else 20
    min_temp = min(temps) if temps else 20
    std_temp = np.std(temps) if len(temps) > 1 else 0
    
    # Palindrome features
    palindrome_count = sum(1 for p in paths if is_palindrome(p))
    palindrome_ratio = palindrome_count / len(paths) if paths else 0
    
    # Path similarity
    all_same_path = len(set(' '.join(p) for p in paths)) == 1
    unique_paths = len(set(' '.join(p) for p in paths))
    path_diversity = unique_paths / len(paths) if paths else 0
    
    # Shared prefix
    shared_prefix = 0
    if not all_same_path and len(paths) > 1:
        min_len = min(len(p) for p in paths)
        for i in range(min_len):
            if len(set(p[i] for p in paths)) == 1:
                shared_prefix += 1
            else:
                break
    
    # BOP diversity per bird
    bops_per_bird = [len(set(p)) for p in paths]
    avg_bops_per_bird = np.mean(bops_per_bird) if bops_per_bird else 0
    max_bops_per_bird = max(bops_per_bird) if bops_per_bird else 0
    min_bops_per_bird = min(bops_per_bird) if bops_per_bird else 0
    std_bops_per_bird = np.std(bops_per_bird) if len(bops_per_bird) > 1 else 0
    
    # Path length statistics
    path_lengths = [len(p) for p in paths]
    avg_path_length = np.mean(path_lengths) if path_lengths else 0
    max_path_length = max(path_lengths) if path_lengths else 0
    min_path_length = min(path_lengths) if path_lengths else 0
    std_path_length = np.std(path_lengths) if len(path_lengths) > 1 else 0
    
    # Interaction features
    bops_per_bird_ratio = len(all_bops) / len(paths) if paths else 0
    temp_times_birds = avg_temp * len(paths)
    
    return {
        # Temperature features
        'avg_temp': avg_temp,
        'max_temp': max_temp,
        'min_temp': min_temp,
        'std_temp': std_temp,
        
        # Path pattern features
        'palindrome_ratio': palindrome_ratio,
        'all_same_path': int(all_same_path),
        'path_diversity': path_diversity,
        'shared_prefix': shared_prefix,
        
        # BOP features
        'num_bops': len(all_bops),
        'avg_bops_per_bird': avg_bops_per_bird,
        'max_bops_per_bird': max_bops_per_bird,
        'min_bops_per_bird': min_bops_per_bird,
        'std_bops_per_bird': std_bops_per_bird,
        'bops_per_bird_ratio': bops_per_bird_ratio,
        
        # Path length features
        'num_birds': len(paths),
        'avg_path_length': avg_path_length,
        'max_path_length': max_path_length,
        'min_path_length': min_path_length,
        'std_path_length': std_path_length,
        
        # Interaction features
        'temp_times_birds': temp_times_birds
    }

def solve_level_4():
    # Load temperature data
    df_temp = pd.read_csv('level_4/all_data_from_level_1.in')
    df_temp.columns = ['BOP', 'Temp', 'Humidity']
    df_temp['Temp_Corrected'] = df_temp['Temp'].apply(correct_temperature)
    bop_temp = dict(zip(df_temp['BOP'].astype(str), df_temp['Temp_Corrected']))
    
    # Read data
    df = pd.read_csv('level_4/level_4.in')
    
    # Group by flock
    flocks = defaultdict(lambda: {'paths': [], 'species': None})
    for _, row in df.iterrows():
        flock_id = str(row['Flock ID'])
        path = row['BOP Path'].split()
        species = row['Species']
        
        flocks[flock_id]['paths'].append(path)
        if species != 'missing':
            flocks[flock_id]['species'] = species
    
    # Extract features for training
    X_train = []
    y_train = []
    
    for flock_id, data in flocks.items():
        if data['species'] and data['species'] != 'missing':
            features = analyze_flock(data['paths'], bop_temp)
            X_train.append(list(features.values()))
            y_train.append(data['species'])
    
    feature_names = list(analyze_flock(list(flocks.values())[0]['paths'], bop_temp).keys())
    
    # Create ensemble of classifiers
    # Random Forest - good for non-linear patterns
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced'
    )
    
    # Gradient Boosting - good for sequential pattern learning
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    
    # Voting ensemble
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft',
        weights=[1.2, 1.0]  # Give slightly more weight to RF
    )
    
    ensemble.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Ensemble cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"CV scores: {cv_scores}")
    
    # Individual model scores
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    
    rf_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    gb_scores = cross_val_score(gb, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"\nRandom Forest CV: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    print(f"Gradient Boosting CV: {gb_scores.mean():.4f} ± {gb_scores.std():.4f}")
    
    # Feature importance from RF
    print("\nTop 10 Random Forest Feature Importances:")
    importances = rf.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {name}: {imp:.4f}")
    
    # Classify missing species
    predictions = {}
    
    for flock_id, data in flocks.items():
        if data['species'] == 'missing' or data['species'] is None:
            features = analyze_flock(data['paths'], bop_temp)
            X_test = [list(features.values())]
            predicted_species = ensemble.predict(X_test)[0]
            predictions[flock_id] = predicted_species
    
    # Write output
    with open('level_4/level_4.out', 'w') as f:
        f.write("Flock ID,Species\n")
        for flock_id in sorted(predictions.keys(), key=int):
            f.write(f"{flock_id},{predictions[flock_id]}\n")
    
    print(f"\nPredictions: {len(predictions)} flocks")
    
    # Show distribution
    from collections import Counter
    species_count = Counter(predictions.values())
    for species in sorted(species_count.keys()):
        print(f"  {species}: {species_count[species]}")

if __name__ == "__main__":
    solve_level_4()
