import pandas as pd
import numpy as np
from collections import defaultdict

def correct_temperature(temp, threshold=42):
    return (temp - 32) * 5/9 if temp > threshold else temp

def is_palindrome(path):
    return path == path[::-1]

def analyze_flock(paths, bop_temp):
    """Analyze a flock's patterns"""
    all_bops = set()
    for p in paths:
        all_bops.update(p)

    # Temperatures
    temps = [bop_temp.get(bop, 20) for bop in all_bops]
    avg_temp = sum(temps) / len(temps) if temps else 20

    # Patterns
    palindrome_count = sum(1 for p in paths if is_palindrome(p))
    palindrome_ratio = palindrome_count / len(paths) if paths else 0

    all_same_path = len(set(' '.join(p) for p in paths)) == 1

    # Shared prefix
    shared_prefix = 0
    if not all_same_path and len(paths) > 1:
        min_len = min(len(p) for p in paths)
        for i in range(min_len):
            if len(set(p[i] for p in paths)) == 1:
                shared_prefix += 1
            else:
                break

    # Number of unique BOPs per bird (for variation measure)
    bops_per_bird = [len(set(p)) for p in paths]
    avg_bops_per_bird = np.mean(bops_per_bird) if bops_per_bird else 0

    # Path length
    path_lengths = [len(p) for p in paths]
    avg_path_length = np.mean(path_lengths) if path_lengths else 0

    return {
        'palindrome_ratio': palindrome_ratio,
        'all_same_path': all_same_path,
        'shared_prefix': shared_prefix,
        'avg_temp': avg_temp,
        'num_bops': len(all_bops),
        'num_birds': len(paths),
        'avg_bops_per_bird': avg_bops_per_bird,
        'avg_path_length': avg_path_length,
        'bops': all_bops
    }

def classify_species(analysis, all_bluetit_bops):
    """Classify based on training data patterns
    
    Decision tree based on training data analysis:
    1. avg_temp > 24.5 → Hurracurra Bird (catches birds with temp > 24.6)
    2. palindrome_ratio > 0.5 AND avg_path_length >= 5 → Medieval Bluetit
    3. avg_path_length < 12 → Sticky Wolfthroat or Red Firefinch
       - Use num_birds >= 6 to distinguish (Red Firefinch min is 6)
    4. num_birds >= 6 → Red Firefinch (for longer paths)
    5. Remaining: Rusty Goldhammer vs Flanking Blackfinch
       - Use decision tree rules
    """

    # Rule 1: Hurracurra Bird - VERY HIGH temperature
    # With shared_prefix=1, Hurracurra has min temp 24.6
    # Use 24.5 to capture all Hurracurra while avoiding most Red Firefinch
    if analysis['avg_temp'] > 24.5:
        return "Hurracurra Bird"

    # Rule 2: Medieval Bluetit - HIGH palindrome ratio (47/49 = 95.9%)
    # But exclude VERY SHORT paths (< 5) which are likely Sticky Wolfthroat
    if analysis['palindrome_ratio'] > 0.5:
        if analysis['avg_path_length'] < 5:
            return "Sticky Wolfthroat"
        else:
            return "Medieval Bluetit"

    # Rule 3: Distinguish between Sticky Wolfthroat and Red Firefinch
    # For short paths (< 12):
    # - Red Firefinch: num_birds >= 6 (min is 6, avg is 9.2)
    # - Sticky Wolfthroat: num_birds < 6 (max is 6, avg is 4.4)
    if analysis['avg_path_length'] < 12:
        if analysis['num_birds'] >= 6:
            return "Red Firefinch"
        else:
            return "Sticky Wolfthroat"

    # Rule 4: Red Firefinch with longer paths but still high num_birds
    # Some Red Firefinch have avg_path_length >= 12 but still high num_birds
    if analysis['num_birds'] >= 6:
        return "Red Firefinch"

    # Rule 5: Distinguish between Rusty Goldhammer and Flanking Blackfinch
    # Using decision tree rules trained on the data (89.4% accuracy):
    # Primary feature: num_bops (64% importance)
    # Secondary features: avg_bops_per_bird, shared_prefix, num_birds, avg_path_length
    
    if analysis['num_bops'] <= 59:
        if analysis['avg_bops_per_bird'] <= 19:
            if analysis['num_bops'] <= 27.5:
                return "Flanking Blackfinch"
            else:
                return "Rusty Goldhammer"
        else:  # avg_bops_per_bird > 19
            if analysis['shared_prefix'] <= 10.5:
                return "Flanking Blackfinch"
            else:
                return "Rusty Goldhammer"
    else:  # num_bops > 59
        if analysis['avg_path_length'] <= 36:
            return "Rusty Goldhammer"
        else:
            if analysis['num_birds'] <= 3.5:
                return "Rusty Goldhammer"
            else:
                return "Flanking Blackfinch"

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

    # Find all Medieval Bluetit BOPs
    all_bluetit_bops = set()
    for flock_id, data in flocks.items():
        if data['species'] == 'Medieval Bluetit':
            for path in data['paths']:
                all_bluetit_bops.update(path)

    # Classify missing species
    predictions = {}

    for flock_id, data in flocks.items():
        if data['species'] == 'missing' or data['species'] is None:
            analysis = analyze_flock(data['paths'], bop_temp)
            predicted_species = classify_species(analysis, all_bluetit_bops)
            predictions[flock_id] = predicted_species

    # Write output
    with open('level_4/level_4.out', 'w') as f:
        f.write("Flock ID,Species\n")
        for flock_id in sorted(predictions.keys(), key=int):
            f.write(f"{flock_id},{predictions[flock_id]}\n")

    print(f"Predictions: {len(predictions)} flocks")

    # Show distribution
    from collections import Counter
    species_count = Counter(predictions.values())
    for species in sorted(species_count.keys()):
        print(f"  {species}: {species_count[species]}")

solve_level_4()
