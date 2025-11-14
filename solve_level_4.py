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

    # Common start/end
    starts = [p[0] for p in paths if len(p) > 0]
    ends = [p[-1] for p in paths if len(p) > 0]
    common_start_end = len(set(starts)) == 1 and len(set(ends)) == 1 and starts[0] == ends[0]

    # Number of unique BOPs per bird (for variation measure)
    bops_per_bird = [len(set(p)) for p in paths]
    avg_bops_per_bird = np.mean(bops_per_bird) if bops_per_bird else 0

    return {
        'palindrome_ratio': palindrome_ratio,
        'all_same_path': all_same_path,
        'shared_prefix': shared_prefix,
        'avg_temp': avg_temp,
        'num_bops': len(all_bops),
        'num_birds': len(paths),
        'common_start_end': common_start_end,
        'avg_bops_per_bird': avg_bops_per_bird,
        'bops': all_bops
    }

def classify_species(analysis, all_bluetit_bops):
    """Classify based on training data patterns"""

    # Rule 1: Medieval Bluetit - Most have palindromes (95%)
    if analysis['palindrome_ratio'] > 0.5:
        return "Medieval Bluetit"

    # Rule 2: Hurracurra Bird - HIGH temperature (27.53°C avg vs ~19°C)
    if analysis['avg_temp'] > 24:
        return "Hurracurra Bird"

    # Rule 3: Sticky Wolfthroat - LOW temperature (16.55°C avg)
    if analysis['avg_temp'] < 17.5:
        return "Sticky Wolfthroat"

    # Check if subset of Bluetit (Wolfthroat behavior)
    if all_bluetit_bops and analysis['bops'].issubset(all_bluetit_bops):
        if len(analysis['bops']) < len(all_bluetit_bops) * 0.9:
            return "Sticky Wolfthroat"

    # Rule 4: Red Firefinch - ALWAYS shared_prefix = 1, high num_birds (9.2 avg)
    if analysis['shared_prefix'] == 1 and analysis['num_birds'] >= 6:
        return "Red Firefinch"

    # Rule 5: Distinguish between Rusty Goldhammer, Flanking Blackfinch, and Red Firefinch
    # Red Firefinch: shared_prefix = 1 (with fewer birds than above)
    if analysis['shared_prefix'] == 1:
        return "Red Firefinch"

    # Rusty Goldhammer vs Flanking Blackfinch - both have shared_prefix 1-4 typically
    # Need to use temperature and other features

    # Rusty Goldhammer: tends to have slightly higher shared_prefix
    # Flanking Blackfinch: shared_prefix distribution similar but maybe more varied

    # Use temperature as tiebreaker: Flanking Blackfinch median is slightly different
    # Also use number of birds: Rusty Goldhammer ~4.3, Flanking Blackfinch ~4.6

    if analysis['shared_prefix'] >= 5:
        return "Rusty Goldhammer"

    # For shared_prefix 2-4: use temperature
    # Rusty Goldhammer avg temp: 19.75°C
    # Flanking Blackfinch avg temp: 19.50°C
    # Red Firefinch avg temp: 19.63°C

    # They're very close, so use num_birds and avg_bops_per_bird as additional signals
    if analysis['shared_prefix'] >= 2:
        # Higher unique BOPs per bird suggests more divergent paths (Rusty Goldhammer)
        if analysis['avg_bops_per_bird'] > 12:
            return "Rusty Goldhammer"
        else:
            return "Flanking Blackfinch"

    # Default
    return "Red Firefinch"

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
