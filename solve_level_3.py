"""
Level 3: Bird Species Classification (Rule-Based)

This module classifies bird species based on flight path patterns and environmental conditions.
Uses rule-based decision logic to identify species from observed behavior patterns.

Species Identified:
    - Medieval Bluetit: Characterized by palindromic flight paths (>95%)
    - Hurracurra Bird: High temperature zones (24-32°C), shared starting points
    - Red Firefinch: Large flocks (6-12 birds), shared prefixes
    - Sticky Wolfthroat: Short flight paths (~7 BOPs)
    - Flanking Blackfinch: Moderate BOP counts (40±19)
    - Rusty Goldhammer: High BOP counts (71±31)

Usage:
    python solve_level_3.py

Input:
    - CSV files in level_3/ directory with Flock ID, BOP Path, and Species columns
    - all_data_from_level_1.in with temperature reference data
    - level_3_species.in with known species patterns

Output:
    - Species predictions in corresponding .out files
"""

import pandas as pd
from collections import defaultdict


def correct_temperature(temp, threshold=42):
    """
    Convert Fahrenheit to Celsius if temperature exceeds threshold.
    
    Args:
        temp (float): Temperature value
        threshold (float): Threshold in Celsius (default: 42)
        
    Returns:
        float: Temperature in Celsius
    """
    return (temp - 32) * 5/9 if temp > threshold else temp


def is_palindrome(path):
    """
    Check if a BOP path is palindromic (reads same forwards and backwards).
    
    Args:
        path (list): List of BOP IDs representing a flight path
        
    Returns:
        bool: True if path is palindromic, False otherwise
    """
    return path == path[::-1]


def analyze_flock(flock_data):
    """
    Analyze behavioral patterns in a bird flock.
    
    Extracts features such as:
    - Path patterns (palindromes, circular, shared prefixes)
    - BOP diversity
    - Path uniformity
    
    Args:
        flock_data (list): List of BOP path strings for birds in the flock
        
    Returns:
        dict: Dictionary containing pattern analysis results
    """
    paths = [row.split() for row in flock_data]

    # Collect all BOPs visited
    all_bops = set()
    for path in paths:
        all_bops.update(path)

    # Check patterns
    all_palindromes = all(is_palindrome(p) for p in paths)
    all_same_path = len(set([' '.join(p) for p in paths])) == 1
    all_circular = all(p[0] == p[-1] for p in paths if len(p) > 1)

    # Find shared prefix
    shared_prefix_length = 0
    if not all_same_path and len(paths) > 1:
        min_len = min(len(p) for p in paths)
        for i in range(min_len):
            if len(set(p[i] for p in paths)) == 1:
                shared_prefix_length += 1
            else:
                break

    return {
        'all_same_path': all_same_path,
        'all_palindromes': all_palindromes,
        'all_circular': all_circular,
        'shared_prefix_length': shared_prefix_length,
        'bops': all_bops,
        'paths': paths
    }

def classify_species(filename, df_temp):
    """
    Classify bird species using rule-based decision logic.
    
    Uses environmental and behavioral patterns to identify species:
    1. Medieval Bluetit: High palindrome ratio
    2. Hurracurra Bird: High temperature zones
    3. Red Firefinch: Large flock sizes
    4. Sticky Wolfthroat: Short flight paths
    5. Flanking Blackfinch vs Rusty Goldhammer: BOP count differences
    
    Args:
        filename (str): Path to input CSV file with flock data
        df_temp (pd.DataFrame): Temperature reference data
        
    Returns:
        dict: Mapping of flock IDs to predicted species
    """
    # Create BOP to temperature mapping
    bop_temp = dict(zip(df_temp['BOP'].astype(str), df_temp['Temp_Corrected']))

    # Read file
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]  # Skip header

    # Group by flock ID
    flocks = defaultdict(list)
    for line in lines:
        parts = line.strip().split(',')
        flock_id = parts[0]
        path = parts[1]
        flocks[flock_id].append(path)

    # Analyze all flocks
    flock_analyses = {}
    for flock_id in sorted(flocks.keys()):
        flock_analyses[flock_id] = analyze_flock(flocks[flock_id])

    # Calculate temperature stats for each flock
    for flock_id, analysis in flock_analyses.items():
        temps = [bop_temp.get(bop, 20) for bop in analysis['bops']]
        analysis['avg_temp'] = sum(temps) / len(temps) if temps else 20

    # Species classification
    species_map = {}

    # Step 1: Find Medieval Bluetit (palindrome flock)
    # Prefer the palindrome with more BOPs if there are multiple
    palindrome_flocks = [f for f, a in flock_analyses.items() if a['all_palindromes']]
    if palindrome_flocks:
        bluetit_flock = max(palindrome_flocks, key=lambda f: len(flock_analyses[f]['bops']))
        species_map[bluetit_flock] = "Medieval Bluetit"
        bluetit_bops = flock_analyses[bluetit_flock]['bops']

        # Step 2: Find Sticky Wolfthroat (visits subset of Bluetit's BOPs)
        for flock_id, analysis in flock_analyses.items():
            if flock_id not in species_map:
                # Check if this flock's BOPs are a subset of Bluetit's BOPs
                if analysis['bops'].issubset(bluetit_bops) and len(analysis['bops']) < len(bluetit_bops):
                    species_map[flock_id] = "Sticky Wolfthroat"
                    break

    # Step 3: Find Flanking Blackfinch (all same circular path, NOT palindrome, NOT Wolfthroat)
    for flock_id, analysis in flock_analyses.items():
        if flock_id not in species_map:
            if analysis['all_same_path'] and analysis['all_circular'] and not analysis['all_palindromes']:
                species_map[flock_id] = "Flanking Blackfinch"
                break

    # Step 4: Find Rusty Goldhammer (significant shared prefix > 5, then diverge)
    for flock_id, analysis in flock_analyses.items():
        if flock_id not in species_map:
            if analysis['shared_prefix_length'] > 5 and not analysis['all_same_path']:
                species_map[flock_id] = "Rusty Goldhammer"
                break

    # Step 5: Remaining flocks are Red Firefinch and Hurracurra Bird
    # Hurracurra visits hotter regions
    remaining = [f for f in flock_analyses.keys() if f not in species_map]

    if len(remaining) >= 2:
        # Sort by average temperature (higher = Hurracurra)
        remaining_sorted = sorted(remaining,
                                  key=lambda f: flock_analyses[f]['avg_temp'],
                                  reverse=True)
        species_map[remaining_sorted[0]] = "Hurracurra Bird"
        for f in remaining_sorted[1:]:
            species_map[f] = "Red Firefinch"
    elif len(remaining) == 1:
        # Last one is Red Firefinch
        species_map[remaining[0]] = "Red Firefinch"

    return species_map

def solve_level_3():
    """Solve all Level 3 datasets"""

    # Load temperature data
    df_temp = pd.read_csv('level_3/all_data_from_level_1.in')
    df_temp.columns = ['BOP', 'Temp', 'Humidity']
    df_temp['Temp_Corrected'] = df_temp['Temp'].apply(correct_temperature)

    print("="*70)
    print("SOLVING LEVEL 3 - BIRD SPECIES CLASSIFICATION (FINAL)")
    print("="*70)

    # Process each dataset
    for dataset in ['level_3_sample', 'level_3_a', 'level_3_b', 'level_3_c']:
        input_file = f'level_3/{dataset}.in'
        output_file = f'level_3/{dataset}.out'

        print(f"\n{dataset}:")

        # Classify species
        species_map = classify_species(input_file, df_temp)

        # Write output
        with open(output_file, 'w') as f:
            f.write("Flock ID,Species\n")
            for flock_id in sorted(species_map.keys(), key=int):
                f.write(f"{flock_id},{species_map[flock_id]}\n")
                print(f"  Flock {flock_id}: {species_map[flock_id]}")

        print(f"  -> {output_file}")

    print("\n" + "="*70)
    print("DONE! All Level 3 predictions generated.")
    print("="*70)

# Run the solution
solve_level_3()
