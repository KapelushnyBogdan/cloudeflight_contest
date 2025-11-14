#!/usr/bin/env python3
"""
Level 3: Ornithologist Records - Bird Species Classification

Species characteristics:
1. Hurracurra Bird: Prefers hot regions (high average temperature on path)
2. Medieval Bluetit: Flies back along the same route (palindromic paths)
3. Flanking Blackfinch: Orbit a chosen land (circular paths, same start/end)
4. Rusty Goldhammer: Split up on way back (common prefix, then diverge)
5. Red Firefinch: Nest together, then adventures (common start, then diverge)
6. Sticky Wolfthroat: Independent groups lying in wait (overlapping segments)
"""

import csv
import sys
from collections import defaultdict
from typing import List, Dict, Set, Tuple


def read_bop_paths(filename: str) -> Dict[int, List[List[int]]]:
    """Read BOP paths from input file and group by flock ID."""
    flocks = defaultdict(list)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            flock_id = int(row[0])
            path = [int(x) for x in row[1].split()]
            flocks[flock_id].append(path)
    return dict(flocks)


def read_temperature_data(filename: str) -> Dict[int, float]:
    """Read temperature data from level 1 file."""
    temps = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            bop = int(row[0])
            temp = float(row[1])
            # Apply temperature correction (Fahrenheit bug fix from level 2)
            if temp > 42:
                temp = (temp - 32) * 5 / 9
            temps[bop] = temp
    return temps


def is_palindrome(path: List[int]) -> bool:
    """Check if a path is a palindrome (same route back)."""
    return path == path[::-1]


def is_circular(path: List[int]) -> bool:
    """Check if a path is circular (starts and ends at same location)."""
    return len(path) > 1 and path[0] == path[-1]


def get_common_prefix_length(paths: List[List[int]]) -> int:
    """Get the length of the common prefix across all paths."""
    if not paths or len(paths) < 2:
        return 0
    
    min_len = min(len(p) for p in paths)
    common_len = 0
    
    for i in range(min_len):
        if all(p[i] == paths[0][i] for p in paths):
            common_len += 1
        else:
            break
    
    return common_len


def get_common_suffix_length(paths: List[List[int]]) -> int:
    """Get the length of the common suffix across all paths."""
    if not paths or len(paths) < 2:
        return 0
    
    min_len = min(len(p) for p in paths)
    common_len = 0
    
    for i in range(1, min_len + 1):
        if all(p[-i] == paths[0][-i] for p in paths):
            common_len += 1
        else:
            break
    
    return common_len


def calculate_path_overlaps(paths: List[List[int]]) -> float:
    """Calculate average segment overlap between different paths."""
    if len(paths) < 2:
        return 0.0
    
    # Create segments from paths
    all_segments = []
    for path in paths:
        segments = set()
        for i in range(len(path) - 1):
            segment = tuple(sorted([path[i], path[i+1]]))
            segments.add(segment)
        all_segments.append(segments)
    
    # Calculate pairwise overlap
    total_overlap = 0
    comparisons = 0
    for i in range(len(all_segments)):
        for j in range(i + 1, len(all_segments)):
            intersection = len(all_segments[i] & all_segments[j])
            union = len(all_segments[i] | all_segments[j])
            if union > 0:
                total_overlap += intersection / union
            comparisons += 1
    
    return total_overlap / comparisons if comparisons > 0 else 0.0


def get_average_temperature(path: List[int], temps: Dict[int, float]) -> float:
    """Calculate average temperature along a path."""
    valid_temps = [temps.get(bop, 20) for bop in path if bop in temps]
    return sum(valid_temps) / len(valid_temps) if valid_temps else 20


def calculate_bop_overlap(paths1: List[List[int]], paths2: List[List[int]]) -> float:
    """Calculate BOP overlap between two flocks."""
    bops1 = set()
    for p in paths1:
        bops1.update(p)
    
    bops2 = set()
    for p in paths2:
        bops2.update(p)
    
    if not bops1 or not bops2:
        return 0.0
    
    intersection = len(bops1 & bops2)
    union = len(bops1 | bops2)
    
    return intersection / union if union > 0 else 0.0


def classify_flocks(flocks: Dict[int, List[List[int]]], temps: Dict[int, float]) -> Dict[int, str]:
    """Classify each flock as one of the 6 bird species."""
    
    classifications = {}
    
    for flock_id, paths in flocks.items():
        features = {}
        
        # Feature 1: All paths are palindromes
        features['all_palindromes'] = all(is_palindrome(p) for p in paths)
        features['palindrome_count'] = sum(is_palindrome(p) for p in paths)
        
        # Feature 2: All paths are circular
        features['all_circular'] = all(is_circular(p) for p in paths)
        
        # Feature 3: Common prefix/suffix lengths
        features['common_prefix'] = get_common_prefix_length(paths)
        features['common_suffix'] = get_common_suffix_length(paths)
        
        # Feature 4: Average temperature
        avg_temps = [get_average_temperature(p, temps) for p in paths]
        features['avg_temp'] = sum(avg_temps) / len(avg_temps) if avg_temps else 20
        
        # Feature 5: Path diversity
        features['unique_paths'] = len(set(tuple(p) for p in paths))
        features['total_paths'] = len(paths)
        features['all_same'] = features['unique_paths'] == 1
        
        # Feature 6: Average path length
        features['avg_length'] = sum(len(p) for p in paths) / len(paths)
        
        classifications[flock_id] = features
    
    # Assign species based on features
    assigned_species = {}
    used_species = set()
    
    # Priority 1: Medieval Bluetit - palindrome (flies back same route)
    # Choose the simplest/most consistent palindrome
    palindrome_flocks = [
        (fid, feat) for fid, feat in classifications.items()
        if feat['all_palindromes']
    ]
    
    if palindrome_flocks:
        # Prefer shortest average length (simplest route)
        bluetit_flock = min(palindrome_flocks, key=lambda x: x[1]['avg_length'])
        assigned_species[bluetit_flock[0]] = "Medieval Bluetit"
        used_species.add("Medieval Bluetit")
        bluetit_flock_id = bluetit_flock[0]
    else:
        bluetit_flock_id = None
    
    # Priority 2: Sticky Wolfthroat - overlaps with Medieval Bluetit
    # (preys on bluetit, lies in wait along their routes)
    if bluetit_flock_id is not None and "Sticky Wolfthroat" not in used_species:
        remaining_flocks = [fid for fid in classifications.keys() if fid not in assigned_species]
        if remaining_flocks:
            # Find flock with highest overlap with bluetit
            bluetit_paths = flocks[bluetit_flock_id]
            best_overlap = 0
            best_flock = None
            
            for fid in remaining_flocks:
                overlap = calculate_bop_overlap(bluetit_paths, flocks[fid])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_flock = fid
            
            # Assign if overlap is significant (> 30%)
            if best_flock and best_overlap > 0.3:
                assigned_species[best_flock] = "Sticky Wolfthroat"
                used_species.add("Sticky Wolfthroat")
    
    # Priority 3: Hurracurra Bird - highest average temperature (hot regions)
    remaining_flocks = [fid for fid in classifications.keys() if fid not in assigned_species]
    if remaining_flocks and "Hurracurra Bird" not in used_species:
        hottest_flock = max(remaining_flocks, key=lambda fid: classifications[fid]['avg_temp'])
        assigned_species[hottest_flock] = "Hurracurra Bird"
        used_species.add("Hurracurra Bird")
    
    # Priority 4: Rusty Goldhammer - common prefix AND suffix, but NOT all same
    # (journey together, split up for way back, reunite)
    remaining_flocks = [fid for fid in classifications.keys() if fid not in assigned_species]
    if remaining_flocks and "Rusty Goldhammer" not in used_species:
        best_flock = None
        best_score = 0
        for fid in remaining_flocks:
            feat = classifications[fid]
            # Both prefix and suffix should be significant, but paths should be diverse (not all same)
            if feat['common_prefix'] > 5 and feat['common_suffix'] > 0 and not feat['all_same']:
                score = feat['common_prefix'] + feat['common_suffix']
                if score > best_score:
                    best_score = score
                    best_flock = fid
        if best_flock:
            assigned_species[best_flock] = "Rusty Goldhammer"
            used_species.add("Rusty Goldhammer")
    
    # Priority 5: Red Firefinch - circular paths, high diversity
    # (nest together in main area, go on different adventures)
    remaining_flocks = [fid for fid in classifications.keys() if fid not in assigned_species]
    if remaining_flocks and "Red Firefinch" not in used_species:
        best_flock = None
        best_diversity = 0
        for fid in remaining_flocks:
            feat = classifications[fid]
            # High diversity (many unique paths), all circular
            if feat['all_circular'] and feat['unique_paths'] > 1:
                diversity = feat['unique_paths'] / feat['total_paths']
                if diversity > best_diversity:
                    best_diversity = diversity
                    best_flock = fid
        if best_flock:
            assigned_species[best_flock] = "Red Firefinch"
            used_species.add("Red Firefinch")
    
    # Priority 6: Flanking Blackfinch - circular species that stay together
    # (orbit a chosen land, stay together - all same path or very similar)
    remaining_flocks = [fid for fid in classifications.keys() if fid not in assigned_species]
    if remaining_flocks and "Flanking Blackfinch" not in used_species:
        # Prefer flock with all same paths (strongest "staying together" behavior)
        best_flock = None
        for fid in remaining_flocks:
            feat = classifications[fid]
            if feat['all_circular'] and feat['all_same']:
                best_flock = fid
                break
        
        # If no all-same flock, take first remaining circular
        if not best_flock:
            for fid in remaining_flocks:
                if classifications[fid]['all_circular']:
                    best_flock = fid
                    break
        
        # If still no match, just take first remaining
        if not best_flock:
            best_flock = remaining_flocks[0]
        
        assigned_species[best_flock] = "Flanking Blackfinch"
        used_species.add("Flanking Blackfinch")
    
    # Assign any remaining flocks
    remaining_flocks = [fid for fid in classifications.keys() if fid not in assigned_species]
    remaining_species = [s for s in [
        "Hurracurra Bird", "Medieval Bluetit", "Flanking Blackfinch",
        "Rusty Goldhammer", "Red Firefinch", "Sticky Wolfthroat"
    ] if s not in used_species]
    
    for fid, species in zip(remaining_flocks, remaining_species):
        assigned_species[fid] = species
    
    return assigned_species


def solve_level_3(input_file: str, output_file: str, temp_file: str):
    """Main solver for level 3."""
    # Read data
    flocks = read_bop_paths(input_file)
    temps = read_temperature_data(temp_file)
    
    # Classify flocks
    classifications = classify_flocks(flocks, temps)
    
    # Write output
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Flock ID', 'Species'])
        for flock_id in sorted(classifications.keys()):
            writer.writerow([flock_id, classifications[flock_id]])
    
    print(f"Processed {input_file}")
    print(f"Classifications:")
    for flock_id in sorted(classifications.keys()):
        print(f"  Flock {flock_id}: {classifications[flock_id]}")


if __name__ == "__main__":
    # Process all test cases
    test_cases = [
        ("level_3/level_3_sample.in", "level_3/level_3_sample_out.txt"),
        ("level_3/level_3_a.in", "level_3/level_3_a.out"),
        ("level_3/level_3_b.in", "level_3/level_3_b.out"),
        ("level_3/level_3_c.in", "level_3/level_3_c.out"),
    ]
    
    temp_file = "level_3/all_data_from_level_1.in"
    
    for input_file, output_file in test_cases:
        try:
            solve_level_3(input_file, output_file, temp_file)
            print(f"✓ Generated {output_file}\n")
        except FileNotFoundError as e:
            print(f"✗ File not found: {e}\n")
        except Exception as e:
            print(f"✗ Error processing {input_file}: {e}\n")
