#!/usr/bin/env python3
"""
Test script for Level 3 bird species classification.
"""

import sys
from solve_level_3 import (
    read_bop_paths,
    read_temperature_data,
    is_palindrome,
    is_circular,
    get_common_prefix_length,
    classify_flocks
)


def test_palindrome():
    """Test palindrome detection."""
    assert is_palindrome([1, 2, 3, 2, 1]) == True
    assert is_palindrome([1, 2, 3, 4, 5]) == False
    assert is_palindrome([1]) == True
    print("✓ Palindrome tests passed")


def test_circular():
    """Test circular path detection."""
    assert is_circular([1, 2, 3, 1]) == True
    assert is_circular([1, 2, 3, 4]) == False
    assert is_circular([1]) == False  # Single BOP is not circular
    print("✓ Circular tests passed")


def test_common_prefix():
    """Test common prefix calculation."""
    paths = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 6, 7],
        [1, 2, 3, 8, 9]
    ]
    assert get_common_prefix_length(paths) == 3
    
    paths2 = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    assert get_common_prefix_length(paths2) == 0
    print("✓ Common prefix tests passed")


def test_sample_classification():
    """Test classification on sample data."""
    flocks = read_bop_paths('level_3/level_3_sample.in')
    temps = read_temperature_data('level_3/all_data_from_level_1.in')
    
    classifications = classify_flocks(flocks, temps)
    
    # Verify sample output
    assert classifications[1] == "Medieval Bluetit", f"Expected Medieval Bluetit, got {classifications[1]}"
    assert classifications[2] == "Sticky Wolfthroat", f"Expected Sticky Wolfthroat, got {classifications[2]}"
    
    print("✓ Sample classification tests passed")


def test_level_3a_classification():
    """Test classification on level_3_a data."""
    flocks = read_bop_paths('level_3/level_3_a.in')
    temps = read_temperature_data('level_3/all_data_from_level_1.in')
    
    classifications = classify_flocks(flocks, temps)
    
    # Verify all 6 species are assigned
    species = set(classifications.values())
    expected_species = {
        "Hurracurra Bird",
        "Medieval Bluetit",
        "Flanking Blackfinch",
        "Rusty Goldhammer",
        "Red Firefinch",
        "Sticky Wolfthroat"
    }
    
    assert species == expected_species, f"Missing or incorrect species: {expected_species - species}"
    
    # Verify specific classifications based on patterns
    assert classifications[1] == "Medieval Bluetit"  # Palindrome
    assert classifications[2] == "Hurracurra Bird"  # Hottest
    assert classifications[6] == "Sticky Wolfthroat"  # Overlaps with bluetit
    
    print("✓ Level 3a classification tests passed")


if __name__ == "__main__":
    try:
        test_palindrome()
        test_circular()
        test_common_prefix()
        test_sample_classification()
        test_level_3a_classification()
        print("\n✓ All tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
