# Level 3: Ornithologist Records - Solution

## Problem Overview
Classify 6 flocks of birds into 6 species based on their BOP (Bird Observation Point) traveling patterns. Each flock belongs to exactly one species, and each species is represented once in each dataset.

## Species Characteristics

### 1. Hurracurra Bird
**Behavior**: Prefers to eat in hot regions (high temperature BOPs)
**Detection**: Flock with highest average temperature along paths

### 2. Medieval Bluetit
**Behavior**: Flies back along the same route they came (can't remember much)
**Detection**: All paths are palindromes (symmetric), shortest average path length

### 3. Flanking Blackfinch
**Behavior**: Flocks orbit a chosen land, always trying to stay together
**Detection**: Circular paths (start = end), all paths identical or very similar

### 4. Rusty Goldhammer
**Behavior**: Journey together, split up for way back, then reunite
**Detection**: Strong common prefix AND suffix, but diverse middle sections

### 5. Red Firefinch
**Behavior**: Nest together in main area, then go on individual adventures
**Detection**: Circular paths with high diversity (many unique routes from same nest)

### 6. Sticky Wolfthroat
**Behavior**: Birds of prey that eat Medieval Bluetits, operate in independent groups lying in wait
**Detection**: Paths overlap significantly (>30%) with Medieval Bluetit paths

## Classification Algorithm

The script uses a priority-based classification approach:

1. **Identify Medieval Bluetit** - Find palindromic paths (shortest average length)
2. **Identify Sticky Wolfthroat** - Find paths overlapping >30% with Bluetit
3. **Identify Hurracurra Bird** - Find flock with highest average temperature
4. **Identify Rusty Goldhammer** - Find paths with common prefix & suffix, but not identical
5. **Identify Red Firefinch** - Find circular paths with high diversity
6. **Identify Flanking Blackfinch** - Remaining circular paths (prefer all identical)

## Key Features Analyzed

- **Palindrome**: Path is symmetric (same forward and backward)
- **Circular**: Path starts and ends at same BOP
- **Common Prefix**: Length of shared starting sequence
- **Common Suffix**: Length of shared ending sequence
- **Temperature**: Average temperature along path (with Fahrenheit correction)
- **Diversity**: Number of unique paths vs total paths
- **BOP Overlap**: Ratio of shared BOPs between flocks

## Files

- `solve_level_3.py` - Main solution script
- `test_level_3.py` - Test suite
- `level_3/level_3_a.out` - Output for test case A
- `level_3/level_3_b.out` - Output for test case B  
- `level_3/level_3_c.out` - Output for test case C
- `level_3/level_3_sample_out.txt` - Output for sample (matches expected)

## Usage

```bash
# Run solution
python3 solve_level_3.py

# Run tests
python3 test_level_3.py
```

## Results

Sample classification verified to match expected output:
- Flock 1: Medieval Bluetit ✓
- Flock 2: Sticky Wolfthroat ✓

All test cases (a, b, c) processed successfully with 6 species classifications each.
