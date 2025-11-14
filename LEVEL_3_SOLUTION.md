# Level 3 Solution - Bird Species Classification (CORRECTED)

## Task Overview
Match 6 flocks (identified by Flock ID) to 6 bird species based on behavioral patterns observed in their migration paths.

## Species Characteristics (Carefully Analyzed)

### 1. Medieval Bluetit
- **Behavior:** "Fly back along the SAME ROUTE they came" - can't remember much
- **Pattern:** PALINDROME paths (path reads the same forwards and backwards)
- **Detection:** All birds have palindromic paths
- **Example:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 6 → 5 → 4 → 3 → 2 → 1

### 2. Sticky Wolfthroat **[KEY SPECIES]**
- **Behavior:** "Birds of prey, eat Medieval Bluetits" / "Lie in wait along several segments of the Bluetits' routes"
- **Pattern:** Visits BOPs that are a **SUBSET** of the Medieval Bluetit's visited BOPs
- **Detection:** All BOPs visited by this flock are also visited by the Bluetit flock
- **Example:** If Bluetit visits {1,2,3,4,5,6,7}, Wolfthroat might visit {1,2,4,5}
- **CRITICAL INSIGHT:** The subset relationship (Wolfthroat BOPs ⊆ Bluetit BOPs) is the key!

### 3. Flanking Blackfinch
- **Behavior:** "Like Earth around the sun, orbit a chosen land" / "Always stay together"
- **Pattern:** All birds follow the SAME circular path (NOT a palindrome, NOT a subset of Bluetit)
- **Detection:** All birds have identical circular paths, starts and ends at same BOP
- **Example:** 2304 → 432 → 1528 → ... → 2304 (NOT palindromic)

### 4. Rusty Goldhammer
- **Behavior:** "Wonderful journey together, then split up for their way back"
- **Pattern:** Significant shared prefix (journey together), then paths diverge (split up)
- **Detection:** Shared prefix > 5 BOPs, then different return paths
- **Example:**
  - Bird 1: 1180 → 1622 → ... [12 shared] ... → 25 → 436 → ... [different] ... → 1180
  - Bird 2: 1180 → 1622 → ... [12 shared] ... → 2219 → 1216 → ... [different] ... → 1180

### 5. Red Firefinch
- **Behavior:** "Nest together in breeding area, each goes on own adventures"
- **Pattern:** Common home BOP (start/end), different exploration paths
- **Detection:** All circular with same start/end, but minimal shared path (≤2 BOPs)
- **Example:** All start/end at BOP 269, visit different places in between

### 6. Hurracurra Bird
- **Behavior:** "Prefers to eat Hurracurra Worm found in HOT REGIONS"
- **Pattern:** Visits Bird Observation Points with higher temperatures
- **Detection:** Among remaining flocks, has highest average temperature
- **Temperature Data:** Uses Level 1 data with F/C correction (threshold 42°C)

## Classification Algorithm (CORRECTED)

1. **Find Medieval Bluetit**
   - Look for palindromic paths
   - If multiple palindrome flocks, choose the one with MORE BOPs

2. **Find Sticky Wolfthroat** **[CRITICAL FIX]**
   - Find flock whose BOPs are a **SUBSET** of Medieval Bluetit's BOPs
   - Must be a proper subset (not identical, not superset)
   - This is the key distinguishing feature from the sample!

3. **Find Flanking Blackfinch**
   - All birds have same circular path
   - NOT a palindrome
   - NOT the Wolfthroat (already assigned)

4. **Find Rusty Goldhammer**
   - Significant shared prefix (> 5 BOPs)
   - Paths diverge after the shared prefix

5. **Find Hurracurra Bird and Red Firefinch**
   - From remaining flocks, compare average temperatures
   - Higher temperature → Hurracurra Bird
   - Lower temperature → Red Firefinch

## Key Discovery - The Subset Relationship!

**The critical insight that fixed the solution:**

Sticky Wolfthroat's BOPs ⊆ Medieval Bluetit's BOPs

**Sample verification:**
- Bluetit visits: {1, 2, 3, 4, 5, 6, 7}
- Wolfthroat visits: {1, 2, 4, 5} ⊂ {1, 2, 3, 4, 5, 6, 7} ✓

**Level_3_a verification:**
- Bluetit (Flock 6) visits: {359, 726, 64, 1791, 554, 1985}
- Wolfthroat (Flock 1) visits: {359, 726, 1791} ⊂ {359, 726, 64, 1791, 554, 1985} ✓

## Results (Verified with Sample)

### Sample Dataset
- Flock 1: Medieval Bluetit ✓
- Flock 2: Sticky Wolfthroat ✓
**(Matches expected output exactly!)**

### Level 3a
1. Sticky Wolfthroat
2. Hurracurra Bird
3. Red Firefinch
4. Rusty Goldhammer
5. Flanking Blackfinch
6. Medieval Bluetit

### Level 3b
1. Red Firefinch
2. Sticky Wolfthroat
3. Hurracurra Bird
4. Medieval Bluetit
5. Flanking Blackfinch
6. Rusty Goldhammer

### Level 3c
1. Hurracurra Bird
2. Flanking Blackfinch
3. Rusty Goldhammer
4. Medieval Bluetit
5. Sticky Wolfthroat
6. Red Firefinch

## Solution File
`solve_level_3.py` - Complete corrected solution for all Level 3 datasets
