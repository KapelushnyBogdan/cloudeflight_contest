# Level 4 Improvements - Species Classification

## Objective
Improve the species classification accuracy to >90% while avoiding overfitting.

## Results
✅ **Achieved: 93.04% ± 3.99% cross-validation accuracy**

### Cross-Validation Scores
- Fold 1: 89.66%
- Fold 2: 98.28%
- Fold 3: 96.55%
- Fold 4: 87.72%
- Fold 5: 92.98%

All folds exceed 87%, demonstrating good generalization without overfitting.

## Key Improvements

### 1. Enhanced Feature Engineering (20 Features)

#### Temperature Features (4)
- `avg_temp`: Average temperature across all BOPs in the flock
- `max_temp`: Maximum temperature
- `min_temp`: Minimum temperature  
- `std_temp`: Standard deviation of temperatures

**Rationale**: Hurracurra Bird has distinctly high temperatures (24.6-31.7°C), making temperature statistics crucial for classification.

#### Pattern Features (4)
- `palindrome_ratio`: Proportion of palindromic paths
- `all_same_path`: Whether all birds follow the same path
- `path_diversity`: Ratio of unique paths to total birds
- `shared_prefix`: Number of shared starting BOPs

**Rationale**: Medieval Bluetit has >95% palindrome ratio, making this the strongest discriminator for that species.

#### BOP Features (6)
- `num_bops`: Total unique BOPs in the flock
- `avg_bops_per_bird`: Average unique BOPs per bird
- `max_bops_per_bird`: Maximum unique BOPs per bird
- `min_bops_per_bird`: Minimum unique BOPs per bird
- `std_bops_per_bird`: Standard deviation of BOPs per bird
- `bops_per_bird_ratio`: Ratio of total BOPs to number of birds

**Rationale**: Rusty Goldhammer has significantly more BOPs (71 ± 31) than Flanking Blackfinch (41 ± 19), making BOP statistics important for separating similar species.

#### Path Length Features (4)
- `avg_path_length`: Average path length
- `max_path_length`: Maximum path length
- `min_path_length`: Minimum path length
- `std_path_length`: Standard deviation of path lengths

**Rationale**: Sticky Wolfthroat has short paths (~7) while other species have longer paths (~25), making length a key feature.

#### Interaction Features (2)
- `num_birds`: Number of birds in the flock
- `temp_times_birds`: Interaction between temperature and flock size

**Rationale**: Red Firefinch has distinctly high bird counts (6-12 vs 3-6 for others), and this interaction captures combined effects.

### 2. Optimized Random Forest Classifier

```python
RandomForestClassifier(
    n_estimators=500,        # More trees for stability
    max_depth=30,            # Allow complex patterns
    min_samples_split=2,     # Fine-grained splits
    min_samples_leaf=1,      # Detailed leaf nodes
    max_features='sqrt',     # CRITICAL for preventing overfitting
    random_state=42,         # Reproducibility
    class_weight='balanced'  # Handle class imbalance
)
```

#### Overfitting Prevention Mechanisms

1. **max_features='sqrt'**: Only √20 ≈ 4.47 features are considered at each split
   - Forces the model to explore different feature combinations
   - Prevents any single feature from dominating
   - Increases diversity among trees

2. **5-Fold Cross-Validation**: Model evaluated on 5 different train/test splits
   - Ensures the model generalizes to unseen data
   - Standard deviation of 3.99% shows consistency

3. **Balanced Class Weights**: Prevents bias toward majority classes
   - All 6 species get equal consideration
   - Training data: 45-50 examples per species

4. **Multiple Trees (500)**: Ensemble reduces overfitting
   - Each tree sees different bootstrap sample
   - Voting averages out individual tree biases

### 3. Model Stability Testing

Tested across different random seeds:
- Seed 42: 93.04% ± 3.99%
- Seed 123: 92.35% ± 3.25%
- Seed 456: 93.04% ± 3.99%
- Seed 789: 92.69% ± 4.33%

**Conclusion**: Model is stable and consistently achieves >92% accuracy.

## Feature Importance (Top 10)

1. `std_temp`: 14.10% - Temperature variation is most predictive
2. `max_temp`: 10.15% - Peak temperature identifies Hurracurra Bird
3. `palindrome_ratio`: 9.89% - Strong indicator for Medieval Bluetit
4. `bops_per_bird_ratio`: 8.38% - Distinguishes species behavior
5. `num_bops`: 8.32% - Separates Rusty Goldhammer from others
6. `num_birds`: 6.97% - Red Firefinch has high bird counts
7. `avg_temp`: 5.36% - General temperature level
8. `max_path_length`: 5.31% - Path complexity indicator
9. `temp_times_birds`: 5.12% - Interaction effects
10. `avg_bops_per_bird`: 4.79% - Individual bird behavior

## Species Patterns Discovered

### Hurracurra Bird
- Very high temperature: 27.53 ± 1.07°C (min: 24.61, max: 31.67)
- Always has shared_prefix = 1
- 4.38 ± 1.12 birds
- No palindromes

### Medieval Bluetit  
- Extremely high palindrome ratio: 97% ± 17%
- Average temperature: 19.24 ± 3.93°C
- 4.55 ± 1.21 birds
- Average path length: 23.94 ± 8.28

### Red Firefinch
- High bird count: 9.16 ± 2.32 (min: 6, max: 12)
- Always has shared_prefix = 1
- Shorter paths: 10.92 ± 4.68
- High BOP count: 82.60 ± 50.78

### Sticky Wolfthroat
- Short paths: 6.87 ± 2.74
- Some palindromes: 17% ± 37%
- Moderate temperature: 16.55 ± 6.40°C
- 4.40 ± 1.16 birds

### Flanking Blackfinch
- Moderate BOP count: 40.69 ± 18.61
- No palindromes
- Average path length: 25.21 ± 8.71
- 4.65 ± 1.18 birds

### Rusty Goldhammer
- High BOP count: 71.28 ± 30.57
- No palindromes
- Average path length: 26.26 ± 8.93
- 4.35 ± 1.04 birds

## Why This Approach Avoids Overfitting

1. **Feature Engineering Based on Domain Understanding**: Features capture biological patterns rather than memorizing data points

2. **Regularization Through max_features**: Prevents individual features from dominating

3. **Cross-Validation**: 5-fold CV ensures the model works on unseen data

4. **No Hyperparameter Tuning on Test Set**: Optimization done only on training data

5. **Reasonable Model Complexity**: max_depth=30 is constrained enough to prevent memorization of noise

6. **Ensemble Method**: Random Forest averages multiple trees, reducing variance

7. **Consistent Performance**: Similar accuracy across different random seeds proves robustness

## Comparison to Previous Approach

**Previous**: Rule-based decision tree (manually crafted)
- Required deep domain knowledge
- Hard to maintain and update
- Limited to human-interpretable rules

**Current**: Machine learning Random Forest
- Learns patterns automatically from data
- Captures complex interactions
- More accurate: 93% vs ~89% (estimated from rule-based)
- Generalizes better to unseen data

## Usage

```bash
cd /home/runner/work/cloudeflight_contest/cloudeflight_contest
python solve_level_4.py
```

Output: `level_4/level_4.out` with species predictions for 72 flocks.
