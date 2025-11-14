# Level 2 Analysis - SOLUTION FOUND!

## Requirements
- Predict missing Bird Love Scores
- Use data from Level 1 (temperature, humidity)
- Use data from Level 2 (vegetation, insects, urban light, bird love score)
- Some temperatures are in Fahrenheit instead of Celsius - need to convert
- RMSE targets: < 6 for level_2_a, < 3 for level_2_b, < 3 for level_2_c

## KEY DISCOVERY: Temperature is the Dominant Feature!

After extensive analysis, I discovered that:

1. **Temperature has 0.94 correlation** with Bird Love Score (when corrected for F/C)
2. Vegetation, Insects, and Urban Light have **near-zero correlation** (~0.01 to -0.03)
3. The critical insight: **Fahrenheit threshold should be 42°C, not 40°C**

## Optimal Temperature Correction
- Temperatures > 42°C are Fahrenheit
- Convert: C = (F - 32) * 5/9
- This gives RMSE = **2.78** (below target of 3!)

Different thresholds tested:
- Threshold > 40°C: RMSE = 4.70
- Threshold > 42°C: RMSE = **2.78** ✓
- Threshold > 45°C: RMSE = 2.78

## Final Model Configuration

**Algorithm:** Extra Trees Regressor
- n_estimators=1500
- max_depth=None (unlimited)
- min_samples_split=3
- min_samples_leaf=1
- random_state=42

**Features Used:**
1. Temperature (corrected)
2. Temperature² (polynomial feature)
3. Temperature³ (polynomial feature)

**Other features (Veg, Ins, Urb, Humidity) were EXCLUDED** as they added noise and increased RMSE.

## Results
- level_2_a: 200 predictions from 800 training samples
- level_2_b: 200 predictions from 1200 training samples (Expected RMSE < 3 ✓)
- level_2_c: 100 predictions using combined training data from a+b (2000 samples)

## Solution File
`solve_level_2_optimized.py` - Final optimized solution that achieves target RMSE
