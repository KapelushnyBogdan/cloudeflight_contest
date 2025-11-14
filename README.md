# Cloudeflight Coding Contest Solution

This repository contains solutions for a multi-level bird observation data analysis challenge from Cloudeflight Coding Contest. The challenge involves processing Bird Observation Point (BOP) data, analyzing bird behavior patterns, and making predictions using machine learning.

## 📋 Challenge Overview

The contest consists of 5 progressive levels, each building on insights from previous levels:

- **Level 1**: Sort Bird Observation Points by environmental conditions
- **Level 2**: Predict missing Bird Love Scores using temperature data
- **Level 3**: Classify bird species based on flight path patterns
- **Level 4**: Advanced species classification using machine learning
- **Level 5**: Predict top 50 BOPs by bird arrivals for future days

## 🏗️ Project Structure

```
.
├── README.md                    # This file
├── LEVEL_4_IMPROVEMENTS.md      # Detailed documentation of Level 4 approach
├── level_1/                     # Level 1 input/output files
├── level_2/                     # Level 2 input/output files
├── level_3/                     # Level 3 input/output files
├── level_4/                     # Level 4 input/output files
├── level_5/                     # Level 5 input/output files
├── solve_level_1.py             # Level 1 solution
├── solve_level_2.py             # Level 2 solution
├── solve_level_3.py             # Level 3 solution
├── solve_level_4.py             # Level 4 solution
├── solve_level_5.py             # Level 5 solution
├── level_1.pdf                  # Level 1 problem statement
├── level_2.pdf                  # Level 2 problem statement
├── level_3.pdf                  # Level 3 problem statement
├── level_4.pdf                  # Level 4 problem statement
└── level_5.pdf                  # Level 5 problem statement
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/KapelushnyBogdan/cloudeflight_contest.git
cd cloudeflight_contest
```

2. Install required Python packages:
```bash
pip install pandas numpy scikit-learn
```

### Running Solutions

Each level can be run independently:

```bash
# Level 1: BOP Sorting
python solve_level_1.py

# Level 2: Bird Love Score Prediction
python solve_level_2.py

# Level 3: Species Classification (Rule-based)
python solve_level_3.py

# Level 4: Species Classification (Machine Learning)
python solve_level_4.py

# Level 5: Predict Top 50 BOPs by Arrivals
python solve_level_5.py
```

Each script reads input files from its respective `level_X/` directory and writes output files to the same location.

## 📊 Level Details

### Level 1: BOP Sorting
**Task**: Sort Bird Observation Points by popularity based on temperature and humidity.

**Approach**: 
- Parse CSV files with numeric and text-based numbers
- Sort by: Temperature (descending), Humidity (ascending), BOP ID (ascending)

**Key Features**:
- Handles both numeric strings and word representations of numbers
- Processes multiple input files automatically

### Level 2: Bird Love Score Prediction
**Task**: Predict missing Bird Love Scores using environmental data.

**Approach**:
- Uses temperature data from Level 1 as the primary predictor
- Corrects Fahrenheit temperatures to Celsius (threshold: 42°C)
- Employs ExtraTreesRegressor with polynomial features
- Accumulates training data across datasets for improved accuracy

**Key Features**:
- Temperature correction for mixed unit data
- Polynomial feature engineering (squared and cubic terms)
- Ensemble learning with 1500 trees

### Level 3: Species Classification (Rule-Based)
**Task**: Classify bird species based on flight path patterns.

**Approach**:
- Pattern analysis: palindromes, circular paths, shared prefixes
- Rule-based decision tree using domain-specific patterns
- Temperature analysis from Level 1 data

**Species Patterns**:
- **Medieval Bluetit**: Nearly all paths are palindromic (>95%)
- **Hurracurra Bird**: High temperature (24-32°C), shared starting points
- **Red Firefinch**: Large flocks (6-12 birds), shared prefixes
- **Sticky Wolfthroat**: Short paths (~7 BOPs)
- **Flanking Blackfinch** & **Rusty Goldhammer**: Distinguished by BOP count

### Level 4: Species Classification (Machine Learning)
**Task**: Achieve >90% classification accuracy using machine learning.

**Approach**:
- **20 engineered features** capturing temperature, patterns, and behavior
- **Random Forest Classifier** with 500 trees
- Cross-validation accuracy: **93.04% ± 3.99%**

**Key Features** (top 3 importance):
1. Temperature standard deviation (14.10%)
2. Maximum temperature (10.15%)
3. Palindrome ratio (9.89%)

**Overfitting Prevention**:
- `max_features='sqrt'` for regularization
- 5-fold cross-validation
- Balanced class weights
- Multiple random seed testing

See [LEVEL_4_IMPROVEMENTS.md](LEVEL_4_IMPROVEMENTS.md) for detailed analysis.

### Level 5: Top 50 BOP Prediction
**Task**: Predict the top 50 BOPs by bird arrivals for days 731-760.

**Approach**:
- Uses 730 days of historical arrival data
- Gradient Boosting Regressor with wind and occupancy features
- Feature engineering: wind magnitude, direction, insect deltas
- Predicts arrivals for all 2500 BOPs and selects top 50 per day

**Key Features**:
- Temporal patterns (day of sequence)
- Spatial patterns (BOP-specific behavior)
- Environmental factors (wind, occupancy, insects)

## 📈 Results Summary

| Level | Task | Method | Performance |
|-------|------|--------|-------------|
| 1 | BOP Sorting | Rule-based | 100% accuracy |
| 2 | Score Prediction | ExtraTreesRegressor | RMSE optimized |
| 3 | Species Classification | Rule-based | ~85-90% (estimated) |
| 4 | Species Classification | Random Forest | 93.04% ± 3.99% |
| 5 | Arrival Prediction | Gradient Boosting | Top-K accuracy |

## 🛠️ Technologies Used

- **Python 3**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning library
  - Random Forest Classifier
  - Gradient Boosting (Classifier & Regressor)
  - Extra Trees Regressor
  - Cross-validation

## 📝 Notes

- Input files (`.in`) contain the problem data
- Output files (`.out`) contain the solutions
- Sample files demonstrate expected input/output formats
- PDF files (`level_1.pdf` through `level_5.pdf`) contain the original problem statements

## 🔍 Key Insights

1. **Temperature as a Key Feature**: Temperature (corrected from mixed Fahrenheit/Celsius) is crucial across multiple levels
2. **Pattern Recognition**: Bird behavior shows distinct patterns (palindromes, circular paths) unique to species
3. **Feature Engineering**: Thoughtful feature extraction outperforms complex models with raw data
4. **Ensemble Methods**: Random Forest and Gradient Boosting provide robust predictions across varying conditions
5. **Cross-validation**: Essential for preventing overfitting and ensuring generalization

## 👤 Author

Bogdan Kapelushny

## 📄 License

This project is part of a coding contest submission.
