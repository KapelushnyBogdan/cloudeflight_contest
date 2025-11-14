"""
Level 5: Predict Top 50 BOPs by Bird Arrivals

This module predicts the top 50 Bird Observation Points by arrivals for days 731-760
using historical data from days 1-730. Employs gradient boosting regression with
comprehensive feature engineering.

Strategy:
    1. Load 730 days of training data with known arrivals
    2. Engineer features from occupancy, wind, insects, BOP ID, and day
    3. Train Gradient Boosting Regressor to predict arrivals
    4. For each prediction day (731-760):
       - Predict arrivals for all 2500 BOPs
       - Sort by predicted arrivals (descending)
       - Select top 50 BOPs

Features:
    - Environmental: Occupancy, Wind (X, Y, magnitude, direction), Insects Delta
    - Spatial: BOP ID (location-specific patterns)
    - Temporal: Day number (seasonal patterns)
    - Derived: Wind magnitude, wind direction angle

Model: Gradient Boosting Regressor optimized for arrival prediction

Usage:
    python solve_level_5.py

Input:
    - level_5/level_5.in with BOP, Day, Occupancy, Wind, Insects, and Arrivals columns

Output:
    - level_5/level_5.out with top 50 BOPs per day for days 731-760
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def solve_level_5():
    """
    Main function to predict top 50 BOPs by arrivals for days 731-760.
    
    Implements a complete pipeline:
    1. Data loading and preprocessing
    2. Feature engineering (wind magnitude/direction, etc.)
    3. Model training with Gradient Boosting
    4. Prediction for all BOPs on days 731-760
    5. Top-50 selection per day
    6. Output generation
    
    Returns:
        None: Writes predictions to level_5/level_5.out
    """
    
    print("="*70)
    print("LEVEL 5: PREDICTING TOP 50 BOPs BY ARRIVALS (DAYS 731-760)")
    print("="*70)
    
    # Load data
    print("\n[1] Loading data...")
    df = pd.read_csv('level_5/level_5.in')
    print(f"    Total records: {len(df)}")
    
    # Separate training and prediction data
    train_df = df[df['Arrivals'] != 'missing'].copy()
    predict_df = df[df['Arrivals'] == 'missing'].copy()
    
    train_df['Arrivals'] = train_df['Arrivals'].astype(float)
    
    print(f"    Training records: {len(train_df)}")
    print(f"    Prediction records: {len(predict_df)}")
    
    # Feature engineering
    print("\n[2] Feature engineering...")
    
    def add_features(df):
        """Add derived features"""
        # Wind magnitude and direction
        df['Wind_Magnitude'] = np.sqrt(df['Wind X [m/s]']**2 + df['Wind Y [m/s]']**2)
        df['Wind_Direction'] = np.arctan2(df['Wind Y [m/s]'], df['Wind X [m/s]'])
        
        # Interaction features
        df['Occupancy_Insects'] = df['Occupancy'] * df['Insects Delta [g/m²]']
        df['Wind_Insects'] = df['Wind_Magnitude'] * df['Insects Delta [g/m²]']
        
        # Polynomial features for key variables
        df['Occupancy_Squared'] = df['Occupancy'] ** 2
        df['Insects_Squared'] = df['Insects Delta [g/m²]'] ** 2
        
        return df
    
    train_df = add_features(train_df)
    predict_df = add_features(predict_df)
    
    # Define features
    feature_cols = [
        'BOP', 'Day', 'Occupancy', 
        'Wind X [m/s]', 'Wind Y [m/s]', 'Wind_Magnitude', 'Wind_Direction',
        'Insects Delta [g/m²]',
        'Occupancy_Insects', 'Wind_Insects',
        'Occupancy_Squared', 'Insects_Squared'
    ]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['Arrivals'].values
    
    # Train fast model - just Random Forest for speed
    print("\n[3] Training Random Forest model (fast)...")
    
    # Random Forest - fast and accurate enough for 60%
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    print("    Training on 1.8M samples...")
    model.fit(X_train, y_train)
    print("    ✓ Random Forest trained")
    
    print("\n[4] Model training complete")
    
    # Make predictions for days 731-760
    print("\n[5] Predicting top 50 BOPs for days 731-760...")
    
    results = []
    prediction_days = sorted(predict_df['Day'].unique())
    
    for day in prediction_days:
        day_data = predict_df[predict_df['Day'] == day].copy()
        X_pred = day_data[feature_cols].values
        
        # Single model prediction
        predictions = model.predict(X_pred)
        
        day_data['Predicted_Arrivals'] = predictions
        
        # Sort by predicted arrivals (descending) and get top 50 BOPs
        top_50 = day_data.nlargest(50, 'Predicted_Arrivals')['BOP'].values
        
        # Format output
        top_50_str = ' '.join(map(str, top_50))
        results.append((day, top_50_str))
        
        print(f"    Day {day}: Top BOP = {top_50[0]} (predicted arrivals: {predictions.max():.1f})")
    
    # Write output
    print("\n[6] Writing results to level_5/level_5.out...")
    with open('level_5/level_5.out', 'w') as f:
        f.write("Day,Top 50 Arrivals BOPs\n")
        for day, top_50_str in results:
            f.write(f"{day},{top_50_str}\n")
    
    print("\n" + "="*70)
    print("DONE! Predictions generated for 30 days.")
    print("="*70)
    
    # Show sample of results
    print("\nSample predictions (first 5 days):")
    for day, top_50_str in results[:5]:
        bops = top_50_str.split()[:10]  # Show first 10 BOPs
        print(f"  Day {day}: {' '.join(bops)}...")

if __name__ == "__main__":
    solve_level_5()
