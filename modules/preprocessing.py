# modules/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, task="grade_category", test_size=0.2, random_state=42):
    """Cleans up student data, defines labels, splits, and scales features."""
    
    # 1. Feature Engineering: Define Pass (1) or Fail (0) based on performance grades
    df['PassStatus'] = np.where(df['FinalGrade'] <= 2, 1, 0)
    
    # Isolate relevant behavioral features
    feature_cols = ['StudyHours', 'Attendance', 'AssignmentCompletion', 'Internet', 'StressLevel', 'Motivation']
    X = df[feature_cols]
    
    # Define target label depending on your chosen project parameter
    if task == "pass_fail":
        y = df['PassStatus']
    elif task == "grade_category":
        y = df['FinalGrade']
    else:
        raise ValueError("Invalid task configuration. Use 'pass_fail' or 'grade_category'.")
        
    # 2. Stratified Train-Test Split to maintain uniform class distributions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 3. Feature Scaling ($z = \frac{x - \mu}{\sigma}$)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X