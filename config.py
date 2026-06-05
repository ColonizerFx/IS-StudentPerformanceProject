# config.py

DATA_PATH = "data/student_performance.csv"

# Toggle between tasks required by your brief: 'pass_fail' or 'grade_category'
CLASSIFICATION_TASK = "grade_category"  

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Experimental hyperparameter settings for your project's evaluation section
NB_VAR_SMOOTHING = [1e-9, 1e-8, 1e-7]
KMEANS_CLUSTERS = [2, 3, 4, 5]