import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Step 1: Load the Data
data = pd.read_csv('path/to/svmTuningData.dat', delimiter='\t')
X = data.iloc[:, :-1].values  # Assuming the last column is the label
y = data.iloc[:, -1].values

# Step 2: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define the Parameter Grid
param_grid = {'C': [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100],
              'gamma': [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100]}

# Step 4: Create the SVM Model
svm_model = SVC(kernel='rbf')

# Step 5: Grid Search
grid_search = GridSearchCV(svm_model, param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

# Step 6: Get Best Parameters and Accuracy
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Accuracy:", best_accuracy)

# Step 7: Report Results
with open('README.md', 'w') as readme_file:
    readme_file.write(f"Optimal Parameters: {best_params}\n")
    readme_file.write(f"Estimated Accuracy: {best_accuracy}\n")
