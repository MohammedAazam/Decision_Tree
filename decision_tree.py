# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split

# Simple dataset: Experience vs Salary
data = {
    'YearsExperience': [1, 2, 3, 4, 5, 6],
    'Salary': [35000, 40000, 50000, 60000, 65000, 70000]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['YearsExperience']]
y = df['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Decision Tree Regressor with max depth of 2 for better visual understanding
model = DecisionTreeRegressor(max_depth=2, random_state=1)
model.fit(X_train, y_train)

# Plotting the Decision Tree clearly
plt.figure(figsize=(10,6))
plot_tree(model, 
          feature_names=['YearsExperience'], 
          filled=True, 
          rounded=True, 
          precision=0, 
          fontsize=12)
plt.title("Simple Decision Tree for Salary Prediction")
plt.show()

# Predict across smooth range
X_range = np.arange(min(X['YearsExperience']), max(X['YearsExperience'])+0.1, 0.1).reshape(-1, 1)
y_pred_range = model.predict(X_range)

# Plotting predictions
plt.figure(figsize=(10,5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_range, y_pred_range, color='red', label='Decision Tree Prediction')
plt.xlabel("Years of Experience")
plt.ylabel("Predicted Salary")
plt.title("Decision Tree Regression Fit")
plt.legend()
plt.grid(True)
plt.show()
