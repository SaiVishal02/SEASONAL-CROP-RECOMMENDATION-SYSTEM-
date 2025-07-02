import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import pickle
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv('crop_recommendation.csv')

# Define features and target
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

# Split data into training and testing sets 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=2)

# Initialize lists to store model names and accuracies
model_names = []
accuracies = []

# Decision Tree Classifier
DecisionTree = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
DecisionTree.fit(Xtrain, Ytrain)
predicted_values = DecisionTree.predict(Xtest)
accuracy = accuracy_score(Ytest, predicted_values)
print("Decision Tree's Accuracy is: ", accuracy * 100)
print(classification_report(Ytest, predicted_values))
model_names.append('Decision Tree')
accuracies.append(accuracy)

# Cross-validation score for Decision Tree
score = cross_val_score(DecisionTree, features, target, cv=5)
print(score)

# Save Decision Tree model
DT_pkl_filename = 'DecisionTree.pkl'
DT_Model_pkl = open(DT_pkl_filename, 'wb')
pickle.dump(DecisionTree, DT_Model_pkl)
DT_Model_pkl.close()

# Random Forest Classifier
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain, Ytrain)
predicted_values = RF.predict(Xtest)
accuracy = accuracy_score(Ytest, predicted_values)
print("Random Forest's Accuracy is: ", accuracy * 100)
print(classification_report(Ytest, predicted_values))
model_names.append('Random Forest')
accuracies.append(accuracy)

# Cross-validation score for Random Forest
score = cross_val_score(RF, features, target, cv=5)
print(score)

# Save Random Forest model
RF_pkl_filename = 'RandomForestClassifier.pkl'
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
RF_Model_pkl.close()

# Logistic Regression
LogReg = LogisticRegression(random_state=2)
LogReg.fit(Xtrain, Ytrain)
predicted_values = LogReg.predict(Xtest)
accuracy = accuracy_score(Ytest, predicted_values)
print("Logistic Regression's Accuracy is: ", accuracy * 100)
print(classification_report(Ytest, predicted_values))
model_names.append('Logistic Regression')
accuracies.append(accuracy)

# Cross-validation score for Logistic Regression
score = cross_val_score(LogReg, features, target, cv=5)
print(score)

# Save Logistic Regression model
LR_pkl_filename = 'LogisticRegression.pkl'
LR_Model_pkl = open(LR_pkl_filename, 'wb')
pickle.dump(LogReg, LR_Model_pkl)
LR_Model_pkl.close()

# Plot accuracy comparison
plt.figure(figsize=[10, 5], dpi=100)
plt.title('Accuracy Comparison')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.show()

# Make a prediction using the best model (Random Forest in this case)
data = np.array([[104, 18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = RF.predict(data)
print("Predicted crop:", prediction)
