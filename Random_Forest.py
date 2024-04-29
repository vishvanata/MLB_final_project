import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance

# PART 1: HYPERPARAMETER TUNING

# Load the one-hot encoded, preprocessed data
df = pd.read_csv('NHIS_onehot_data.csv')
# Selecting the target variable
X = df.drop('LSATIS4_A', axis=1)
y = df['LSATIS4_A']
# Splitting the dataset into test and train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'max_features': ['auto', 'sqrt', 'log2', 0.3, 0.5]
}
# Defining the random forest classifier for grid search
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
# Outputting the selected parameter set and its score
print("Best parameters:", grid_search.best_params_)
# Evaluating the chosen parameters on the test dataset
y_pred = grid_search.predict(X_test)
print('Test accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
# Storing accuracy values for all combinations of the first two parameters
scores_matrix = grid_search.cv_results_['mean_test_score'].reshape(len(param_grid['max_depth']), len(param_grid['n_estimators']), len(param_grid['min_samples_split']), len(param_grid['min_samples_leaf']), len(param_grid['max_features']))
# Slicing out a 2D matrix of scores
scores = scores_matrix[:, :, 0, 0, 1]
# Heatmap for 'n_estimators' and 'max_depth'
plt.figure(figsize=(10, 8))
sns.heatmap(scores, annot=True, fmt=".2f", xticklabels=param_grid['n_estimators'], yticklabels=param_grid['max_depth'])
plt.xlabel('Number of Estimators')
plt.ylabel('Max Depth')
plt.title('Grid Search Scores for n_estimators and max_depth')
plt.show()


# Defining the parameter distributions for Random Search
param_values = {
    'n_estimators': randint(200, 1200),
    'max_depth': [None] + list(range(10, 110, 10)),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 0.1, 0.2, 0.3, 0.4, 0.5]
}
# Defining the random forest classifier for random search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42), 
    param_distributions=param_values, 
    n_iter=100,
    cv=3, 
    verbose=1, 
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
# Outputting the selected parameter set and its score
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score: {:.2f}".format(random_search.best_score_))

# Evaluating the chosen parameters on the test dataset
y_pred = random_search.predict(X_test)
print('Test accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))

n_estimators_range = [estimator for estimator in param_values['n_estimators'].rvs(size=100)]
max_depth_range = param_values['max_depth']
# Adjusting dimensions
scores_matrix = random_search.cv_results_['mean_test_score'].reshape(10, 10)

# Plotting heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(scores_matrix, annot=True, fmt=".2f", xticklabels=n_estimators_range[:10], yticklabels=max_depth_range[:10])
plt.xlabel('Number of Estimators')
plt.ylabel('Max Depth')
plt.title('Random Search Scores for n_estimators and max_depth')
plt.show()


# PART 2: MODEL EVALUATION

# Testing the model with the final parameter set
rf_model = RandomForestClassifier(
    n_estimators=1000,
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=None,
    max_features='sqrt',
    random_state=42 
)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
# Printing the classification report
print(classification_report(y_test, y_pred, zero_division=1))
# Calculating the confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Plotting the confusion matrix
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Predicting probabilities for the test set
y_prob = rf_model.predict_proba(X_test)
# Binarizing the predicted labels
y_test_binarized = label_binarize(y_test, classes=np.unique(y))
# Computing the ROC curves and areas for each class
fpr = dict()
tpr = dict()
n_classes = y.nunique()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Ploting the ROC curves
plt.figure(figsize=(8, 6))
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.show()


# PART 3: FEATURE IMPORTANCE ANALYSIS


# Getting built-in feature importances through the RF model
importances = rf_model.feature_importances_
feature_importances = pd.DataFrame(importances, index=X_train.columns, columns=["Importance"])
# Sortting the features by importance score
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)
# Plotting the features
plt.figure(figsize=(10, 8))
palette = sns.color_palette("husl", len(feature_importances))
bars = plt.barh(feature_importances.index[:20], feature_importances["Importance"][:20], color=palette[:20])
plt.gca().invert_yaxis()
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout() 
plt.show()

# Getting permutation feature importances
results = permutation_importance(rf_model, X_test, y_test, n_repeats=10,
                                 random_state=42, n_jobs=-1)
perm_importances = pd.DataFrame(results.importances_mean, index=X_train.columns,
                                columns=["importance"]).sort_values("importance", ascending=False)

# Selecting the top 20 features
top_features = perm_importances.head(20)
palette = sns.color_palette("crest", as_cmap=False, n_colors=len(top_features))
palette.reverse()
# Plotting the features
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y=top_features.index, data=top_features, palette=palette)
plt.title('Permutation Feature Importances (Top 20)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()