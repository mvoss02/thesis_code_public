#!/usr/bin/env python
# coding: utf-8

# ## ESG controversy analysis - Modeling
# Import packages
from sklearn.inspection import permutation_importance
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
#from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
import re

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import AdaBoostClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import ADASYN


# ### Modeling
# Import data
os.chdir(
    r"/home/mlvoss0202/"
)

df_merged = pd.read_csv("merged_data.csv", index_col=['id', 'year'])


# Define the columns to be one-hot encoded
categorical_cols = df_merged.select_dtypes(include=['object']).columns.tolist()
categorical_cols = categorical_cols[1:]


# Create empty dataframe that compares output

df_results = pd.read_csv("results_social.csv")
# df_results = pd.DataFrame(columns=[
#                           'Model', 'Parameters', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Best params'])


# **Logistic Regression (perhaps with imputation and regularisation)**


# Split the data into training and testing sets
#df_merged.dropna(axis=0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df_merged.drop(['ISIN Code', 'GICS Industry Group Name', 'country',
                                                                    'Environmental Controversies Count', 'Social Controversies Count',
                                                                    'Governance Controversies Count',
                                                                    'Governance_controversy_binary',
                                                                    'Social_controversy_binary',
                                                                    'Environmental_controversy_binary',
                                                                    'Recent Governance Controversies',
                                                                    'Recent Social Controversies',
                                                                    'Governance_controversy_binary'], axis=1),
                                                    df_merged['Social_controversy_binary'], stratify=df_merged['Social_controversy_binary'], test_size=0.3, random_state=42)


# # Compute class weights for logistic regression
# class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)

# # # One-hot encode vairables
# # # preprocessor = ColumnTransformer(
# # #     transformers=[
# # #         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
# # #     ]
# # # )

# # Create a pipeline with two steps: StandardScaler and LogisticRegression
# pipe = Pipeline([  # ('preprocessor', preprocessor),
#     ('imputer', KNNImputer(metric='nan_euclidean')),
#     #('smote', SMOTE(random_state=42)),
#     ('ada', ADASYN()),
#     ('lr', LogisticRegression(random_state=42, max_iter=5000, solver='saga', class_weight={0: class_weights[0], 1: class_weights[1]}))])
# # Define a param_grid for GridSearchCV that includes the regularization parameter C
# param_grid = {
#     'imputer__n_neighbors': [3, 5, 7],
#     'lr__C': [0.001, 0.01, 1, 10],
#     'lr__penalty': ['l1', 'l2', 'elasticnet']
# }

# # 'imputer__n_neighbors': [3, 5, 7],
# #               'smote__sampling_strategy': ['minority', 'not minority'],
# #               'lr__C': [0.001, 0.1, 1],
# #               'lr__penalty': ['elasticnet', 'l1', 'l2'],


# # Fit the pipeline with GridSearchCV to the training data
# cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# grid_search = GridSearchCV(pipe, param_grid=param_grid,
#                            cv=cv, verbose=1, n_jobs=6, scoring='f1')
# grid_search.fit(X_train, y_train)


# # Use the best estimator from GridSearchCV to predict on the testing data
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_

# # Check paramters of best performing model
# best_params

# # Predict on y test
# y_pred = best_model.predict(X_test)

# # Evaluate the model performance
# print('Accuracy:', accuracy_score(y_test, y_pred))
# print('Precision:', precision_score(y_test, y_pred))
# print('Recall:', recall_score(y_test, y_pred))
# print('F1 score:', f1_score(y_test, y_pred))


# # Append to dataframe
# df_results = df_results.append({'Model': 'LR_ada', 'Accuracy': accuracy_score(y_test, y_pred),
#                                 'Parameters': pipe.named_steps,
#                                 'Precision': precision_score(y_test, y_pred),
#                                 'Recall': recall_score(y_test, y_pred),
#                                 'F1 Score': f1_score(y_test, y_pred),
#                                 'AUC': roc_auc_score(y_test, y_pred),
#                                 'Best params': best_params},
#                                ignore_index=True)

# # print(df_results)

# # write to csv
# df_results.to_csv(r"results_social.csv")

# LR with SMote

# Compute class weights for logistic regression
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)

# One-hot encode vairables
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
#     ]
# )

# Create a pipeline with two steps: StandardScaler and LogisticRegression
pipe = Pipeline([  # ('preprocessor', preprocessor),
    ('imputer', KNNImputer(metric='nan_euclidean')),
    ('smote', SMOTE(random_state=42)),
    #('ada', ADASYN()),
    ('lr', LogisticRegression(random_state=42, max_iter=5000, solver='saga', class_weight={0: class_weights[0], 1: class_weights[1]}))])
# Define a param_grid for GridSearchCV that includes the regularization parameter C
param_grid = {
    'imputer__n_neighbors': [3, 5, 7],
    'smote__sampling_strategy': ['minority'],
    'lr__C': [0.001, 0.01, 1, 10],
    'lr__penalty': ['l1', 'l2', 'elasticnet']
}

# 'imputer__n_neighbors': [3, 5, 7],
#               'smote__sampling_strategy': ['minority', 'not minority'],
#               'lr__C': [0.001, 0.1, 1],
#               'lr__penalty': ['elasticnet', 'l1', 'l2'],


# Fit the pipeline with GridSearchCV to the training data
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

grid_search = GridSearchCV(pipe, param_grid=param_grid,
                           cv=cv, verbose=1, n_jobs=6, scoring='f1')
grid_search.fit(X_train, y_train)


# Use the best estimator from GridSearchCV to predict on the testing data
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Check paramters of best performing model
best_params

# Predict on y test
y_pred = best_model.predict(X_test)

# Evaluate the model performance
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 score:', f1_score(y_test, y_pred))


# Append to dataframe
df_results = df_results.append({'Model': 'LR_smote', 'Accuracy': accuracy_score(y_test, y_pred),
                                'Parameters': pipe.named_steps,
                                'Precision': precision_score(y_test, y_pred),
                                'Recall': recall_score(y_test, y_pred),
                                'F1 Score': f1_score(y_test, y_pred),
                                'AUC': roc_auc_score(y_test, y_pred),
                                'Best params': best_params},
                               ignore_index=True)

# print(df_results)

# write to csv
df_results.to_csv(r"results_social.csv")

# # **Light Gradient Boosting**


# df_merged_gbm = df_merged.rename(
#     columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
# df_merged_gbm.dropna(inplace=True)


# # Define the columns to be one-hot encoded
# categorical_cols_gbm = df_merged_gbm.select_dtypes(
#     include=['object']).columns.tolist()
# categorical_cols_gbm = categorical_cols[:]


# X_train_gbm, X_test_gbm, y_train_gbm, y_test_gbm = train_test_split(df_merged_gbm.drop(['ISINCode', 'GICSIndustryGroupName', 'country',
#                                                                     'EnvironmentalControversiesCount', 'SocialControversiesCount',
#                                                                                         'GovernanceControversiesCount',
#                                                                                         'Governance_controversy_binary',
#                                                                                         'Environmental_controversy_binary',
#                                                                                         'Governance_controversy_binary',
#                                                                                         'RecentGovernanceControversies',
#                                                                                         'RecentSocialControversies',
#                                                                                         'Social_controversy_binary'], axis=1),
#                                                                     df_merged_gbm['Social_controversy_binary'], stratify=df_merged_gbm['Social_controversy_binary'], test_size=0.3, random_state=42)


# #('r', SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))),
# # Compute class weights for logistic regression
# class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train_gbm)

# # # One-hot encode vairables
# # preprocessor = ColumnTransformer(
# #     transformers=[
# #         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols_gbm)
# #     ]
# # )


# # Create a pipeline with two steps: StandardScaler and LogisticRegression
# pipe = Pipeline([  # ('scaler', StandardScaler()),
#                 ('imputer', KNNImputer(metric='nan_euclidean')),
#                 ('smote', SMOTE(random_state=42)),
#                 # ('preprocessor', preprocessor), # without preprocessing much higher dont know why!!
#                 # ('resample', TomekLinks(sampling_strategy='majority')), #SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))),
#                 # gives quite a balanced result - p .25, recall .26
#                 #('ada', ADASYN()),
#                 ('classifier', LGBMClassifier(random_state=42))])  # class_weight={0: class_weights[0], 1: class_weights[1]}

# # Define a param_grid for GridSearchCV that includes the regularization parameter C
# param_grid = {
#     'classifier__learning_rate': [0.1, 0.001, 0.01, 1, 0.002],
#     'classifier__max_depth': [3, 5, 7, 10, 20],
#     'classifier__n_estimators': [100, 300, 500, 700],
#     'imputer__n_neighbors': [3, 5, 7]
# }


# # Fit the pipeline with GridSearchCV to the training data
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# grid_search = GridSearchCV(pipe, param_grid=param_grid,
#                            cv=cv, verbose=1, n_jobs=6, scoring='f1')
# grid_search.fit(X_train_gbm, y_train_gbm)


# # Use the best estimator from GridSearchCV to predict on the testing data
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_


# best_params


# # Predict on y test
# y_pred_gbm = best_model.predict(X_test_gbm)

# # Evaluate the model performance
# print('Accuracy:', accuracy_score(y_test_gbm, y_pred_gbm))
# print('Precision:', precision_score(y_test_gbm, y_pred_gbm))
# print('Recall:', recall_score(y_test_gbm, y_pred_gbm))
# print('F1 score:', f1_score(y_test_gbm, y_pred_gbm))


# # Append to dataframe
# df_results = df_results.append({'Model': 'GBM_ada', 'Accuracy': accuracy_score(y_test_gbm, y_pred_gbm),
#                                 'Parameters': pipe.named_steps,
#                                 'Precision': precision_score(y_test_gbm, y_pred_gbm),
#                                 'Recall': recall_score(y_test_gbm, y_pred_gbm),
#                                 'F1 Score': f1_score(y_test_gbm, y_pred_gbm),
#                                 'AUC': roc_auc_score(y_test_gbm, y_pred_gbm),
#                                 'Best params': best_params},
#                                ignore_index=True)

# print(df_results)

# # write to csv
# df_results.to_csv(r"results_social.csv")


# # Compute class weights for logistic regression
# class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train_gbm)

# # # One-hot encode vairables
# # preprocessor = ColumnTransformer(
# #     transformers=[
# #         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols_gbm)
# #     ]
# # )


# # Create a pipeline with two steps: StandardScaler and LogisticRegression
# pipe = Pipeline([  # ('scaler', StandardScaler()),
#                 ('imputer', KNNImputer(metric='nan_euclidean')),
#                 ('smote', SMOTE(random_state=42)),
#                 # ('preprocessor', preprocessor), # without preprocessing much higher dont know why!!
#                 # ('resample', TomekLinks(sampling_strategy='majority')), #SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))),
#                 # ('ada', ADASYN()), #gives quite a balanced result - p .25, recall .26
#                 ('classifier', LGBMClassifier(random_state=42))])  # class_weight={0: class_weights[0], 1: class_weights[1]}

# # Define a param_grid for GridSearchCV that includes the regularization parameter C
# param_grid = {
#     'classifier__learning_rate': [0.1, 0.001, 0.01, 1, 0.002],
#     'smote__sampling_strategy': ['minority', 'not minority'],
#     'classifier__max_depth': [3, 5, 7, 10, 20],
#     'classifier__n_estimators': [100, 300, 500, 700],
#     'imputer__n_neighbors': [3, 5, 7]
# }


# # Fit the pipeline with GridSearchCV to the training data
# cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# grid_search = GridSearchCV(pipe, param_grid=param_grid,
#                            cv=cv, verbose=1, n_jobs=6, scoring='f1')
# grid_search.fit(X_train_gbm, y_train_gbm)


# # Use the best estimator from GridSearchCV to predict on the testing data
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_


# best_params


# # Predict on y test
# y_pred_gbm = best_model.predict(X_test_gbm)

# # Evaluate the model performance
# print('Accuracy:', accuracy_score(y_test_gbm, y_pred_gbm))
# print('Precision:', precision_score(y_test_gbm, y_pred_gbm))
# print('Recall:', recall_score(y_test_gbm, y_pred_gbm))
# print('F1 score:', f1_score(y_test_gbm, y_pred_gbm))


# # Append to dataframe
# df_results = df_results.append({'Model': 'GBM_smote', 'Accuracy': accuracy_score(y_test_gbm, y_pred_gbm),
#                                 'Parameters': pipe.named_steps,
#                                 'Precision': precision_score(y_test_gbm, y_pred_gbm),
#                                 'Recall': recall_score(y_test_gbm, y_pred_gbm),
#                                 'F1 Score': f1_score(y_test_gbm, y_pred_gbm),
#                                 'AUC': roc_auc_score(y_test_gbm, y_pred_gbm),
#                                 'Best params': best_params},
#                                ignore_index=True)

# # print(df_results)

# # write to csv
# df_results.to_csv(r"results_social.csv")


# # Create a pipeline with two steps: StandardScaler and LogisticRegression
# pipe = Pipeline([  # ('scaler', StandardScaler()),
#                 #('imputer', KNNImputer(metric='nan_euclidean')),
#                 #('smote', SMOTE(random_state=42)),
#                 # ('preprocessor', preprocessor), # without preprocessing much higher dont know why!!
#                 # ('resample', TomekLinks(sampling_strategy='majority')), #SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))),
#                 # ('ada', ADASYN()), #gives quite a balanced result - p .25, recall .26
#                 ('classifier', LGBMClassifier(random_state=42))])  # class_weight={0: class_weights[0], 1: class_weights[1]}

# # Define a param_grid for GridSearchCV that includes the regularization parameter C
# param_grid = {
#     'classifier__learning_rate': [0.1, 0.001, 0.01, 1, 0.002],
#     # 'smote__sampling_strategy': ['minority', 'not minority'],
#     'classifier__max_depth': [3, 5, 7, 10, 20],
#     'classifier__n_estimators': [100, 300, 500, 700],
#     # 'imputer__n_neighbors': [3, 5, 7]
# }


# # Fit the pipeline with GridSearchCV to the training data
# cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# grid_search = GridSearchCV(pipe, param_grid=param_grid,
#                            cv=cv, verbose=1, n_jobs=6, scoring='f1')
# grid_search.fit(X_train_gbm, y_train_gbm)


# # Use the best estimator from GridSearchCV to predict on the testing data
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_


# best_params


# # Predict on y test
# y_pred_gbm = best_model.predict(X_test_gbm)

# # Evaluate the model performance
# print('Accuracy:', accuracy_score(y_test_gbm, y_pred_gbm))
# print('Precision:', precision_score(y_test_gbm, y_pred_gbm))
# print('Recall:', recall_score(y_test_gbm, y_pred_gbm))
# print('F1 score:', f1_score(y_test_gbm, y_pred_gbm))


# # Append to dataframe
# df_results = df_results.append({'Model': 'GBM_smote', 'Accuracy': accuracy_score(y_test_gbm, y_pred_gbm),
#                                 'Parameters': pipe.named_steps,
#                                 'Precision': precision_score(y_test_gbm, y_pred_gbm),
#                                 'Recall': recall_score(y_test_gbm, y_pred_gbm),
#                                 'F1 Score': f1_score(y_test_gbm, y_pred_gbm),
#                                 'AUC': roc_auc_score(y_test_gbm, y_pred_gbm),
#                                 'Best params': best_params},
#                                ignore_index=True)

# # print(df_results)

# # write to csv
# df_results.to_csv(r"results_social.csv")

# # **Quadratic Discriminant Analysis**


# # # One-hot encode vairables
# # preprocessor = ColumnTransformer(
# #     transformers=[
# #         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
# #     ]
# # )

# pipeline = Pipeline([  # ('preprocessor', preprocessor),
#     ('imputer', KNNImputer(metric='nan_euclidean')),
#     ('ada', ADASYN()),
#     ('qda', QuadraticDiscriminantAnalysis())])

# param_grid = {
#     'imputer__n_neighbors': [3, 5, 7],
#     'qda__reg_param': list(np.logspace(-40.0, 3.0, 40))
# }

# cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# grid_search = GridSearchCV(
#     pipeline, param_grid=param_grid, cv=cv, n_jobs=6, verbose=1)
# grid_search.fit(X_train, y_train)


# # Use the best estimator from GridSearchCV to predict on the testing data
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_


# best_params


# # Predict on y test
# y_pred = best_model.predict(X_test)

# # Evaluate the model performance
# print('Accuracy:', accuracy_score(y_test, y_pred))
# print('Precision:', precision_score(y_test, y_pred))
# print('Recall:', recall_score(y_test, y_pred))
# print('F1 score:', f1_score(y_test, y_pred))


# # Append to dataframe
# df_results = df_results.append({'Model': 'QDA_ada', 'Accuracy': accuracy_score(y_test, y_pred),
#                                 'Parameters': pipeline.named_steps,
#                                 'Precision': precision_score(y_test, y_pred),
#                                 'Recall': recall_score(y_test, y_pred),
#                                 'F1 Score': f1_score(y_test, y_pred),
#                                 'AUC': roc_auc_score(y_test, y_pred),
#                                 'Best params': best_params},
#                                ignore_index=True)

# # print(df_results)

# # write to csv
# df_results.to_csv(r"results_social.csv")


# # Print the best parameters and score
# print("Best parameters: ", grid_search.best_params_)
# print("Train score: ", grid_search.best_score_)
# print("Test score: ", grid_search.score(X_test, y_test))


# # **Quadratic Discriminant Analysis**


# # # One-hot encode vairables
# # preprocessor = ColumnTransformer(
# #     transformers=[
# #         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
# #     ]
# # )

# pipeline = Pipeline([  # ('preprocessor', preprocessor),
#     ('imputer', KNNImputer(metric='nan_euclidean')),
#     ('smote', SMOTE(random_state=42)),
#     #('ada', ADASYN()),
#     ('qda', QuadraticDiscriminantAnalysis())])

# param_grid = {
#     'imputer__n_neighbors': [3, 5, 7],
#     'smote__sampling_strategy': ['minority', 'not minority'],
#     'qda__reg_param': list(np.logspace(-40.0, 3.0, 40))
# }

# cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# grid_search = GridSearchCV(
#     pipeline, param_grid=param_grid, cv=cv, n_jobs=6, verbose=1)
# grid_search.fit(X_train, y_train)


# # Use the best estimator from GridSearchCV to predict on the testing data
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_


# best_params


# # Predict on y test
# y_pred = best_model.predict(X_test)

# # Evaluate the model performance
# print('Accuracy:', accuracy_score(y_test, y_pred))
# print('Precision:', precision_score(y_test, y_pred))
# print('Recall:', recall_score(y_test, y_pred))
# print('F1 score:', f1_score(y_test, y_pred))


# # Append to dataframe
# df_results = df_results.append({'Model': 'QDA_smote', 'Accuracy': accuracy_score(y_test, y_pred),
#                                 'Parameters': pipeline.named_steps,
#                                 'Precision': precision_score(y_test, y_pred),
#                                 'Recall': recall_score(y_test, y_pred),
#                                 'F1 Score': f1_score(y_test, y_pred),
#                                 'AUC': roc_auc_score(y_test, y_pred),
#                                 'Best params': best_params},
#                                ignore_index=True)

# # print(df_results)

# # write to csv
# df_results.to_csv(r"results_social.csv")


# # Print the best parameters and score
# print("Best parameters: ", grid_search.best_params_)
# print("Train score: ", grid_search.best_score_)
# print("Test score: ", grid_search.score(X_test, y_test))
