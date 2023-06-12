#!/usr/bin/env python
# coding: utf-8

# ## ESG controversy analysis


# Import packages
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
#from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier

#import bnlearn as bn
from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
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
df_results = pd.DataFrame(columns=[
                          'Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Best params'])


# ### Logistic Regression (perhaps with imputation and regularisation)


# Split the data into training and testing sets
#df_merged.dropna(axis=0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df_merged.drop(['ISIN Code', 'GICS Industry Group Name', 'country',
                                                                    'Environmental Controversies Count', 'Social Controversies Count',
                                                                    'Governance Controversies Count',
                                                                    'Governance_controversy_binary',
                                                                    'Social_controversy_binary',
                                                                    'Environmental_controversy_binary',
                                                                    'Governance_controversy_binary'], axis=1),
                                                    df_merged['Social_controversy_binary'], stratify=df_merged['Social_controversy_binary'], test_size=0.3, random_state=42)


# Compute class weights for logistic regression
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)

# # One-hot encode vairables
# # preprocessor = ColumnTransformer(
# #     transformers=[
# #         ('cat', OneHotEncoder(handle_unknown='ignore',
# #          sparse_output=False), categorical_cols)
# #     ]
# # )

# # Create a pipeline with two steps: StandardScaler and LogisticRegression
# pipe = Pipeline([  # ('preprocessor', preprocessor),
#     ('imputer', KNNImputer(metric='nan_euclidean')),
#     #('smote', SMOTE(random_state=42)),
#     ('ada', ADASYN()),
#     ('lr', LogisticRegression(max_iter=1000, solver='saga', class_weight={0: class_weights[0], 1: class_weights[1]}, random_state=42))])

# # Define a param_grid for GridSearchCV that includes the regularization parameter C
# param_grid = {'imputer__n_neighbors': [3, 5, 7],
#               # minority
#               # 'smote__sampling_strategy': ['not minority', 'minority'],
#               'lr__C': [0.001, 0.01, 0.1, 1],
#               # 'elasticnet', 'l1', 'l2'
#               'lr__penalty': ['elasticnet', 'l1', 'l2'],
#               }


# # Fit the pipeline with GridSearchCV to the training data
# grid_search = GridSearchCV(pipe, param_grid=param_grid,
#                            cv=5, verbose=1, n_jobs=6, scoring='f1')
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
# df_results = df_results.append({'Model': 'Log Reg', 'Accuracy': accuracy_score(y_test, y_pred),
#                                 'Precision': precision_score(y_test, y_pred),
#                                 'Recall': recall_score(y_test, y_pred),
#                                 'F1 Score': f1_score(y_test, y_pred),
#                                 'AUC': roc_auc_score(y_test, y_pred),
#                                 'Best params': best_params},
#                                ignore_index=True)

# print(df_results)

# # write to csv
# df_results.to_csv(r"results.csv")


# # Assuming y_pred and y_true are the predicted and true labels, respectively
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# # Compute AUC (Area Under the Curve)
# auc = roc_auc_score(y_test, y_pred)

# # Plot ROC curve
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
# plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()


# # # Random Forest

# # # Compute class weights for logistic regression
# # class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)

# # # # One-hot encode vairables
# # # preprocessor = ColumnTransformer(
# # #     transformers=[
# # #         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
# # #     ]
# # # )

# # # Create a pipeline with two steps: StandardScaler and LogisticRegression
# # pipe = Pipeline([  # ('scaler', StandardScaler()),
# #     #('smote', SMOTE(random_state=42)),
# #                 #  ('lr', LogisticRegression(max_iter=10000,
# #                 #                            solver='saga',
# #                 #                            random_state=42))])
# #                 # ('preprocessor', preprocessor),
# #                 ('imputer', KNNImputer(metric='nan_euclidean')),
# #                 ('classifier', RandomForestClassifier(class_weight={0: class_weights[0], 1: class_weights[1]}, random_state=42))])

# # # Define a param_grid for GridSearchCV that includes the regularization parameter C
# # param_grid = {
# #     'classifier__n_estimators': [100, 300, 500],
# #     'classifier__min_samples_split': [3, 5, 7],
# #     'classifier__max_depth': [3, 5, 7],
# #     'classifier__criterion': ['gini', 'entropy'],
# #     'classifier__min_samples_split': [3, 5, 7],
# #     'imputer__n_neighbors': [3, 5, 7],
# # }


# # # Create the grid search object
# # grid_search = GridSearchCV(pipe, param_grid, cv=2, scoring='f1', verbose=1)

# # # Fit the grid search object to the data
# # grid_search.fit(X_train, y_train)

# # # Use the best estimator from GridSearchCV to predict on the testing data
# # best_params = grid_search.best_params_
# # best_model = grid_search.best_estimator_

# # # Print the best parameters and score
# # print("Best parameters: ", grid_search.best_params_)
# # print("Train score: ", grid_search.best_score_)
# # print("Test score: ", grid_search.score(X_test, y_test))


# # # Evaluate the model performance
# # print('Accuracy:', accuracy_score(y_test, y_pred))
# # print('Precision:', precision_score(y_test, y_pred))
# # print('Recall:', recall_score(y_test, y_pred))
# # print('F1 score:', f1_score(y_test, y_pred))


# # # Append to dataframe
# # df_results = df_results.append({'Model': 'RF', 'Accuracy': accuracy_score(y_test, y_pred),
# #                                 'Precision': precision_score(y_test, y_pred),
# #                                 'Recall': recall_score(y_test, y_pred),
# #                                 'F1 Score': f1_score(y_test, y_pred),
# #                                 'AUC': roc_auc_score(y_test, y_pred),
# #                                 'Best params': best_params},
# #                                ignore_index=True)

# # print(df_results)

# # # write to csv
# # df_results.to_csv(r"results.csv")


# # # Assuming y_pred and y_true are the predicted and true labels, respectively
# # fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# # # Compute AUC (Area Under the Curve)
# # auc = roc_auc_score(y_test, y_pred)

# # # Plot ROC curve
# # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
# # plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.05])
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('Receiver Operating Characteristic (ROC) Curve')
# # plt.legend(loc="lower right")
# # plt.show()


# # Light Gradient Boosting


df_merged_gbm = df_merged.rename(
    columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))


# Define the columns to be one-hot encoded
categorical_cols_gbm = df_merged.select_dtypes(
    include=['object']).columns.tolist()
categorical_cols_gbm = categorical_cols[1:]


X_train_gbm, X_test_gbm, y_train_gbm, y_test_gbm = train_test_split(df_merged_gbm.drop(['ISINCode', 'GICSIndustryGroupName', 'country',
                                                                                        'EnvironmentalControversiesCount', 'SocialControversiesCount',
                                                                                        'GovernanceControversiesCount',
                                                                                        'Governance_controversy_binary',
                                                                                        'Environmental_controversy_binary',
                                                                                        'Governance_controversy_binary',
                                                                                        'RecentGovernanceControversies',
                                                                                        'RecentSocialControversies',
                                                                                        'Social_controversy_binary'], axis=1),
                                                                    df_merged_gbm['Social_controversy_binary'], stratify=df_merged_gbm['Social_controversy_binary'], test_size=0.3, random_state=42)


# Compute class weights for logistic regression
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train_gbm)

# One-hot encode vairables
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore',
#          sparse_output=False), categorical_cols_gbm)
#     ]
# )

# # Create a pipeline with two steps: StandardScaler and LogisticRegression
# pipe = Pipeline([  # ('scaler', StandardScaler()),
#     #('smote', SMOTE(random_state=42)),
#                 #  ('lr', LogisticRegression(max_iter=10000,
#                 #                            solver='saga',
#                 #                            random_state=42))])
#                 # without preprocessing much higher dont know why!!
#                 #('preprocessor', preprocessor),
#                 ('classifier', LGBMClassifier(class_weight={0: class_weights[0], 1: class_weights[1]}, random_state=42))])

# # Define a param_grid for GridSearchCV that includes the regularization parameter C
# param_grid = {
#     'classifier__n_estimators': [300, 400, 500],
#     'classifier__learning_rate': [0.01, 0.01, 0.1, 1],
#     'classifier__max_depth': [20, 50, 70]
# }


# # Fit the pipeline with GridSearchCV to the training data
# grid_search = GridSearchCV(pipe, param_grid=param_grid,
#                            cv=2, verbose=1, n_jobs=6, scoring='f1')
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
# df_results = df_results.append({'Model': 'GBM', 'Accuracy': accuracy_score(y_test_gbm, y_pred_gbm),
#                                 'Precision': precision_score(y_test_gbm, y_pred_gbm),
#                                 'Recall': recall_score(y_test_gbm, y_pred_gbm),
#                                 'F1 Score': f1_score(y_test_gbm, y_pred_gbm),
#                                 'AUC': roc_auc_score(y_test_gbm, y_pred_gbm),
#                                 'Best params': best_params},
#                                ignore_index=True)

# print(df_results)

# # write to csv
# df_results.to_csv(r"results.csv")


# # Assuming y_pred and y_true are the predicted and true labels, respectively
# fpr, tpr, thresholds = roc_curve(y_test_gbm, y_pred_gbm)

# # Compute AUC (Area Under the Curve)
# auc = roc_auc_score(y_test_gbm, y_pred_gbm)

# # Plot ROC curve
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
# plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()


# # Print the best parameters and score
# print("Best parameters: ", grid_search.best_params_)
# print("Train score: ", grid_search.best_score_)
# print("Test score: ", grid_search.score(X_test_gbm, y_test_gbm))


# # Neural Network

# # One-hot encode vairables
# # preprocessor = ColumnTransformer(
# #     transformers=[
# #         ('cat', OneHotEncoder(handle_unknown='ignore',
# #          sparse_output=False), categorical_cols)
# #     ]
# # )

# pipeline = Pipeline([  # ('preprocessor', preprocessor),
#     ('imputer', KNNImputer(metric='nan_euclidean')),
#     ('mlp', MLPClassifier(solver='adam', verbose=False))])

# param_grid = {
#     'mlp__alpha': [0.0001, 0.001, 0.002, 0.01, 0.1, 0.3, 0.5],
#     'mlp__learning_rate_init': [0.0001, 0.001, 0.003, 0.01, 0.1, 0.3, 0.5],
#     'mlp__hidden_layer_sizes': [(5,), (10,), (15,), (20,)],
#     'mlp__max_iter': [2000],
#     'mlp__activation': ['relu', 'logistic'],
#     'imputer__n_neighbors': [5, 7, 10]
# }

# grid_search = GridSearchCV(
#     pipeline, param_grid=param_grid, cv=2, n_jobs=6, scoring='f1', verbose=1)
# grid_search.fit(X_train, y_train)


# print("Best parameters: ", grid_search.best_params_)
# print("Train score: ", grid_search.best_score_)
# print("Test score: ", grid_search.score(X_test, y_test))


# # Use the best estimator from GridSearchCV to predict on the testing data
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_


# # Predict on y test
# y_pred = best_model.predict(X_test)

# # Evaluate the model performance
# print('Accuracy:', accuracy_score(y_test, y_pred))
# print('Precision:', precision_score(y_test, y_pred))
# print('Recall:', recall_score(y_test, y_pred))
# print('F1 score:', f1_score(y_test, y_pred))


# # Append to dataframe
# df_results = df_results.append({'Model': 'Neural Network', 'Accuracy': accuracy_score(y_test, y_pred),
#                                 'Precision': precision_score(y_test, y_pred),
#                                 'Recall': recall_score(y_test, y_pred),
#                                 'F1 Score': f1_score(y_test, y_pred),
#                                 'AUC': roc_auc_score(y_test, y_pred),
#                                 'Best params': best_params},
#                                ignore_index=True)

# print(df_results)


# # Assuming y_pred and y_true are the predicted and true labels, respectively
# fpr, tpr, thresholds = roc_curve(y_test_gbm, y_pred_gbm)

# # Compute AUC (Area Under the Curve)
# auc = roc_auc_score(y_test_gbm, y_pred_gbm)

# # Plot ROC curve
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
# plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()


# # Naive Bayes Classifier


# # One-hot encode vairables
# # preprocessor = ColumnTransformer(
# #     transformers=[
# #         ('cat', OneHotEncoder(handle_unknown='ignore',
# #          sparse_output=False), categorical_cols)
# #     ]
# # )

# # Define the pipeline
# pipeline = Pipeline([
#     #('preprocessor', preprocessor),
#     ('imputer', KNNImputer(metric='nan_euclidean')),
#     ('smote', SMOTE(random_state=42)),
#     ('classifier', MultinomialNB())  # Naive Bayes classifier
# ])


# # Define the hyperparameters to tune
# parameters = {
#     'imputer__n_neighbors': [3, 5, 7, 10, 15, 20],
#     'smote__sampling_strategy': ['minority', 'not minority'],
#     # Smoothing parameter for Naive Bayes
#     'classifier__alpha': list(np.logspace(-40.0, 3.0, 100))
# }

# # Perform grid search cross-validation to find the best hyperparameters
# grid_search = GridSearchCV(pipeline, parameters, cv=2, verbose=1)
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
# df_results = df_results.append({'Model': 'Naive Bayes', 'Accuracy': accuracy_score(y_test, y_pred),
#                                 'Precision': precision_score(y_test, y_pred),
#                                 'Recall': recall_score(y_test, y_pred),
#                                 'F1 Score': f1_score(y_test, y_pred),
#                                 'AUC': roc_auc_score(y_test, y_pred),
#                                 'Best params': best_params},
#                                ignore_index=True)

# print(df_results)


# # Assuming y_pred and y_true are the predicted and true labels, respectively
# fpr, tpr, thresholds = roc_curve(y_test_gbm, y_pred_gbm)

# # Compute AUC (Area Under the Curve)
# auc = roc_auc_score(y_test_gbm, y_pred_gbm)

# # Plot ROC curve
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
# plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()


# Bayesian Network Classifier


# # Define structure of Bayesian network using structure learning
# structure = bn.structure_learning.fit(df_merged)

# # Define the parameter grid for the Bayesian network classifier
# param_grid = {
#     'alpha': [0.1, 1, 10],
#     'beta': [0.1, 1, 10]
# }

# # Define the pipeline
# pipeline = Pipeline([
#     ('structure', structure),
#     ('cpds', bn.parameter_learning.fit),
#     ('clf', bn.BayesianNetworkClassifier())
# ])

# # Perform grid search to find the best hyperparameters
# grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# # Print the best hyperparameters and the classification report
# print("Best hyperparameters:", grid_search.best_params_)
# print("Classification report:")
# print(classification_report(y_test, grid_search.predict(y_test)))


# # One-hot encode vairables
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
#     ]
# )

# pipeline = Pipeline([  # ('preprocessor', preprocessor),
#     ('imputer', KNNImputer(metric='nan_euclidean')),
#     ('qda', QuadraticDiscriminantAnalysis())])

# param_grid = {
#     'imputer__n_neighbors': [3, 5, 7],
#     'qda__reg_param': list(np.logspace(-30.0, 3.0, 20))
# }

# grid_search = GridSearchCV(
#     pipeline, param_grid=param_grid, cv=2, n_jobs=6, verbose=1)
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


# # Assuming y_pred and y_true are the predicted and true labels, respectively
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# # Compute AUC (Area Under the Curve)
# auc = roc_auc_score(y_test, y_pred)

# # Plot ROC curve
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
# plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()


# # Append to dataframe
# df_results = df_results.append({'Model': 'QDA', 'Accuracy': accuracy_score(y_test, y_pred),
#                                 'Precision': precision_score(y_test, y_pred),
#                                 'Recall': recall_score(y_test, y_pred),
#                                 'F1 Score': f1_score(y_test, y_pred),
#                                 'AUC': roc_auc_score(y_test, y_pred),
#                                 'Best params': best_params},
#                                ignore_index=True)

# print(df_results)

# # write to csv
# df_results.to_csv(r"results.csv")


# # Support Vector Machine

# # One-hot encode vairables
# # preprocessor = ColumnTransformer(
# #     transformers=[
# #         ('cat', OneHotEncoder(handle_unknown='ignore',
# #          sparse_output=False), categorical_cols)
# #     ]
# # )

# pipeline = Pipeline([  # ('preprocessor', preprocessor),
#     ('imputer', KNNImputer(metric='nan_euclidean')),
#     ('smote', SMOTE(random_state=42)),
#     ('svm', SVC())])

# param_grid = {
#     'imputer__n_neighbors': [7],
#     'smote__sampling_strategy': ['minority'],  # 'not minority'
#     'svm__C': [1],
#     'svm__kernel': ['linear']  # 'linear','poly', 'rbf'
# }

# grid_search = GridSearchCV(
#     pipeline, param_grid=param_grid, cv=2, n_jobs=6, verbose=2)
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


# print("Best parameters: ", grid_search.best_params_)
# print("Train score: ", grid_search.best_score_)
# print("Test score: ", grid_search.score(X_test, y_test))


# # Append to dataframe
# df_results = df_results.append({'Model': 'SVM', 'Accuracy': accuracy_score(y_test, y_pred),
#                                 'Precision': precision_score(y_test, y_pred),
#                                 'Recall': recall_score(y_test, y_pred),
#                                 'F1 Score': f1_score(y_test, y_pred),
#                                 'AUC': roc_auc_score(y_test, y_pred),
#                                 'Best params': best_params},
#                                ignore_index=True)

# print(df_results)


# # Assuming y_pred and y_true are the predicted and true labels, respectively
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# # Compute AUC (Area Under the Curve)
# auc = roc_auc_score(y_test, y_pred)

# # Plot ROC curve
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
# plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()


# # write to csv
# df_results.to_csv(r"results.csv")

# Ada Boost
# Compute class weights for logistic regression
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train_gbm)

# # One-hot encode vairables
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols_gbm)
#     ]
# )

# Create a pipeline with two steps: StandardScaler and LogisticRegression
pipe = Pipeline([  # ('scaler', StandardScaler()),
    ('imputer', KNNImputer(metric='nan_euclidean')),
    ('ada', ADASYN()),
    #('smote', SMOTE(random_state=42)),
    #  ('lr', LogisticRegression(max_iter=10000,
    #                            solver='saga',
    #                            random_state=42))])
    # ('preprocessor', preprocessor), # without preprocessing much higher dont know why!!
    ('classifier', AdaBoostClassifier())])

# Define a param_grid for GridSearchCV that includes the regularization parameter C
param_grid = {
    'imputer__n_neighbors': [3, 5, 7],
    # 'smote__sampling_strategy': ['minority', 'not minority'],
    'classifier__n_estimators': [100, 300, 500],
    'classifier__algorithm': ['SAMME.R', 'SAMME'],
    'classifier__learning_rate': [0.01, 0.1, 1]
}

# Fit the pipeline with GridSearchCV to the training data
grid_search = GridSearchCV(pipe, param_grid=param_grid,
                           cv=2, verbose=1, n_jobs=6, scoring='f1')
grid_search.fit(X_train_gbm, y_train_gbm)

# Use the best estimator from GridSearchCV to predict on the testing data
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

best_params

# Predict on y test
y_pred_gbm = best_model.predict(X_test_gbm)

# Evaluate the model performance
print('Accuracy:', accuracy_score(y_test_gbm, y_pred_gbm))
print('Precision:', precision_score(y_test_gbm, y_pred_gbm))
print('Recall:', recall_score(y_test_gbm, y_pred_gbm))
print('F1 score:', f1_score(y_test_gbm, y_pred_gbm))

# Print the best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Train score: ", grid_search.best_score_)
print("Test score: ", grid_search.score(X_test_gbm, y_test_gbm))

# Append to dataframe
df_results = df_results.append({'Model': 'ADA', 'Accuracy': accuracy_score(y_test, y_pred),
                                'Precision': precision_score(y_test, y_pred),
                                'Recall': recall_score(y_test, y_pred),
                                'F1 Score': f1_score(y_test, y_pred),
                                'AUC': roc_auc_score(y_test, y_pred),
                                'Best params': best_params},
                               ignore_index=True)

print(df_results)

# write to csv
df_results.to_csv(r"results.csv")
