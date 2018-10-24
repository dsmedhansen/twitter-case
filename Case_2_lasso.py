#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 13:25:15 2018

@author: martijn
"""
#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from pandas.tools.plotting import scatter_matrix
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

#%%

#df = pd.read_csv('/home/martijn/Downloads/use_case_2/photo_df.csv', delimiter=",")
df = pd.read_csv('/Users/Daniel/twitter-case/photo_df.csv', delimiter=",")
df = df.drop(['index','image_posted_time'], axis=1)
df = df.dropna()
correlation_matrix = df.corr()
print(correlation_matrix)

plt.matshow(correlation_matrix)

scatter_matrix(correlation_matrix, alpha=0.2)
df_enriched = pd.get_dummies(df, columns=['image_filter'])
#print(df_enriched.head())
#print(df_enriched.shape)
#print(df_enriched.dtypes)


df_PERMA = df_enriched.drop(['P','E','R','M','A'], axis=1)
df_P = df_enriched.drop(['PERMA','E','R','M','A'], axis=1)
df_E = df_enriched.drop(['P','PERMA','R','M','A'], axis=1)
df_R = df_enriched.drop(['P','E','PERMA','M','A'], axis=1)
df_M = df_enriched.drop(['P','E','R','PERMA','A'], axis=1)
df_A = df_enriched.drop(['P','E','R','M','PERMA'], axis=1)


# extract our target variable into an array 
y = df_PERMA.PERMA.values
# Drop PERMA score
df_PERMA2 = df_PERMA.drop(['PERMA'], axis=1)
# Create a matrix from the remaining data
X = df_PERMA2.values
# Store the column/feature names into a list "colnames"
colnames = df_PERMA2.columns
# create a lasso regressor
lasso = Lasso(alpha=0.00001, normalize=True)
# Fit the regressor to the data
lasso.fit(X,y)
# Compute and print the coefficients
lasso_coef = lasso.coef_

merged_df = pd.DataFrame({'variables': colnames, 'lasso_coef':lasso_coef})
print(merged_df)
#filters that score above 0.4
#image_filter_Brooklyn    0.577593
#image_filter_Dogpatch    0.921600
#image_filter_Gotham    0.834619
#image_filter_Maven    1.586243
#image_filter_Poprocket    0.458768
#image_filter_Vesper    0.459742

# Plot the coefficients
plt.plot(range(len(colnames)), lasso_coef)
plt.xticks(range(len(colnames)), colnames.values, rotation=60)
plt.margins(0.02)
plt.show()

# Import the necessary module


# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

# find the mean of our cv scores here
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Create an array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Use this function to create a plot    
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

# Display the plot
display_plot(ridge_scores, ridge_scores_std)

'''
y = df_P.P.values
# Drop PERMA score
df_P2 = df_P.drop(['P'], axis=1)
# Create a matrix from the remaining data
X = df_P2.values
# Store the column/feature names into a list "colnames"
colnames = df_P2.columns
# create a lasso regressor
lasso = Lasso(alpha=0.00001, normalize=True)
# Fit the regressor to the data
lasso.fit(X,y)
# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

import matplotlib.pyplot as plt
# Plot the coefficients
plt.plot(range(len(colnames)), lasso_coef)
plt.xticks(range(len(colnames)), colnames.values, rotation=60)
plt.margins(0.02)
plt.show()
'''

#%%

def format_data(df):
    # Targets are perma scores
    labels = df_enriched['PERMA']
    
    df = df_enriched[['PERMA',
                      'image_filter_Brooklyn',
                      'image_filter_Dogpatch',
                      'image_filter_Gotham',
                      'image_filter_Maven',
                      'image_filter_Poprocket',
                      'image_filter_Vesper']]
    
    # Split into training/testing sets with 25% split
    X_train, X_test, y_train, y_test = train_test_split(df, labels, 
                                                        test_size = 0.25,
                                                        random_state=42)

    
    return X_train, X_test, y_train, y_test

def format_all_features(df):
    # Targets are perma scores
    labels = df_enriched['PERMA']
    
    df = df_enriched.loc[:,'PERMA': 'image_filter_X-Pro II']
    
    # Split into training/testing sets with 25% split
    X_train, X_test, y_train, y_test = train_test_split(df, labels, 
                                                        test_size = 0.25,
                                                        random_state=42)

    
    return X_train, X_test, y_train, y_test

X_train2, X_test2, y_train2, y_test2 = format_all_features(df_enriched)
X_train, X_test, y_train, y_test = format_data(df_enriched)


#%%

# Normalize features
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

mean = X_train2.mean(axis=0)
std = X_train2.std(axis=0)
X_train2 = (X_train2 - mean) / std
X_test2 = (X_test2 - mean) / std


#%%

# Calculate Mean average error and Root mean square deviation
def evaluate_predictions(predictions, true):
    mae = np.mean(abs(predictions - true))
    rmse = np.sqrt(np.mean((predictions - true) ** 2))
    
    return mae, rmse

# Evaluate several ml models by training on training set and testing on testing set
def evaluate(X_train, X_test, y_train, y_test):
    # Names of models
    
    model_name_list = ['Linear Regression', 'ElasticNet Regression',
                      'Random Forest', 'Extra Trees', 'Gradient Boosted',
                      'Baseline']
    
    X_train = X_train.drop(columns='PERMA')
    X_test = X_test.drop(columns='PERMA')
    
    # Instantiate the models
    model1 = LinearRegression()
    model2 = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model3 = RandomForestRegressor(n_estimators=50)
    model4 = ExtraTreesRegressor(n_estimators=50)
    model5 = GradientBoostingRegressor(n_estimators=100)
    
    # Dataframe for results
    results = pd.DataFrame(columns=['Mean Average Error',
                                    'Root-Mean-Squared-Error'],
                                       index = model_name_list)
    
    # Train and predict with each model
    for i, model in enumerate([model1, model2, model3, model4, model5]):

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Metrics
        mae = np.mean(abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        
        # Insert results into the dataframe
        model_name = model_name_list[i]
        results.loc[model_name, :] = [mae, rmse]
        #print("...")
    
    # Median Value Baseline Metrics
    baseline = np.median(y_train)
    baseline_mae = np.mean(abs(baseline - y_test))
    baseline_rmse = np.sqrt(np.mean((baseline - y_test) ** 2))
    
    results.loc['Baseline', :] = [baseline_mae, baseline_rmse]
    
    return results
   
# Naive baseline is the median
median_pred = X_train['PERMA'].median()
median_preds = [median_pred for _ in range(len(X_test))]
true = X_test['PERMA']

#median_pred2 = X_train2['PERMA'].median()
#median_preds2 = [median_pred2 for _ in range(len(X_test2))]
#true2 = X_test2['PERMA']

# Display the naive baseline metrics
mb_mae, mb_rmse = evaluate_predictions(median_preds, true)
#mb_mae2, mb_rmse2 = evaluate_predictions(median_preds2, true2)

print('\nMedian Baseline Mean Average Error: {:.4f}'.format(mb_mae))
print('Median Baseline Root-Mean-Sqare Error: {:.4f}'.format(mb_rmse))
#print('Median Baseline all features MAE: {:.4f}'.format(mb_mae2))
#print('Median Baseline all features RMSE: {:.4f}'.format(mb_rmse2))

results_selected_features = evaluate(X_train, X_test, y_train, y_test)
results_all_features = evaluate(X_train2, X_test2, y_train2, y_test2)

print("\nSelected features:\n", results_selected_features,"\n")
print("All features:\n", results_all_features,"\n")




