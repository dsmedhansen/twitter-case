#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 13:25:15 2018

@author: martijn
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from pandas.tools.plotting import scatter_matrix

df = pd.read_csv('/home/martijn/Downloads/use_case_2/photo_df.csv', delimiter=",")
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
print(lasso_coef)


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