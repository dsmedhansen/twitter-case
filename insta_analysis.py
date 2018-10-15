#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Daniel
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#%%

#Read the individual data frames

folder = "/Users/Daniel/Google Drive/Master/Fundamentals of data science/Visual_well_being/"
anp_df = pd.read_pickle(folder + r'anp.pickle') 
face_df = pd.read_pickle(folder + r'face.pickle') 
image_df = pd.read_pickle(folder + r'image_data.pickle') 
metrics_df = pd.read_pickle(folder + r'image_metrics.pickle') 
object_labels_df = pd.read_pickle(folder + r'object_labels.pickle') 
survey_df = pd.read_pickle(folder + r'survey.pickle') 


#%%

# Use imputation to deal with the one missing value... 

#%%

# Merge them based on the image_id so that we have a large data frame containing all the elements

image_anp_frame = pd.merge(image_df, anp_df, how='inner', on='image_id') # This is handy!!
im_anp_obj_frame = pd.merge(image_anp_frame, object_labels_df, how='inner', on='image_id')
im_anp_obj_face_frame = pd.merge(im_anp_obj_frame, face_df, how='inner', on='image_id')
im_anp_obj_face_frame = pd.merge(im_anp_obj_frame, face_df, how='inner', on='image_id')

del anp_df, face_df, image_df, metrics_df, object_labels_df, im_anp_obj_frame, image_anp_frame

im_anp_obj_face_frame =  im_anp_obj_face_frame.drop(
        
                            ['image_link', 
                            'image_url', 
                            'user_full_name',
                            'user_name',
                            'user_website',
                            'user_profile_pic',
                            'user_bio',
                            'face_mustache',
                            'face_beard',
                            'face_beard_confidence',
                            'face_sunglasses',
                            'face_mustache_confidence',
                            'image_posted_time_unix',
                            'image_height',
                            'image_width',
                            'anp_label',
                            'data_amz_label',
                            'data_amz_label_confidence',
                            'emotion_label',
                            'eyeglasses',
                            'eyeglasses_confidence',
                            'face_gender_confidence',
                            'face_smile_confidence',
                            'face_id',
                            'emo_confidence',
                            'face_age_range_low',
                            'face_age_range_high'], axis=1)

#%%

survey_df.rename(columns={'insta_user_id':'user_id'}, inplace = True)
survey_df['user_id'] = survey_df['user_id'].astype(int)
im_anp_obj_face_frame['user_id'] = im_anp_obj_face_frame['user_id'].astype(int)

#%%

df = pd.merge(survey_df[['PERMA', 'user_id']], im_anp_obj_face_frame, how='inner', on='user_id') # For now we just take the outcome variable 

df = df.drop_duplicates(subset=None, keep='first', inplace=False)

print(im_anp_obj_face_frame['user_id'].nunique(), "unique respondents in features data")
print(survey_df['user_id'].nunique(), "unique respondents in survey data")
print("When merged, we have", df['user_id'].nunique(), "unique respondents in the df, with", df['image_id'].nunique(), "unique images")

#%%

# Missing data in the PERMA variable? Replace with imputation...

#%%

df =  df.drop(
                            ['image_id',
                             'image_posted_time',
                             'user_id',
                             ], axis=1)

del im_anp_obj_face_frame

#%%

# Enrich data with one-hot vectors aka dummy-variables

df_enriched = pd.get_dummies(df, columns=['image_filter', 'face_smile', 'face_gender', 'face_emo'])
#df_enriched.to_csv("/Users/Daniel/Desktop/enriched_df.csv", sep=";" , index=False)

#%%

df_enriched = pd.read_csv("/Users/Daniel/Desktop/enriched_df.csv", sep=";")

#%%

# And now we do the feature selection

df_enriched.corr()['PERMA'].sort_values()

# Drop outliers to avoid problems with overfitting
        # Use the Kalman filter to find and replace outliers with expected values

#%%

# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Splitting data into training/testing
from sklearn.model_selection import train_test_split

#%%


# Takes in a dataframe, finds the most correlated variables with the
# grade and returns training and testing datasets
def format_data(df):
    # Targets are perma scores
    labels = df['PERMA']

    # Find correlations with PERMA
    most_correlated = df.corr().abs()['PERMA'].sort_values(ascending=False)
    
    # Maintain the top 6 most correlation features with Grade
    most_correlated = most_correlated[:10]
    
    df = df.loc[:, most_correlated.index]
    
    # Split into training/testing sets with 25% split
    X_train, X_test, y_train, y_test = train_test_split(df, labels, 
                                                        test_size = 0.25,
                                                        random_state=42)
    
    print("Most correlated with target variable", most_correlated)
    
    return X_train, X_test, y_train, y_test

#%%

# Use PCA to find most significant variables and compare with pure correlations

#X_train = X_train.dropna(axis=0).reset_index()

df_enriched = df_enriched.dropna(axis=0)
X_train, X_test, y_train, y_test = format_data(df_enriched)
test = X_test.sample(n=20) # Note that some of the same filters as in the paper are correlated with PERMA...

#%% Normalize features

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

#%%

print(X_train.shape)
print(X_test.shape)

#%%

# Calculate mae and rmse
def evaluate_predictions(predictions, true):
    mae = np.mean(abs(predictions - true))
    rmse = np.sqrt(np.mean((predictions - true) ** 2))
    
    return mae, rmse

# Naive baseline is the median
median_pred = X_train['PERMA'].median()
median_preds = [median_pred for _ in range(len(X_test))]
true = X_test['PERMA']

# Display the naive baseline metrics
mb_mae, mb_rmse = evaluate_predictions(median_preds, true)
print('Median Baseline MAE: {:.4f}'.format(mb_mae))
print('Median Baseline RMSE: {:.4f}'.format(mb_rmse))

#%%

# Evaluate several ml models by training on training set and testing on testing set
def evaluate(X_train, X_test, y_train, y_test):
    # Names of models
    #model_name_list = ['Linear Regression', 'ElasticNet Regression',
                      #'Random Forest', 'Extra Trees', 'SVM',
                       #'Gradient Boosted', 'Baseline']
    
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
    #model5 = SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')
    model6 = GradientBoostingRegressor(n_estimators=20)
    
    # Dataframe for results
    results = pd.DataFrame(columns=['mae', 'rmse'], index = model_name_list)
    
    # Train and predict with each model
    #for i, model in enumerate([model1, model2, model3, model4, model5, model6]):
    for i, model in enumerate([model1, model2, model3, model4, model6]):

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Metrics
        mae = np.mean(abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        
        # Insert results into the dataframe
        model_name = model_name_list[i]
        results.loc[model_name, :] = [mae, rmse]
        print("...")
    
    # Median Value Baseline Metrics
    baseline = np.median(y_train)
    baseline_mae = np.mean(abs(baseline - y_test))
    baseline_rmse = np.sqrt(np.mean((baseline - y_test) ** 2))
    
    # Can we add precision and recall to this output?ß
    
    results.loc['Baseline', :] = [baseline_mae, baseline_rmse]
    
    return results

#%%

results4 = evaluate(X_train, X_test, y_train, y_test)

#%%

# Merge on respondent ID?
# Merge with PERMA data
# Sentiment label
# Find more variables from paper...
# Order image filter into scale from happy to not happy
    # Face smile (True vs. False)
    # Face emotion
    # Use filter assign values to happiness... 
    # See if PERMA actually is one dimensional (check whether it indeed is a scale, with factor analysis)
        # Ask Emma how this works...
    # Use one-hot encoder to generate features from filters
    # Do the same with face-smile
    # Maybe use time of posting as variable as well... Use regex to remove date
    # Drop NaN's from survey_df
    # Merge with survey and run first regression to see which variables are correlated
    
    
#df['hID'].nunique()
print(survey_df['user_id'].nunique(), "unique respondents in the survey data")

#combined = pd.merge(survey_df, im_anp_obj_face_frame, how='inner', on='image_id')

#%%

corr = im_anp_obj_face_frame.corr()
 
plot = sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

plot.subplots(figsize=(20,15))

figure = plot.get_figure()    


#%%

# Compute the correlation matrix
corr = im_anp_obj_face_frame.corr(method='spearman')
#corr2 = im_anp_obj_face_frame.corr()

#%%

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns_plot = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Save result 
    # Plot p-values...
    # This makes no sense without the outcome variable...
    
sns_plot.savefig("/Users/Daniel/Desktop/rank.png")
