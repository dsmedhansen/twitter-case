#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Daniel
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
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

# We just reconstruct the scale from the documentation that was provided on Canvas where the PERMA score is made up of a mean for all scores
print("First we have", survey_df['PERMA'].isnull().sum(), "missing values")
missing_PERMA_score = (survey_df.iloc[4][survey_df.columns[pd.Series(survey_df.columns).str.startswith(('E_','R_','M_','A_'))]].sum()+survey_df.iloc[4]['P_1']+survey_df.iloc[4]['P_2']+survey_df.iloc[4]['P_3'])/15

# survey_df.iloc[4]['PERMA'] # Location of missing value

survey_df.PERMA = survey_df['PERMA'].fillna(missing_PERMA_score)

print("After imputing with the single PERMA-score we have", survey_df['PERMA'].isnull().sum(), "missing values")

survey_df = survey_df.drop(['index'], axis = 1)

# remove duplicated user_id
survey_df = survey_df[~survey_df['insta_user_id'].duplicated(keep='first')]

#%%

from factor_analyzer import FactorAnalyzer

fa = FactorAnalyzer()
fa_features = survey_df[['P','E','R','M','A']]
fa.analyze(fa_features, 3, rotation=None) # No rotation = no correlation between factors
ev, v = fa.get_eigenvalues()

ev # Eigenvalue drops below 
v
fa.loadings

#%%

# Factor analysis to check if all five variables load on the same latent construct
# Construct argument from this: no need to look at the questions when we have FA
    # The all load on the same latent construct
    # However, when using the orthogonal rotation...

fa = FactorAnalyzer()
fa.analyze(fa_features, 3, rotation='oblimin') # Oblique rotatio: allows for inter-correlation
ev, v = fa.get_eigenvalues()

ev # Eigenvalue drops below 
v
fa.loadings


#%%

"""
From an older paper I wrote on FA: 
In essence, in factor analysis, the factor loadings are a measure of the strength of the interaction 
between variables in a dataset and each factor. In assessing the significance of factor loadings, 
this paper will use the rule of thumb suggested by Stevens (1992) that a factor loading of 0.4 four 
is enough to consider the loadings of the item reliable. To determine the optimal amount of factors for 
a model DeVellis (2003, p. 114) points out that factors with eigenvalues less than 1 should 
not be included in the model because an eigenvalue below 1 marks the point at which the items 
will contain “less information than the average item” (DeVellis, 2003, p. 114). 
The eigenvalue corresponds to the amount of variance in a model that is accounted for by the factors. 
In consequence, an eigenvalue of 1 corresponds to 100% of the total variance accounted for by the 
items in the model. In a Scree plot this corresponds to the “elbow” in the plot (DeVellis, 2003).

"""

#%%

# Merge them based on the image_id so that we have a large data frame containing all the elements
image_anp = pd.merge(image_df, anp_df, how='inner', on='image_id') # This is handy!!
del image_df, anp_df

#%%
# Merge with metricts
image_anp_metrics = pd.merge(image_anp, metrics_df, how='inner', on='image_id')
del image_anp, metrics_df

#%%
# Merge with object labels
image_anp_metrics_objectlabes = pd.merge(image_anp_metrics, object_labels_df, how='inner', on='image_id')
del image_anp_metrics, object_labels_df

#%%
# Merge wth face labels
image_anp_metrics_objectlabes_face = pd.merge(image_anp_metrics_objectlabes, face_df, how='inner', on='image_id')
del image_anp_metrics_objectlabes, face_df

#%%
# Rename df to correspond to old name 
im_anp_obj_face_frame = image_anp_metrics_objectlabes_face

#%%
im_anp_obj_face_frame =  im_anp_obj_face_frame.drop(
        
                            ['face_emo',
                            'image_link', 
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
                            'face_age_range_high',
                            'comment_count_time_created',
                            'like_count_time_created'], axis=1)

#%%

im_anp_obj_face_frame = im_anp_obj_face_frame.drop_duplicates(subset=None, keep='first', inplace=False)

#%%

survey_df.rename(columns={'insta_user_id':'user_id'}, inplace = True)
survey_df['user_id'] = survey_df['user_id'].astype(int)
im_anp_obj_face_frame['user_id'] = im_anp_obj_face_frame['user_id'].astype(int)

#%%

df = pd.merge(survey_df[['P','E','R','M','A','PERMA', 'user_id']], im_anp_obj_face_frame, how='inner', on='user_id') # For now we just take the outcome variable 

df = df.drop_duplicates(subset=None, keep='first', inplace=False) # Drop all duplicates

print(im_anp_obj_face_frame['user_id'].nunique(), "unique respondents in features data")
print(survey_df['user_id'].nunique(), "unique respondents in survey data")
print("When merged, we have", df['user_id'].nunique(), "unique respondents in the df, with", df['image_id'].nunique(), "unique images")

#%%

#df =  df.drop(
#                            ['image_id',
#                             'user_id',
#                             ], axis=1)

del im_anp_obj_face_frame, image_anp_metrics_objectlabes_face


#%% Make time of day variable

# Use this one for day or night validation: https://stackoverflow.com/questions/43299500/pandas-how-to-know-if-its-day-or-night-using-timestamp

df['image_posted_time'] = pd.to_datetime(df['image_posted_time'])
df['time_of_day'] = df['image_posted_time'].dt.time
time_category = pd.Series(["Morning","Afternoon","Evening","Night"], dtype="category")

#%% Aggregation of data - each image an obs
 
score = df[['image_id','anp_sentiment','emotion_score']]
score = score.groupby('image_id').mean()
score['image_id'] = score.index

df2 = df[:]
df2 = df2.drop(['anp_sentiment','emotion_score','face_gender','face_smile'],axis=1)
df2 = df2.drop_duplicates(subset=None, keep='first', inplace=False)

df2 = pd.merge(df2,score,how='inner',on='image_id')
df2 = df2.drop(['image_id','user_id'],axis=1)


#%%

# Enrich data with one-hot vectors aka dummy-variables

df_enriched = pd.get_dummies(df, columns=['image_filter', 'face_smile', 'face_gender', 'face_emo'])
#df_enriched.to_csv("/Users/Daniel/Desktop/enriched_df.csv", sep=";" , index=False)

#%%

df_enriched = pd.read_csv("/Users/Daniel/Desktop/enriched_df.csv", sep=";")

#%%
    # @Martijn: Use Lasso filter to find best variables

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
# PERMA and returns training and testing datasets

def format_data(df):
    # Targets are perma scores
    labels = df['PERMA']

    # Find correlations with PERMA
    most_correlated = df.corr().abs()['PERMA'].sort_values(ascending=False)
    
    # Maintain the top 6 most correlation features with PERMA
    most_correlated = most_correlated[:15]
    
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
