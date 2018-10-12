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

#df.drop(['B', 'C'], axis=1)

#%%

survey_df.rename(columns={'insta_user_id':'user_id'}, inplace = True)
survey_df['user_id'] = survey_df['user_id'].astype(int)
im_anp_obj_face_frame['user_id'] = im_anp_obj_face_frame['user_id'].astype(int)

#%%

df = pd.merge(survey_df[['PERMA', 'user_id']], im_anp_obj_face_frame, how='inner', on='user_id') # For now we just take the outcome variable 

df = df.drop_duplicates(subset=None, keep='first', inplace=False)
#df = df.drop(['index'],
             #axis=1)
print(im_anp_obj_face_frame['user_id'].nunique(), "unique respondents in features data")
print(survey_df['user_id'].nunique(), "unique respondents in survey data")
print("When merged, we have", df['user_id'].nunique(), "unique respondents in the df, with", df['image_id'].nunique(), "unique images")

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
sample = df_enriched.sample(frac=0.01, replace=False) # 1% sample becasue my laptop sucks
sample['PERMA'].nunique() # Same abount of unique PERMA-scores as in whole dataset
#sample.to_csv("/Users/Daniel/Desktop/enriched_df.csv", sep=";" , index=False)


#%% And now for the fun stuff
        # Release da tensorflow... 





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









