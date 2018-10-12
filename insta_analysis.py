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
    
    
#df['hID'].nunique()
print(survey_df['insta_user_id'].nunique(), "unique respondents in the survey data")

combined = pd.merge(survey_df, im_anp_obj_frame, how='inner', on='image_id')


#%%

corr = im_anp_obj_face_frame.corr()
 
plot = sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

plot.subplots(figsize=(20,15))

figure = plot.get_figure()    

#%%

# Spearman rank correlation 

from scipy.stats import spearmanr, pearsonr

# SERIES OF TUPLES (<scipy.stats.stats.SpearmanrResult> class)
spr_all_result = a.apply(lambda col: spearmanr(col, b.ix[:,0]), axis=0)

# SERIES OF FLOATS
spr_corr = a.apply(lambda col: spearmanr(col, b.ix[:,0])[0], axis=0)
spr_pvalues = a.apply(lambda col: spearmanr(col, b.ix[:,0])[1], axis=0)

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

