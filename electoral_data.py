#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:16:32 2018

@author: Daniel
"""
import os
import pandas as pd

#%%

PATH = '/Users/Daniel/twitter-case/'
os.chdir(PATH)
print("Working directory: %s" % os.getcwd() )

primary = pd.read_csv("primary_results.csv")
demographics = pd.read_csv("county_facts.csv")

# Get winners and vote-shares

#%%

votes = primary.groupby(by="party")


#%%


