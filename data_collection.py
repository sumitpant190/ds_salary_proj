# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:22:47 2023

@author: sumit
"""
import glassdoor_scraper as gs
import pandas as pd
path = 'C:/Users/sumit/Documents/ds_salary_proj/chromedriver'

df = gs.get_jobs('data scientist',1000, False, path, 15)

df.to_csv('glassdoor_jobs.csv', index = False)


