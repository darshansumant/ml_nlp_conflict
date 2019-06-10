import pandas as pd
import numpy as np
import pandas as pd
import spacy
import nltk
from nltk.tag import StanfordNERTagger

data = pd.read_csv('data/050319_acled_all.csv')

print(data.columns)

print(data.shape)


# The intereaction codes from which we can assume directionality are:

# 10: Sole military action.
# 20: Sole rebel action.
# 26: Rebels vs protesters
# 27 Rebels vs civilians
# 30: Sole Political militiaa action
# 36: Political militia vs protesters
# 37: Political militia versus civilans
# 50: Sole Rioter action
# 56: Rioters vs protesters
# 57: Rioters versus civilans


mlist = [26, 27, 36, 37, 56, 57]
data_filtered = data.loc[data['interaction'].isin(mlist)]
# data_filt2 = data_filtered[data_filtered['actor2'].str.contains('Civilians') == True]
# data_filt2.head()
# data_filtered.shape
data_filtered_f = data_filtered.loc[data_filtered['notes'].str.len() > 100 ]
data_filtered_f.shape
data_filtered = data.loc[data['notes'].str.len() > 100 ]
data_filtered.shape
data.shape

data_filtered_f['relation'] = data_filtered_f[]
