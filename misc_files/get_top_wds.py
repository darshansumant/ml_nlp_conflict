import pandas as pd
import torch
import torch.utils.data as tud
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter, defaultdict
import operator
import os, math
import numpy as np
import random
import copy
import spacy
nlp = spacy.load("en_core_web_sm")

data = pd.read_csv('data/acled_all.csv')
VOCAB_SIZE = 5000
seed = 30255
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
data.head()


def remove_stopwords(l):
    STOP  = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
    l_clean = []
    for i in l:
        if i not in STOP:
            l_clean.append(i)
    return l_clean


def word_tokenize(s, clean = False):
    if type(s) != str:
        return None
    split_l = s.lower().replace('.', '').replace(',', '').replace(';', '').replace(':', '').replace('!', '').replace('?', '').split()
    if clean:
        clean_l = remove_stopwords(split_l)
        return clean_l

    return split_l

class Model:
    def __init__(self, data, clean = False):
        # Vocabulary is a set that stores every word seen in the
        # training data
        self.vocab_counts = Counter([word for content in data
                              for word in word_tokenize(content, clean) if word]
                            ).most_common(VOCAB_SIZE-1)
        # word to index mapping
        self.word_to_idx = {k[0]: v+1 for v, k in
                            enumerate(self.vocab_counts)}
        # all the unknown words will be mapped to index 0
        self.word_to_idx["UNK"] = 0
        self.vocab = set(self.word_to_idx.keys())

        self.verb_counts = Counter([token.lemma_ for content in data
                              for token in nlp(content) if token.pos_ == "VERB"]
                            ).most_common(VOCAB_SIZE-1)

        self.noun_counts = Counter([chunk.text for content in data
                              for chunk in nlp(content).noun_chunks]
                            ).most_common(VOCAB_SIZE-1)


class TextClassificationDataset(tud.Dataset):
    '''
    PyTorch provides a common dataset interface.
    See https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    The dataset encodes documents into indices.
    With the PyTorch dataloader, you can easily get batched data for
    training and evaluation.
    '''
    def __init__(self, word_to_idx, data):

        self.data = data
        self.word_to_idx = word_to_idx
        self.vocab_size = VOCAB_SIZE

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = np.zeros(self.vocab_size)
        item = torch.from_numpy(item)
        for word in word_tokenize(self.data[idx]):
            item[self.word_to_idx.get(word, 0)] += 1
        return item

data.dropna(subset=['NOTES'], inplace = True)
x_wstopwords = Model(data['NOTES'], False)
verb_count_complete = x_wstopwords.verb_counts
#x_nstopwords = Model(data['NOTES'], True)
#x_nstopwords.vocab_counts[:20]
verb_count_complete
# verbs that are not related to conflict: ["be", "report", "have", "take", "code", "suspect", "follow", "use", "cause", "leave", "occur", "try", "know", \
# "", "manage", "do", "begin", "form"]

# ******************************************************************************
# Getting the verbs for a subset of type of events in which we know what the relation of the direction is.
# ******************************************************************************
verb_count_complete
mlist = [26, 27, 36, 37, 56, 57]
data_filtered = data.loc[data['INTERACTION'].isin(mlist)]
data_filtered.shape
# y_wstopwords = Model(data_filtered['NOTES'], False)
# verb_count_filtered = y_wstopwords.verb_counts

final_data = data_filtered.loc[data_filtered['NOTES'].str.len() > 100 ]

# ******************************************************************************
# Getting the nouns for a subset of type of events in which we know civilians are attacked
# ******************************************************************************

# data_filt2 = data_filtered[data_filtered['ACTOR2'].str.contains('Civilians') == True]
# z_wstopwords = Model(data_filt2['NOTES'], False)
#
# noun_count_complete = z_wstopwords.noun_counts
# noun_count_filtered = noun_count_complete
#

# ******************************************************************************
# Turn into list of verbs
# ******************************************************************************

# Remove verbs we're not interested in

# vbs_complete = [word for word, count in verb_count_complete]
# vbs_filtered = [word for word, count in verb_count_filtered]
#
# noun_filtered = [word for word, count in noun_count_filtered]
# print(vbs_complete)
# print(vbs_filtered)
# print(noun_count_filtered)

# ******************************************************************************
# Read processed list of verbs
# ******************************************************************************
vbs = pd.read_csv('data/verbs.csv')
vbs.head()

vbs['clean_list'] = vbs['List'].str.replace('(', '')
vbs['clean_list'] = vbs['clean_list'].str.replace('[', '')
vbs['clean_list'] = vbs['clean_list'].str.replace(']', '')
vbs['clean_list'] = vbs['clean_list'].str.replace("'", '')
vbs['clean_list'] = vbs['clean_list'].str.strip()

vbs = vbs[vbs['Filter'] == 1]
vbs.shape
list_vbs = vbs.clean_list
list_vbs[:15]

# ******************************************************************************
# Create automatic labelling.
# ******************************************************************************
final_data['tagged_rel'] = ''
num_v = len(list_vbs)
for cnt, vb in enumerate(list_vbs):
    final_data['tagged_rel'] = final_data.apply(lambda x: vb if vb in x['NOTES'] else x['tagged_rel'], axis =1)
    print(str(cnt) + '/' + str(num_v))
print('end_loop')

final_data['tagged_rel'].head(100)

# ******************************************************************************
# Replace part of strings in the representation.
# ******************************************************************************

# 1. Replace aggresors *********************************************************
# -------- Get actors
final_data['clean_actor'] = final_data[final_data['ACTOR1'].str.lower() != "civilians"]
final_data['clean_actor'] = final_data['ACTOR1'].str.split('(').str[0]
final_data['temp_clean_actor1'] = final_data['clean_actor'].str.split(':').str[0].str.strip().apply(lambda x: ' ' + str(x) + ' ')
final_data['temp_clean_actor2'] = final_data['clean_actor'].str.split(':').str[1].str.strip().apply(lambda x: ' ' + str(x) + ' ')
uniq_actors = list(final_data['temp_clean_actor1'].str.lower().dropna().unique()) + list(final_data['temp_clean_actor2'].str.lower().dropna().unique())

# Getting those that are unidentified actors & different ways of spelling them
list_unident_actors = final_data['NOTES'].str.split('unidentified armed').str[1].str.split(' ').str[1].str.replace('.', '').str.replace(',', '').str.replace(';', '').str.replace("'s", "").str.lower().dropna().unique()
unident_actors_l = list_unident_actors[((list_unident_actors != 'and') & (list_unident_actors != 'in') & (list_unident_actors != 'on')& (list_unident_actors != 'nan'))]
unident_actors = list(map(lambda x: 'unidentified armed '+ str(x), unident_actors_l))
unident_actors_v2 = list(map(lambda x: 'armed '+ str(x), unident_actors_l))
unident_actors_v3 = list(map(lambda x: 'unidentified '+ str(x), unident_actors_l))

list_aggrs = uniq_actors + unident_actors + unident_actors_v2 + unident_actors_v3
num = len(list_aggrs)
final_data['tagged_str'] = final_data['NOTES']
# --------
for cnt, token in enumerate(list_aggrs):
    final_data['tagged_str'] = final_data['tagged_str'].str.lower().str.replace(token, 'aggresor')
    print(str(cnt) + '/' + str(num))
print('end loop')

final_data['tagged_str'] = final_data['tagged_str'].str.lower().str.replace('aggresors', 'AGGRESOR')
final_data['tagged_str'] = final_data['tagged_str'].str.lower().str.replace('aggresor', 'AGGRESOR')
x = final_data[final_data['tagged_str'].str.contains('AGGRESOR')]
x.shape # These are for how many we did NOT get the tag
x = final_data[final_data['tagged_str'].str.contains('AGGRESOR')]
final_data['tagged_str']

final_data[['NOTES','tagged_str']].head(100)

# 2. Replace civilians *********************************************************
list_civs = ['passengers', 'civilians','residents' 'passenger', 'civilian', 'village', 'family', 'families', 'people', 'tourist','tourists' 'villagers', 'women', 'children', 'citizens', 'population']

for cnt, token in enumerate(list_civs):
    if token:
        final_data['tagged_repr'] = final_data['tagged_repr'].str.lower().str.replace(token,'victim')
        print(str(cnt) + '/' + str(num))
print('end loop')

final_data['tagged_str'] = final_data['tagged_str'].str.lower().str.replace('victims', 'VICTIM')
final_data['tagged_str'] = final_data['tagged_str'].str.lower().str.replace('victim', 'VICTIM')
y = final_data[final_data['tagged_str'].str.contains('AGGRESOR')]

y = final_data[~final_data['tagged_str'].str.contains('AGGRESOR')]
y.shape # These are for how many we did NOT get the tag
final_data[['tagged_str', 'NOTES']]
# 3. Subset our data for this case *********************************************
final_data_subset = final_data[final_data['tagged_rel'] != '']
print('we lose {} observations by filtering through our classification of relationship, which is equivalent to {} %'.format(final_data.shape[0] - final_data_subset.shape[0], ((final_data.shape[0] - final_data_subset.shape[0])/final_data.shape[0])*100))

final_data_subset_2 = final_data[final_data['tagged_repr'].str.contains('AGGRESOR')]
print('we lose {} observations by replacing aggresor in phrase, which is equivalent to {} %'.format(final_data.shape[0] - final_data_subset_2.shape[0], ((final_data.shape[0] - final_data_subset_2.shape[0])/final_data.shape[0])*100))

final_data_subset_3 = final_data[final_data['tagged_repr'].str.contains('VICTIM')]
print('we lose {} observations by replacing victim in phrase, which is equivalent to {} %'.format(final_data.shape[0] - final_data_subset_3.shape[0], ((final_data.shape[0] - final_data_subset_3.shape[0])/final_data.shape[0])*100))

final_pre = final_data[final_data['tagged_repr'].str.contains('AGGRESOR')]
final= final_pre[final_data_pre['tagged_repr'].str.contains('VICTIM')]

x = pd.read_csv('data/data_prepr.csv')

x[['NOTES', 'tagged_str']]
x['tagged_rel'].value_counts(normalize=True)
