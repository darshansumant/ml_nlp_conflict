# --------------------------------- #
# Topic Modeling for the ACLED data #
# --------------------------------- #

import pandas as pd
import spacy
import os

rel_path = './'
infile = '050319_acled_all.csv'

# Load the spacy model installed
nlp = spacy.load('en_core_web_md')

# Read in the raw file
df = pd.read_csv(os.path.join(rel_path, infile))
