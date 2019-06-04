# --------------------------------- #
# Topic Modeling for the ACLED data #
# --------------------------------- #

# regular imports
import os
import numpy as np
import pandas as pd
import time
import string
import matplotlib.pyplot as plt

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# imports for scikit-learn & LDA
import sklearn
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
import concurrent.futures

# imports for scikit-learn & LDA
import pyLDAvis.sklearn
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig

# Plotly based imports for visualization
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# spaCy based imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# Define the punctuations & stop words
PUNCTUATIONS = string.punctuation
STOPWORDS = list(STOP_WORDS)

# Load the spacy model installed (using the medium model)
NLP = spacy.load('en_core_web_md')

# Define the working directory & raw input datasets
REL_PATH = './'
INFILE = '050319_acled_all.csv'

# Define the directory for saving LDA visualizations as HTML files
LDA_VIS_PATH = './lda_vis/'

# Read in the raw file
df = pd.read_csv(os.path.join(REL_PATH, INFILE))

# Parser & Tokenizer function for conlfict notes
parser = English()
def spacy_tokenizer(note):
    mytokens = parser(str(note))
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in STOPWORDS and word not in PUNCTUATIONS ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

# Tokenizing the entire dataset
df["processed_notes"] = df["notes"].progress_apply(spacy_tokenizer)

# Creating a vectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                             stop_words='english',
                             lowercase=True,
                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(df["processed_notes"])

# Number of Topics/Clusters to create
NUM_TOPICS = 10


# Define the Model
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

# Train the model
lda.fit(data_vectorized)

# Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]])

# Keywords for topics clustered by Latent Dirichlet Allocation
print("LDA Model:")
selected_topics(lda, vectorizer)

# Create the Visualization
pyLDAvis.enable_notebook()
dash = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds='tsne')

# Save the Visualization as an HTML file
out_vis_file = 'full_data_10_topics.html'
pyLDAvis.save_html(dash, fileobj=os.path.join(LDA_VIS_PATH, out_vis_file))

# Class definition for LDA Models
class LDA_Model(object):
    """docstring for LDA_Model."""

    def __init__(self, num_topics, max_iter, vectorizer_type):
        super(LDA_Model, self).__init__()
        self.num_topics = num_topics
        self.max_iter = max_iter
        self.lda = LatentDirichletAllocation(n_components=self.num_topics,
                                             max_iter=self.max_iter,
                                             learning_method='online',
                                             learning_offset=50.,
                                             random_state=0)
        self.count_vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                                                stop_words=STOPWORDS,
                                                lowercase=True,
                                                token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                                stop_words=STOPWORDS)
        # Choose vectorizer type based on flag passed in
        self.vectorizer_type = vectorizer_type
        if self.vectorizer_type == 'tfidf':
            self.vectorizer = self.tfidf_vectorizer
        else:
            self.vectorizer = self.count_vectorizer

    def vectorize_data(self, list):
        self.data_vectorized = self.vectorizer.fit_transform(list)

    def train_lda(self):
        self.lda.fit(self.data_vectorized)

    def selected_topics(self, top_n=10):
        for idx, topic in enumerate(self.lda.components_):
            print("LDA Model:")
            print("Topic %d:" % (idx))
            print([(self.vectorizer.get_feature_names()[i], topic[i]) \
                               for i in topic.argsort()[:-top_n - 1:-1]])

    def visualize(self, out_vis_file):
        # Build a Visualization using pyLDAvis
        self.dash = pyLDAvis.sklearn.prepare(self.lda,
                                             self.data_vectorized,
                                             self.vectorizer,
                                             mds='tsne')
        # Save the Visualization built as an HTML file
        pyLDAvis.save_html(self.dash, fileobj=os.path.join(LDA_VIS_PATH, out_vis_file))

    def map_topics(self, X):
        return self.lda.transform(X)
