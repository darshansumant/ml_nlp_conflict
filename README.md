# Systematizing Notes on Conflict Events
Advanced Machine Learning for Public Policy course project, primarily Natural Language Processing using Deep Learning.


### Data
Armed	Conflict	Location	&	Event	Data	Project (ACLED);
https://www.acleddata.com/


### Folder Structure
This repository contains several test & preliminary jupyter notebooks & python scripts, along with the final files. The main project files are in the 'final_files/' folder, as detailed below:

##### 1. Objective 1 - Conflict Actors Identification
    - objective_1_final.ipynb - *pre-processing, implementation of 2HL NN & analysis*

##### 2. Objective 2 - Relation Extraction
    - objective_2_pre_proc.ipynb - *pre-processing done to modify the strings of text and label the observations*
    - objective_2_final.ipynb - *implementation of LSTM NN*

##### 3. Objective 3 - Topic Modeling
    - Topic_Modeling_runs.ipynb - *complete end-to-end pipeline for Topic Modeling & Visualizations (interactive & static)*

#### Topic Modeling Visualizations
 - D3.js based Interactive Visualizations are in the 'lda_vis/' folder
 - Static Visualizations for topic distribution trends are in the 'lda_vis/distributions/' folder

### Code References
- Tutorials from PyTorch
- https://github.com/claravania/lstm-pytorch/blob/master/model.py for LSTM implementation
- https://github.com/yuchenlin/lstm_sentence_classifier/ for LSTM implementation
- Code for homework 1 of the Advanced Machine Learning class.
- https://www.kaggle.com/thebrownviking20/topic-modelling-with-spacy-and-scikit-learn/notebook for Topic Modeling with spaCy & scikit-learn
