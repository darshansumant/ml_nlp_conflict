class text_data:
    def __init__(self, text, vocab_size = 5000):
        self.VOCAB_SIZE = vocab_size
        self.tokens = word_tokenize(text)
        self.tagged = pos_tagging(text)
        self.bigrams = word_ngrams(text, 2)
        self.trigrams = word_ngrams(text, 3)
        self.vocab_counts_tok = Counter([word for content, label in data
                              for word in self.tokens]).most_common(VOCAB_SIZE-1)
        self.vocab_counts_big = Counter([word for content, label in data
                      for word in self.bigrams]).most_common(VOCAB_SIZE-1)
        self.vocab_counts_trig = Counter([word for content, label in data
              for word in self.trigrams]).most_common(VOCAB_SIZE-1)

    def remove_stopw(tokens):
        STOP  = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
        l_clean = []
        for i in STOP:
            if i in l:
                l_clean.append(i)
        return l_clean

    def word_tokenize(s):
        split_l = s.lower().replace('.', '').replace(',', '').replace(';', '').replace(':', '').replace('!', '').replace('?', '').split()
        return split_l

    def create_ngrams(s):
        pass

    def pos_tagging(s):
        pass

    def rel_extraction(s):
        pass

class set_data():
    def __init__(self, data):

        self.train_set = set_train()
        self.test_set = set_test()
        self.val_set = set_val()

    def set_test(data):
        return data[data[year] < 2019]

    def set_train(data):
        return data[data[year] < 2018]

    def set_val(data):
        return data[data[year] == 2019]

class Model:
    def __init__(self, data):
        # Vocabulary is a set that stores every word seen in the
        # training data
        self.vocab_counts = Counter([word for content, label in data
                              for word in word_tokenize(content)]
                            ).most_common(VOCAB_SIZE-1)
        # word to index mapping
        self.word_to_idx = {k[0]: v+1 for v, k in
                            enumerate(self.vocab_counts)}
        # all the unknown words will be mapped to index 0
        self.word_to_idx["UNK"] = 0
        self.idx_to_word = {v:k for k, v in self.word_to_idx.items()}
        self.label_to_idx = {POS_LABEL: 0, NEG_LABEL: 1}
        self.idx_to_label = [POS_LABEL, NEG_LABEL]
        self.vocab = set(self.word_to_idx.keys())

    def train_model(self, data):
        '''
        Train the model with the provided training data
        '''
        raise NotImplementedError

    def classify(self, data):
        '''
        Classify the documents with the model
        '''

        raise NotImplementedError
