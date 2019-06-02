import utils

class Dataset(data):
    def __init__(self, data):
        self._data = data
        self.hard_test = False
        self._test = None
        self._train = None
        self._val = None

    def split_data_random(self, percentage_train, percentage_test):
        '''
        Set random split of the data
        '''
        print("train, test and validation sets should sum up to 1")
        data['rand_split'] = np.random.randn(0, len(data))
        randint.rand(0, len(data))
        if (percentage_train + percentage_test) >= 1:
            raise Error # Complete raise error implementation.
        else:
            percentage_val = 1 - (pecentage_train + percentage_test)

            self._test = self.data['rand_split' == len(data)]
        pass

    def set_test(self, index, value):
        '''
        Train the model with the provided training data
        '''
        self.hard_test = True
        self.test = self.data[index == value]

    def split_data_time(self, index, percentage_test, percentage_train):
        '''
        Split the data by entire years.
        '''

        self._train = self.bup

class Features(data):
    def __init__(self, data):
        self.Features = []

    def tokenize(data):
        tokens = utils.tokenize.remove_char().remove_stopw().lemmatize()

    def fname(arg):
        pass

    def fname(arg):
        pass
