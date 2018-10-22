import pickle
from sklearn.model_selection import train_test_split


class BaseModel:
    def __init__(self):

        self.model = None
        self.max_len = 100
        self.num_words = 6855

        self.embedding_matrix = None
        self.embedding_trainable = False
        self.EMBEDDING_DIM = 200

        self.train_word_inputs, self.train_entity_inputs, self.train_labels = self.load_data(train=True)
        self.test_word_inputs, self.test_entity_inputs, self.test_labels = self.load_data(train=False)
        self.dev_word_inputs, self.dev_entity_inputs, self.dev_labels = [None, None, None]

        self.split_train_set(rate=0.2)

        self.save_dir = "../saved_models/"
        self.dir = "../example/"
        self.embedding_dir = "../resource/embedding_matrix.pk"

        self.embedding_matrix = self.load_embeddings()

    def build_model(self):
        pass

    def train_model(self):
        pass

    def predict(self):
        pass

    def save_model(self):
        pass

    def load_data(self, train=True):
        if train:
            path = self.dir + "train_"
        else:
            path = self.dir + "test_"

        rf = open(path + "input.pk", 'rb')
        word_inputs = pickle.load(rf)
        rf.close()

        rf = open(path + "entity_inputs.pk", 'rb')
        entity_inputs = pickle.load(rf)
        rf.close()

        rf = open(path + "labels.pk", 'rb')
        labels = pickle.load(rf)
        rf.close()

        return word_inputs, entity_inputs, labels

    def load_embeddings(self):
        rf = open(self.embedding_dir, 'rb')
        embedding_matrix = pickle.load(rf)
        rf.close()
        return embedding_matrix

    def split_train_set(self, rate=0.2):
        self.train_word_inputs, \
        self.dev_word_inputs, \
        self.train_entity_inputs, \
        self.dev_entity_inputs, \
        self.train_labels, \
        self.dev_labels = train_test_split(self.train_word_inputs,
                                           self.train_entity_inputs,
                                           self.train_labels,
                                           test_size=rate,
                                           random_state=0)
