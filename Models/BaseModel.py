import pickle

class BaseModel:

    def __init__(self):

        self.model = None
        self.max_len = 100
        self.num_words = 6855

        self.embedding_matrix = None
        self.embedding_trainable = False
        self.EMBEDDING_DIM = 200

        self.save_dir = "../saved_models/"
        self.dir = "../example/"

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
