class Model:

    def __init__(self):

        self.model = None
        self.max_len = 100

        self.embedding_matrix = None
        self.embedding_trainable = False
        self.EMBEDDING_DIM = 200

        self.save_dir = "../saved_models/"

    def build_model(self):
        pass

    def train_model(self):
        pass

    def predict(self):
        pass

    def save_model(self):
        pass
