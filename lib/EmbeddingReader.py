import numpy as np
import pickle


class EmbeddingReader(object):
    """
    construct embedding matrix and pickle them.
    """
    def __init__(self):
        self.dir = "../resource/"
        self.WORD_EMBEDDING_DIM = 200
        self.ENTITY_EMBEDDING_DIM = 50

        self.word_embedding_file = self.dir + "dim200vecs"

    def trim_word_embedding(self, word_num=20000, word_index={}):
        rf = open(self.word_embedding_file, 'r', encoding='utf-8')

        embeddings_index = {}

        for line in rf:
            values = line.split()
            index = len(values) - self.WORD_EMBEDDING_DIM
            if len(values) > (self.WORD_EMBEDDING_DIM + 1):
                word = ""
                for i in range(len(values) - self.WORD_EMBEDDING_DIM):
                    word += values[i] + " "
                word = word.strip()
            else:
                word = values[0]
            # print(line)
            coefs = np.asarray(values[index:], dtype='float32')
            embeddings_index[word] = coefs

        rf.close()

        num_words = min(word_num, len(word_index))
        print("word num: " + str(num_words))
        embedding_matrix = np.zeros((num_words + 2, self.WORD_EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= word_num:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        wf = open(self.dir + "embedding_matrix.pk", 'wb')
        pickle.dump(embedding_matrix, wf)
        wf.close()

    def read_ids(self, file):
        rf = open("../example/" + file, 'rb')
        return pickle.load(rf)

    def load_embedding_file(self, file):
        rf = open(self.dir + file, 'r', encoding='utf-8')
        entity_type_matrix = []
        while True:
            line = rf.readline()
            if line == "":
                break
            temp = line.strip().split()
            for i in range(len(temp)):
                temp[i] = float(temp[i])
            entity_type_matrix.append(temp)
        rf.close()
        entity_type_matrix = np.array(entity_type_matrix)

        wf = open(self.dir + file.split(".")[0] + ".pk", 'wb')
        pickle.dump(entity_type_matrix, wf)
        wf.close()
        return entity_type_matrix


if __name__ == '__main__':
    e = EmbeddingReader()
    word_index = e.read_ids("word_index.pk")

    e.trim_word_embedding(word_index=word_index)

    entity_embedding_file = "entity_type_matrix.txt"
    e.load_embedding_file(entity_embedding_file)
