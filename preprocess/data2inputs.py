from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle


class Data2Inputs(object):

    def __init__(self):
        self.dir = "../example/"
        self.max_len = 125

    def get_data(self, train=True):
        inputs, labels, entity_labels, deps = self.read_instance_files(train)
        return inputs, labels, entity_labels, deps

    def read_instance_files(self, train=True):
        """
        read preprocessed data from files.
        here we read token, entity type and dependency clues.
        """
        if train:
            output_file = self.dir + "train_"
        else:
            output_file = self.dir + "test_"

        rf1 = open(output_file + "token.txt", 'r', encoding='utf-8')
        rf2 = open(output_file + "label.txt", 'r', encoding='utf-8')
        rf3 = open(output_file + "entity_type.txt", 'r', encoding='utf-8')
        rf4 = open(output_file + "dep.txt", 'r', encoding='utf-8')

        inputs = []
        labels = []
        entity_labels = []
        deps = []

        while True:
            line = rf1.readline()
            if line == "":
                break
            inputs.append(line)
            line = rf2.readline()
            labels.append(line)
            line = rf3.readline()
            entity_labels.append(line)
            line = rf4.readline()
            deps.append(line)

        rf1.close()
        rf2.close()
        rf3.close()
        rf4.close()

        return inputs, labels, entity_labels, deps

    def convert_text_to_index(self, train_inputs, test_inputs):
        print("start convert text to index.")
        tk = Tokenizer(num_words=10000, filters="", split=" ")
        tk.fit_on_texts(train_inputs)
        tk.fit_on_texts(test_inputs)
        train_inputs = tk.texts_to_sequences(train_inputs)
        test_inputs = tk.texts_to_sequences(test_inputs)
        print("finish!")
        self.write_ids(tk.word_index, "word_index.pk")
        self.write_word_inputs(self.pad_inputs(train_inputs))
        self.write_word_inputs(self.pad_inputs(test_inputs), False)
        return train_inputs, test_inputs, tk.word_index

    def pad_inputs(self, inputs, length=125):
        return pad_sequences(inputs, maxlen=length, padding='post')

    def write_ids(self, ids={}, file=""):
        wf = open(self.dir + file, 'wb')
        pickle.dump(ids, wf)
        wf.close()

    def write_word_inputs(self, inputs=[], train=True):
        if train:
            output_file = self.dir + "train_input.txt"
        else:
            output_file = self.dir + "test_input.txt"

        wf = open(output_file, 'w', encoding='utf-8')
        for sentence in inputs:
            for index in sentence:
                wf.write(str(index) + " ")
            wf.write("\n")
        wf.close()


if __name__ == '__main__':
    d = Data2Inputs()
    train_inputs, _, _, _ = d.get_data()
    test_inputs, _, _, _ = d.get_data(train=False)
    d.convert_text_to_index(train_inputs=train_inputs, test_inputs=test_inputs)

