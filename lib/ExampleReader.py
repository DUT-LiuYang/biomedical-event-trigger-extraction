from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle


class ExampleReader(object):
    def __init__(self, max_len, train):
        self.dir = "../example/"
        self.max_len = max_len
        self.train = train

        if train:
            self.output_file = self.dir + "train_"
        else:
            self.output_file = self.dir + "test_"

    def get_data(self):
        inputs, labels, entity_labels, deps = self.read_instance_files()
        return inputs, labels, entity_labels, deps

    def read_instance_files(self):
        """
        read preprocessed data from files.
        here we read token, entity type and dependency clues.
        """

        rf1 = open(self.output_file + "input.txt", 'r', encoding='utf-8')
        rf2 = open(self.output_file + "label.txt", 'r', encoding='utf-8')
        rf3 = open(self.output_file + "entity_type.txt", 'r', encoding='utf-8')
        rf4 = open(self.output_file + "dep.txt", 'r', encoding='utf-8')

        inputs = []
        labels = []
        entity_labels = []
        deps = []

        while True:
            line = rf1.readline()
            if line == "":
                break
            # print(line)
            line = line.strip("\n").strip().split(" ")
            inputs.append([])

            for index in line:
                inputs[-1].append(int(index))

            line = rf2.readline().strip("\n").strip()
            labels.append(line)
            line = rf3.readline().strip("\n").strip()
            entity_labels.append(line)
            line = rf4.readline().strip("\n").strip()
            deps.append(line)

        rf1.close()
        rf2.close()
        rf3.close()
        rf4.close()

        inputs = np.array(inputs)
        wf = open(self.output_file + "input.pk", 'wb')
        pickle.dump(inputs, wf)
        wf.close()

        return inputs, labels, entity_labels, deps

    def read_ids(self, file):
        rf = open(self.dir + file, 'r', encoding='utf-8')
        ids = {}
        anti_ids = {}
        while True:
            line = rf.readline().strip("\n")
            if line == "":
                break
            line = line.split("\t")

            ids[line[0]] = int(line[1])
            anti_ids[int(line[1])] = line[0]
        rf.close()
        return ids, anti_ids

    def get_label(self, ori_label=[], class_ids={}, class_num=71):

        labels = []

        for i in range(len(ori_label)):
            ori_label[i] = ori_label[i].split()
            labels.append([])
            for j in range(len(ori_label[i])):
                k = int(class_ids.get(ori_label[i][j]))
                labels[i].append(k)

        k = int(class_ids.get("O"))
        labels = pad_sequences(labels, maxlen=self.max_len, value=k, padding='post')

        res = []

        for i in range(len(labels)):
            res.append([])
            for j in range(len(labels[i])):
                k = int(labels[i][j])
                res[i].append([])
                for x in range(class_num):
                    res[i][j].append(0)
                res[i][j][k] = 1

        res = np.array(res)
        wf = open(self.output_file + "labels.pk", 'wb')
        pickle.dump(res, wf)
        wf.close()

        return res

    def get_entity_input(self, ori_label=[], class_ids={}):

        labels = []

        for i in range(len(ori_label)):
            ori_label[i] = ori_label[i].split()
            labels.append([])
            for j in range(len(ori_label[i])):
                k = int(class_ids.get(ori_label[i][j])) + 1
                labels[i].append(k)

        k = int(class_ids.get("O"))
        labels = pad_sequences(labels, maxlen=self.max_len, value=k, padding='post')

        labels = np.array(labels)
        wf = open(self.output_file + "entity_inputs.pk", 'wb')
        pickle.dump(labels, wf)
        wf.close()

        return labels


if __name__ == '__main__':
    e = ExampleReader(max_len=125, train=True)

    tri_ids_file = "tri_ids.txt"
    tri_class_id, tri_index_ids = e.read_ids(tri_ids_file)

    entity_ids_file = "entity_ids.txt"
    entity_class_ids, _ = e.read_ids(entity_ids_file)

    train_inputs, train_labels, train_entity_labels, train_deps = e.get_data()
    train_labels = e.get_label(train_labels, tri_class_id, class_num=73)
    train_entity_inputs = e.get_entity_input(train_entity_labels, entity_class_ids)

    e = ExampleReader(max_len=125, train=False)
    test_inputs, test_labels, test_entity_labels, test_deps = e.get_data()
    test_labels = e.get_label(test_labels, tri_class_id, class_num=73)
    test_entity_inputs = e.get_entity_input(test_entity_labels, entity_class_ids)

    wf = open("../example/tri_index_ids.pk", 'wb')
    pickle.dump(tri_index_ids, wf)
    wf.close()
