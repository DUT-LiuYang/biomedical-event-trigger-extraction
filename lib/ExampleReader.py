from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os


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

    def get_attention_label(self,
                            charoffset_file="",
                            interaction_info_file="",
                            max_len=125,):

        duplicated_dict = self.get_duplicated_dict()

        attention_label = []

        token_offsets = []
        tri_offsets = []
        entity_offsets = []
        tri_ids = []
        entity_ids = []

        interaction_e1 = []
        interaction_e2 = []
        interaction_type = []

        rf = open(self.dir + charoffset_file, 'r', encoding='utf-8')
        while True:
            line = rf.readline()
            if line == "":
                break
            line = line.split("#")
            token_offsets.append(line[0].strip().split())
            tri_offsets.append(line[1].strip().split())
            tri_ids.append((line[2].strip().split()))
            entity_offsets.append((line[3].strip().split()))
            entity_ids.append(line[4].strip().split())
        rf.close()

        print(np.array(token_offsets).shape)

        rf = open(self.dir + interaction_info_file, 'r', encoding='utf-8')
        while True:
            line = rf.readline()
            if line == "":
                break
            line = line.split("#")
            interaction_e1.append(line[0].strip().split())
            interaction_e2.append(line[1].strip().split())
            interaction_type.append(line[2].strip().split())
        rf.close()

        # construct attention label. [sen_num, max_len, max_len - 1]
        sen_num = len(interaction_type)
        for i in range(sen_num):
            attention_label.append([])

            # get the trigger labels sequence
            tri_id_label, tri_offsets_ids = self.get_id_label_type(char_offsets=token_offsets[i], offsets=tri_offsets[i], ids=tri_ids[i])
            tri_id_label = tri_id_label.strip().split()
            # get the entity labels sequence
            entity_id_label, _ = self.get_id_label_type(char_offsets=token_offsets[i], offsets=entity_offsets[i], ids=entity_ids[i])
            entity_id_label = entity_id_label.strip().split()

            # get the num of the entities and triggers in the current sentence.
            tri_attention_label, num1 = self.get_average_attention(tri_id_label, tri_ids[i])
            entity_attention_label, num2 = self.get_average_attention(entity_id_label, entity_ids[i])
            num = num1 + num2
            if num == 0:
                ave_score = 0
            else:
                ave_score = 1.0 / num

            for j in range(max_len):
                attention_label[i].append([])
                if j >= len(token_offsets[i]):
                    attention_label[i][j] = [0] * max_len  # the padding part gets 0 attention.
                elif tri_offsets_ids.get(token_offsets[i][j]) is None:

                    for k in range(max_len):
                        # the padding part and the current word get 0 attention.
                        if k >= len(tri_attention_label) or j == k:
                            attention_label[i][j].append(0)
                        elif tri_attention_label[k] == 0 and entity_attention_label[k] == 0:
                                attention_label[i][j].append(0)
                        else:
                            # print(str(score))
                            attention_label[i][j].append(ave_score)
                else:
                    current_id = tri_offsets_ids.get(token_offsets[i][j])
                    tri_attention_label, num1 = self.get_attention_label_type(tri_id_label, current_id, interaction_e1[i], interaction_e2[i], duplicated_dict, token_offsets[i][j])
                    entity_attention_label, num2 = self.get_attention_label_type(entity_id_label, current_id, interaction_e1[i], interaction_e2[i], duplicated_dict, token_offsets[i][j])
                    num = num1 + num2
                    # if this sentence doesn't contain a entity or trigger except the current entity.
                    if num == 0:
                        score = 0
                    else:
                        score = 1.0 / num
                    for k in range(max_len):
                        if k >= len(tri_attention_label) or j == k:
                            attention_label[i][j].append(0)
                        elif tri_attention_label[k] == 0 and entity_attention_label[k] == 0:
                            attention_label[i][j].append(0)
                        else:
                            # print(str(score))
                            attention_label[i][j].append(score)

        wf = open(self.output_file + "attention_label.pk", 'wb')
        pickle.dump(np.asarray(attention_label, dtype='float32'), wf)
        wf.close()

        return attention_label

    def get_id_label_type(self, char_offsets, offsets, ids):
        label = ""
        j = 0
        signal = False
        offset_ids = {}

        if len(offsets) == 0:
            for i in range(len(char_offsets)):
                label += "O "
            return label, offset_ids

        for i in range(len(char_offsets)):
            if j < len(offsets):
                s1 = int(char_offsets[i].split("-")[0])
                e1 = int(char_offsets[i].split("-")[1])
                s2 = int(offsets[j].split("-")[0])
                e2 = int(offsets[j].split("-")[1])
                if s1 >= s2 and e1 <= e2:
                    if signal and e1 == e2:
                        label_type = "E-" + ids[j]
                        offset_ids[char_offsets[i]] = ids[j]
                        j += 1
                        signal = False
                    elif signal:
                        label_type = "I-" + ids[j]
                        offset_ids[char_offsets[i]] = ids[j]
                    elif e1 == e2 and s1 == s2:
                        label_type = "S-" + ids[j]
                        offset_ids[char_offsets[i]] = ids[j]
                        j += 1
                    else:
                        label_type = "B-" + ids[j]
                        offset_ids[char_offsets[i]] = ids[j]
                        signal = True
                    label += " " + label_type
                else:
                    label += " O"
            else:
                label += " O"
        # print(label)
        return label, offset_ids

    # however, entity that consists of several words need one attention?
    def get_average_attention(self, id_label=[], interaction_e2=[]):
        attention_label = []
        num = 0
        for i in range(len(id_label)):
            attention_label.append(0)
            for j in range(len(interaction_e2)):
                if interaction_e2[j] == id_label[i][2:]:
                    num += 1
                    attention_label[i] = 1
                    break
        return attention_label, num

    def get_attention_label_type(self, id_label=[], current_id="", interaction_e1=[], interaction_e2=[], duplicated_dict={}, offset=""):
        attention_label = []
        num = 0
        # finish origin labeling
        for i in range(len(id_label)):
            attention_label.append(0)
            for j in range(len(interaction_e2)):
                # print(current_id + "-" + interaction_e1[j] + "-" + interaction_e2[j] + "-" + id_label[i][2:])
                if current_id == interaction_e1[j] and interaction_e2[j] == id_label[i][2:]:
                    num += 1
                    attention_label[i] = 1
                    break

        # if the trigger has brothers with different ids...
        if offset in duplicated_dict.keys():
            duplicated_ids = duplicated_dict[offset].strip().split("#")
            for ids in duplicated_ids:
                for i in range(len(id_label)):
                    for j in range(len(interaction_e2)):
                        # print(current_id + "-" + interaction_e1[j] + "-" + interaction_e2[j] + "-" + id_label[i][2:])
                        if ids == interaction_e1[j] and interaction_e2[j] == id_label[i][2:]:
                            if attention_label[i] == 0:
                                num += 1
                            attention_label[i] = 1
                            break

        # print("===" + str(id_label))
        # print("+++" + current_id + " " + str(num))
        return attention_label, num

    def get_duplicated_dict(self):
        rf = open(self.output_file + "duplicated.txt", 'r', encoding='utf-8')
        duplicated_dict = {}
        while True:
            line = rf.readline()
            if line == "":
                break
            line = line.split("*")
            key = line[0]
            value = line[1]
            duplicated_dict[key] = value
        rf.close()
        return duplicated_dict

    def each_file(self, file_path=""):
        """
        Get the doc ids of development set files.
        :param file_path: the dir of the raw development files.
        :return:
        """
        file_list = os.listdir(file_path)
        doc_ids = []
        for file in file_list:
            if ".a1" in file:
                doc_ids.append(file.split(".")[0])
        wf = open(self.dir + "../example/development_doc_ids.pk", 'wb')
        pickle.dump(doc_ids, wf)
        wf.close()
        return doc_ids


if __name__ == '__main__':
    e = ExampleReader(max_len=125, train=True)
    e.each_file("../resource/MLEE-1.0.2-rev1/standoff/development/test")
    tri_ids_file = "tri_ids.txt"
    tri_class_id, tri_index_ids = e.read_ids(tri_ids_file)

    entity_ids_file = "entity_ids.txt"
    entity_class_ids, _ = e.read_ids(entity_ids_file)

    train_inputs, train_labels, train_entity_labels, train_deps = e.get_data()
    train_labels = e.get_label(train_labels, tri_class_id, class_num=73)
    train_entity_inputs = e.get_entity_input(train_entity_labels, entity_class_ids)

    train_attention_label = e.get_attention_label(charoffset_file="train_offset_id.txt",
                                                  interaction_info_file="train_interaction.txt")

    e = ExampleReader(max_len=125, train=False)
    test_inputs, test_labels, test_entity_labels, test_deps = e.get_data()
    test_labels = e.get_label(test_labels, tri_class_id, class_num=73)
    test_entity_inputs = e.get_entity_input(test_entity_labels, entity_class_ids)

    wf = open("../example/tri_index_ids.pk", 'wb')
    pickle.dump(tri_index_ids, wf)
    wf.close()

    test_attention_label = e.get_attention_label(charoffset_file="test_offset_id.txt",
                                                 interaction_info_file="test_interaction.txt")
