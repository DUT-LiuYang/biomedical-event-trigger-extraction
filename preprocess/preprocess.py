try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class PreProcessor(object):
    """
    read the origin xml and extract related clues.
    write the processed results to files.
    """
    def __init__(self):
        self.all_tri_type = ["Regulation", "Cell_proliferation", "Gene_expression", "Binding", "Positive_regulation",
                             "Transcription", "Dephosphorylation", "Development", "Blood_vessel_development",
                             "Catabolism", "Negative_regulation", "Remodeling", "Breakdown", "Localization",
                             "Synthesis", "Death", "Planned_process", "Growth", "Phosphorylation"]
        self.dir = "../resource/"
        self.output_dir = "../example/"

        self.trigger_class_ids = {}
        self.count_trigger_class = 0

        self.entity_class_ids = {}
        self.count_entity_class = 0

        self.dep_class_ids = {}
        self.count_dep_class = 0

        self.special_char = ["-", "[", "(", ")", "]", "&gt;", "/"]

        self.non_find = []
        rf = open(self.dir + "non_find", 'r', encoding='utf-8')
        while True:
            line = rf.readline()
            if line == "":
                break
            line = line[0:len(line)-1]
            self.non_find.append(line)
        rf.close()

    def read_file(self, xml="", train=True):

        if train:
            output_file = self.output_dir + "train_"
        else:
            output_file = self.output_dir + "test_"

        wf1 = open(output_file + "token.txt", 'w', encoding='utf-8')
        wf2 = open(output_file + "dep.txt", 'w', encoding='utf-8')
        wf3 = open(output_file + "label.txt", 'w', encoding='utf-8')
        wf4 = open(output_file + "entity_type.txt", 'w', encoding='utf-8')
        wf5 = open(output_file + "interaction.txt", 'w', encoding='utf-8')
        wf6 = open(output_file + "offset_id.txt", 'w', encoding='utf-8')

        tree = ET.parse(self.dir + xml)
        root = tree.getroot()

        max_len = 0

        for document in root:

            for sentence in document:
                line = ""   # used for appending tokens.
                entities = sentence.findall("entity")
                # ============ tri info===============
                tri_ids = []
                tri_offsets = []
                tri_types = []
                tri_texts = []

                all_train_triggers = []

                wrong = 0
                duplicated_tri = {}  # aim at getting attention of triggers with different ids
                duplicated_tri_ids = {}
                count_duplicated = 0
                wf7 = open(output_file + 'duplicated.txt', 'w', encoding='utf-8')
                # =========== entity info============
                entity_ids = []
                entity_offsets = []
                entity_types = []
                entity_texts = []

                if entities is not None:
                    for entity in entities:
                        if entity.get("BANNER") is not None:
                            continue

                        char_offset = entity.get("charOffset")
                        eid = entity.get("id")
                        e_type = entity.get("type")
                        head_offset = entity.get("headOffset")
                        text = entity.get("text")

                        s1 = int(char_offset.split("-")[0])
                        s2 = int(head_offset.split("-")[0])
                        e1 = int(char_offset.split("-")[1])
                        e2 = int(head_offset.split("-")[1])

                        if head_offset != char_offset and s1 >= s2 and e1 <= e2:
                            char_offset = head_offset

                        if entity.get('given') is not None:
                            entity_ids.append(eid)
                            entity_offsets.append(char_offset)
                            entity_types.append(e_type)
                            entity_texts.append(text)
                        else:
                            if char_offset not in tri_offsets:
                                if len(text.split(" ")) > 1:    # check whether there are triggers containing multiple words.
                                    wrong += 1

                                tri_ids.append(eid)
                                tri_offsets.append(char_offset)
                                tri_types.append(e_type)
                                tri_texts.append(text)
                                if train:
                                    all_train_triggers.append(tri_texts)
                            else:
                                index0 = tri_offsets.index(char_offset)
                                duplicated_tri_ids[eid] = tri_ids[index0]
                                if char_offset in duplicated_tri.keys():
                                    duplicated_tri[char_offset] += "#" + eid
                                else:
                                    duplicated_tri[char_offset] = eid
                                count_duplicated += 1

                interactions = sentence.findall("interaction")
                interaction_e1 = ""
                interaction_e2 = ""
                interaction_type = ""
                if interactions is not None:
                    for interaction in interactions:
                        event = interaction.get("event")
                        if event != 'True':
                            continue
                        e1 = interaction.get("e1")
                        e2 = interaction.get("e2")
                        inter_type = interaction.get("type")
                        interaction_e1 += str(e1) + " "
                        interaction_e2 += str(e2) + " "
                        interaction_type += str(inter_type) + " "

                pos = ""
                char_offsets = ""
                count = 0

                for token in sentence.find("analyses").find("tokenization"):
                    text = token.get("text")
                    char_offset = token.get("charOffset")
                    start = int(char_offset.split("-")[0])
                    end = int(char_offset.split("-")[1])

                    tok_pos = token.get("POS")
                    if len(text) > 1:
                        for sc in self.special_char:
                            text = text.strip(sc)

                    if train and text in self.non_find and '-' in text:
                        if text in tri_texts:
                            texts = text.split("-")
                            for i, word in enumerate(texts):
                                line += word + " "
                                pos += tok_pos + " "

                                if i == len(texts) - 1:
                                    char_offsets += str(start) + "-" + str(end) + " "
                                else:
                                    char_offsets += str(start) + "-" + str(start + len(word)) + " "
                                    start = start + len(word)
                                count += 1
                        else:
                            line += text + " "
                            char_offsets += char_offset + " "
                            pos += tok_pos + " "
                            count += 1
                    elif not train and text in self.non_find and '-' in text:
                        if text in all_train_triggers:
                            texts = text.split("-")
                            for i, word in enumerate(texts):
                                line += word + " "
                                pos += tok_pos + " "
                                # note that I get rid of '-'.
                                if i == len(texts) - 1:
                                    char_offsets += str(start) + "-" + str(end) + " "
                                else:
                                    char_offsets += str(start) + "-" + str(start + len(word)) + " "
                                    start = start + len(word)
                                count += 1
                        else:
                            line += text + " "
                            char_offsets += char_offset + " "
                            pos += tok_pos + " "
                            count += 1
                    else:
                        line += text + " "
                        char_offsets += char_offset + " "
                        pos += tok_pos + " "
                        count += 1

                dep_info = ""
                for dependency in sentence.find("analyses").find("parse"):
                    if dependency.tag == "dependency":
                        # print(dependency.get("id"))
                        dep_info += dependency.get("t1")[3:] + "#"
                        dep_info += dependency.get("t2")[3:] + "#"
                        dep_type = dependency.get("type")
                        dep_info += dep_type + " "

                line = line.strip()
                char_offsets = char_offsets.strip()
                pos = pos.strip()

                wf1.write(line + "\n")

                if count > max_len:
                    max_len = count

                tri_offsets, tri_types, tri_ids = self.sort_offset(tri_offsets, tri_types, tri_ids)
                entity_offsets, entity_types, entity_ids = self.sort_offset(entity_offsets, entity_types, entity_ids)
                label_temp, self.count_trigger_class = self.get_label_type(char_offsets, tri_offsets, tri_types,
                                                                           self.count_trigger_class,
                                                                           self.trigger_class_ids)
                label_temp = label_temp.strip()
                wf3.write(label_temp + "\n")

                label_temp, self.count_entity_class = self.get_label_type(char_offsets, entity_offsets, entity_types,
                                                                          self.count_entity_class,
                                                                          self.entity_class_ids)
                label_temp = label_temp.strip()
                wf4.write(label_temp + "\n")
                wf2.write(dep_info + "\n")

                wf5.write(interaction_e1.strip() + "#" + interaction_e2.strip() +
                          "#" + interaction_type.strip() + "\n")
                wf6.write(char_offsets + "#" + self.list2str(tri_offsets) + "#" +
                          self.list2str(tri_ids) + "#" + self.list2str(entity_offsets) +
                          "#" + self.list2str(entity_ids) + "\n")

        # for key, value in duplicated_tri.items():
        #     wf7.write(key + "*" + value + "\n")
        # wf7.close()

        wf1.close()
        wf2.close()
        wf3.close()
        wf4.close()
        wf5.close()
        wf6.close()
        wf7.close()

    def sort_offset(self, tri_offsets=[], tri_types=[], tri_ids=[]):
        temp = []
        for i in range(len(tri_offsets)):
            # print(tri_offsets[i])
            temp.append(int(tri_offsets[i].split("-")[0]))
        num = len(temp)

        for i in range(num - 1):
            for j in range(num - i - 1):
                if temp[j] > temp[j + 1]:
                    temp[j], temp[j + 1] = temp[j + 1], temp[j]
                    tri_offsets[j], tri_offsets[j+1] = tri_offsets[j+1], tri_offsets[j]
                    tri_types[j], tri_types[j+1] = tri_types[j+1], tri_types[j]
                    tri_ids[j], tri_ids[j+1] = tri_ids[j+1], tri_ids[j]

        return tri_offsets, tri_types, tri_ids

    def list2str(self, info_list=[]):
        res = ""
        for i in range(len(info_list)):
            res += str(info_list[i]) + " "
        return res.strip()

    def get_label_type(self, char_offsets, offsets, types, num, class_ids):
        char_offsets = char_offsets.split()
        label = ""
        j = 0
        signal = False

        if len(offsets) == 0:
            for i in range(len(char_offsets)):
                if "O" not in class_ids:
                    class_ids["O"] = num
                    num += 1
                label += "O "
            return label, num

        for i in range(len(char_offsets)):
            if j < len(offsets):
                s1 = int(char_offsets[i].split("-")[0])
                e1 = int(char_offsets[i].split("-")[1])
                s2 = int(offsets[j].split("-")[0])
                e2 = int(offsets[j].split("-")[1])
                if s1 >= s2 and e1 <= e2:
                    if signal and e1 == e2:
                        label_type = "E-" + types[j]
                        j += 1
                        signal = False
                    elif signal:
                        label_type = "I-" + types[j]
                    elif e1 == e2 and s1 == s2:
                        label_type = "S-" + types[j]
                        j += 1
                    else:
                        label_type = "B-" + types[j]
                        signal = True
                    label += " " + label_type
                    if label_type not in class_ids:
                        # print(label_type)
                        class_ids[label_type] = num
                        num += 1
                else:
                    label += " O"
                    if "O" not in class_ids:
                        class_ids["O"] = num
                        num += 1
            else:
                label += " O"
                if "O" not in class_ids:
                    class_ids["O"] = num
                    num += 1
        return label, num


if __name__ == '__main__':
    train_file = "MLEE-test-train-preprocessed.xml"
    p = PreProcessor()
    p.read_file(xml=train_file, train=True)
