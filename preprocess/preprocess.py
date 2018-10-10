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

    def read_file(self, xml="", train=True):

        if train:
            output_file = self.output_dir + "train_"
        else:
            output_file = self.output_dir + "test_"

        wf1 = open(output_file + "index.txt", 'w', encoding='utf-8')
        wf2 = open(output_file + "dep.txt", 'w', encoding='utf-8')
        wf3 = open(output_file + "label.txt", 'w', encoding='utf-8')
        wf4 = open(output_file + "entity_type.txt", 'w', encoding='utf-8')
        wf5 = open(output_file + "interaction.txt", 'w', encoding='utf-8')
        wf6 = open(output_file + "offset_id.txt", 'w', encoding='utf-8')

        tree = ET.parse(self.dir + xml)
        root = tree.getroot()
        for document in root:
            print(document.tag, ":", document.attrib)
            for sentence in document:
                line = ""   # used for appending tokens.

                entities = sentence.find("entity")
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

                        if entity.find("BANNER") is not None:
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

                        if entity.get('given') is None:
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

                interactions = sentence.find("interaction")
                if interactions is not None:
                    interaction_e1 = ""
                    interaction_e2 = ""
                    interaction_type = ""
                    for interaction in interactions:
                        e1 = interaction.get("e1")
                        e2 = interaction.get("e2")
                        inter_type = interaction.get("type")
                        interaction_e1 += str(e1) + " "
                        interaction_e2 += str(e2) + " "
                        interaction_type += str(inter_type) + " "

        wf1.close()
        wf2.close()
        wf3.close()
        wf4.close()
        wf5.close()
        wf6.close()
        wf7.close()


if __name__ == '__main__':
    train_file = "MLEE-test-train-preprocessed.xml"
    p = PreProcessor()
    p.read_file(xml=train_file, train=True)
