class Evaluator(object):

    def __init__(self, true_labels=None, sentences=None, index_ids={}):
        self.max_F1 = -1
        self.max_F1_epoch = -1

        self.sentences = sentences

        self.index_ids = index_ids
        self.true_labels = self.get_true_label(true_labels)

        self.output_dir = "../results/F_"

    # convert 01 to BIO
    def get_true_label(self, label=None):
        true_label = []
        for i in range(len(label)):
            true_label.append([])
            for j in range(len(label[i])):
                max_num = -100
                index = 0
                x = 0
                for k in range(len(label[i][j])):
                    if label[i][j][k] > max_num:
                        max_num = label[i][j][k]
                        index = x
                    x += 1
                true_label[i].append(str(self.index_ids.get(index)))

        return true_label

    def process_bie(self, sen_label=None):
        b_signal = False
        x = 0
        for i in range(len(sen_label)):
            for j in range(len(sen_label[i])):
                if not b_signal and 'B-' in sen_label[i][j]:
                    b_signal = True
                    x = 1
                elif b_signal and ('I-' not in sen_label[i][j] and 'E-' not in sen_label[i][j]):
                    b_signal = False
                    for k in range(x):
                        sen_label[i][j - k - 1] = 'O'
                    x = 0
                elif b_signal and 'E-' in sen_label[i][j]:
                    b_signal = False
                    x = 0
                elif not b_signal and ('I-' in sen_label[i][j] or 'E-' in sen_label[i][j]):
                    sen_label[i][j] = 'O'
                    x = 0
                elif b_signal and 'I-' in sen_label[i][j]:
                    x += 1

        return sen_label

    def get_true_prf(self, sen_label, epoch=1):

        label = self.true_labels

        tri_type = ["Regulation", "Cell_proliferation", "Gene_expression", "Binding",
                    "Positive_regulation", "Transcription", "Dephosphorylation", "Development",
                    "Blood_vessel_development", "Catabolism", "Negative_regulation", "Remodeling",
                    "Breakdown", "Localization", "Synthesis", "Death", "Planned_process",
                    "Growth", "Phosphorylation"]

        b_signal = False

        p_e = 0  # num of positive examples
        p_es = [0] * 19

        for i in range(len(label)):
            for j in range(len(label[i])):
                if label[i][j] != 'O':
                    if 'S-' in label[i][j] and label[i][j].split("-")[1] in tri_type:
                        p_e += 1
                        index = tri_type.index(label[i][j].split("-")[1])
                        p_es[index] += 1
                    elif 'B-' in label[i][j] and label[i][j].split("-")[1] in tri_type:
                        p_e += 1
                        index = tri_type.index(label[i][j].split("-")[1])
                        p_es[index] += 1

        p_p_e = 0  # num of predicted positive examples
        p_p_es = [0] * 19

        for i in range(len(sen_label)):
            for j in range(len(sen_label[i])):
                if sen_label[i][j] != 'O':
                    if 'S-' in sen_label[i][j] and sen_label[i][j].split("-")[1] in tri_type:
                        p_p_e += 1
                        index = tri_type.index(sen_label[i][j].split("-")[1])
                        p_p_es[index] += 1
                    elif 'B-' in sen_label[i][j] and sen_label[i][j].split("-")[1] in tri_type:
                        p_p_e += 1
                        index = tri_type.index(sen_label[i][j].split("-")[1])
                        p_p_es[index] += 1

        pr_p_e = 0  # num of examples that are predicted rightly
        pr_p_es = [0] * 19

        for i in range(len(sen_label)):
            for j in range(len(sen_label[i])):
                if sen_label[i][j] != 'O' and sen_label[i][j].split("-")[1] in tri_type:
                    if 'S-' in sen_label[i][j] and sen_label[i][j] == label[i][j]:
                        pr_p_e += 1
                        index = tri_type.index(sen_label[i][j].split("-")[1])
                        pr_p_es[index] += 1
                    elif not b_signal and 'B-' in sen_label[i][j]:
                        if sen_label[i][j] == label[i][j]:
                            b_signal = True
                        else:
                            b_signal = False
                    elif 'I-' in sen_label[i][j]:
                        if b_signal:
                            if sen_label[i][j] == label[i][j]:
                                b_signal = True
                            else:
                                b_signal = False
                    elif 'E-' in sen_label[i][j]:
                        if b_signal:
                            if sen_label[i][j] == label[i][j]:
                                pr_p_e += 1
                                index = tri_type.index(sen_label[i][j].split("-")[1])
                                pr_p_es[index] += 1
                            b_signal = False

        if p_p_e == 0:
            print(str(0))
            return 0, 0, 0
        if p_e == 0:
            print(str(0))
            return 0, 0, 0
        p1 = float(pr_p_e) / p_p_e
        r1 = float(pr_p_e) / p_e
        f1 = 2 * p1 * r1 / (p1 + r1)
        print("p: " + str(p1 * 100))
        print("r: " + str(r1 * 100))
        print("f: " + str(f1 * 100))

        # for i in range(19):
        #     if p_p_es[i] == 0:
        #         print(str(0))
        #         continue
        #     if p_es[i] == 0:
        #         print(str(0))
        #         continue
        #     p = float(pr_p_es[i]) / p_p_es[i]
        #     r = float(pr_p_es[i]) / p_es[i]
        #     if p == 0 and r == 0:
        #         print(str(0))
        #         continue
        #     f = 2 * p * r / (p + r)
        #     print(tri_type[i] + " p - " + str(p) + " r - " + str(r) + " f - " + str(f))

        if f1 > self.max_F1:
            self.max_F1 = f1
            self.max_F1_epoch = epoch
            wf = open(self.output_dir + str(f1 * 100) + "_" + str(epoch) + ".txt", 'w', encoding='utf-8')
            for i in range(len(sen_label)):
                for j in range(len(sen_label[i])):
                    wf.write(sen_label[i][j] + " ")
                wf.write("\n")
            wf.close()

        if epoch % 5 == 0:
            print("max F1 achieved at epoch: " + str(self.max_F1_epoch))
            print("max F1 score is: " + str(self.max_F1))

        return f1, p1, r1
