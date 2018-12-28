from Models.BaseModel import BaseModel
from lib.Evaluator import Evaluator
import keras
from keras.layers import Bidirectional, GRU, TimeDistributed, Dense, Dropout
from Models.util import *


class SelfAttentionModel(BaseModel):
    """
    The model in bibm 2017 paper without attention label.
    """
    def __init__(self, max_len=100, class_num=73, use_development_set=False):

        super(SelfAttentionModel, self).__init__(use_development_set)

        self.max_len = max_len
        self.class_num = class_num
        self.name = "SA"

    def build_model(self):
        x = TimeDistributed(Dense(200, activation='tanh'))(self.sen_embedding)
        self_attention_embedding, _ = soft_attention_alignment(x, x)

        inputs = keras.layers.concatenate([self.sen_embedding, self.entity_embedding])

        encoded_sentence_embedding = Bidirectional(GRU(200,
                                                       activation="relu",
                                                       return_sequences=True,
                                                       recurrent_dropout=0.3,
                                                       dropout=0.3))(inputs)
        # encoded_sentence_embedding = Dropout(rate=0.5)(encoded_sentence_embedding)

        x = Concatenate()([encoded_sentence_embedding, self_attention_embedding])
        x = Dropout(rate=0.5)(x)

        predictions = TimeDistributed(Dense(self.class_num, activation='softmax'))(x)

        return predictions

    def train_model(self, max_epoch=30):

        e1 = Evaluator(true_labels=self.test_labels, sentences=self.test_word_inputs, index_ids=self.index_ids, val=False)
        if self.use_development_set:
            e2 = Evaluator(true_labels=self.dev_labels, sentences=self.dev_word_inputs, index_ids=self.index_ids)
        log = open("../log/" + self.name + ".txt", 'a+', encoding='utf-8')
        for i in range(max_epoch):
            print("====== epoch " + str(i + 1) + " ======")

            # -------- if use development set, the early stop will be introduced.
            if self.use_development_set:

                self.model.fit({'sentence_input': self.train_word_inputs,
                                'entity_type_input': self.train_entity_inputs},
                               self.train_labels,
                               epochs=1,
                               batch_size=32,
                               validation_data=([self.dev_word_inputs,
                                                 self.dev_entity_inputs], self.dev_labels),
                               verbose=2)

                print("# -- develop set --- #")
                results = self.model.predict({'sentence_input': self.dev_word_inputs,
                                              'entity_type_input': self.dev_entity_inputs},
                                             batch_size=64,
                                             verbose=0)
                results = e2.get_true_label(label=results)
                results = e2.process_bie(sen_label=results)
                f1, _, _ = e2.get_true_prf(results, epoch=i + 1)
                if f1 < 0:
                    break
            else:
                self.model.fit({'sentence_input': self.train_word_inputs,
                                'entity_type_input': self.train_entity_inputs},
                               self.train_labels,
                               epochs=1,
                               batch_size=32,
                               verbose=2)

            print("# -- test set --- #")
            results = self.model.predict({'sentence_input': self.test_word_inputs,
                                          'entity_type_input': self.test_entity_inputs},
                                         batch_size=64,
                                         verbose=0)

            results = e1.get_true_label(label=results)
            results = e1.process_bie(sen_label=results)
            f1, p1, r1 = e1.get_true_prf(results, epoch=i + 1)
            log.write("epoch:{} p:{} r:{} f:{}\n".format(i + 1, p1, r1, f1))
        log.close()


if __name__ == '__main__':

    s = SelfAttentionModel(max_len=125, class_num=73)
    for i in range(5):
        s.compile_model()
        s.train_model(max_epoch=45)
