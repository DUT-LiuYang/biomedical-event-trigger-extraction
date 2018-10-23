from Models.BaseModel import BaseModel
from lib.Evaluator import Evaluator
import keras
from keras.layers import Input, Embedding, Bidirectional, GRU, TimeDistributed, Dense, Dropout
from keras.models import Model


class SelfAttentionModel(BaseModel):

    def __init__(self, max_len=100, class_num=73):

        super(SelfAttentionModel, self).__init__()

        self.max_len = max_len
        self.class_num = class_num

    def build_model(self):
        sentences = Input(shape=(self.max_len,), dtype='int32', name='sentence_input')
        entity_types = Input(shape=(self.max_len,), dtype='int32', name='entity_type_input')

        sentence_embedding_layer = Embedding(self.num_words + 2,
                                             self.EMBEDDING_DIM,
                                             weights=[self.embedding_matrix],
                                             input_length=self.max_len,
                                             trainable=False,
                                             mask_zero=True)
        sentence_embedding = sentence_embedding_layer(sentences)

        entity_embedding_layer = Embedding(self.entity_type_num + 2,
                                           self.ENTITY_TYPE_VEC_DIM,
                                           weights=[self.entity_embedding_matrix],
                                           input_length=self.max_len,
                                           trainable=True,
                                           mask_zero=True)
        entity_embedding = entity_embedding_layer(entity_types)

        inputs = keras.layers.concatenate([sentence_embedding, entity_embedding])

        sentence_embedding = Bidirectional(GRU(200,
                                               activation="relu",
                                               return_sequences=True,
                                               recurrent_dropout=0.3,
                                               dropout=0.3))(inputs)

        x = TimeDistributed(Dense(200, activation='tanh'))(sentence_embedding)
        # x = Dropout(rate=0.3)(x)
        predictions = TimeDistributed(Dense(self.class_num, activation='softmax'))(x)

        self.model = Model(inputs=[sentences, entity_types], outputs=predictions)
        self.model.compile(loss=['categorical_crossentropy'], optimizer='rmsprop', metrics=['accuracy'])

        return self.model

    def train_model(self, max_epoch=30):

        e = Evaluator(true_labels=self.test_labels, sentences=self.test_word_inputs, index_ids=self.index_ids)

        for i in range(max_epoch):
            print("====== epoch " + str(i + 1) + " ======")
            self.model.fit({'sentence_input': self.train_word_inputs,
                            'entity_type_input': self.train_entity_inputs},
                           self.train_labels,
                           epochs=1,
                           batch_size=32,
                           # validation_data=([self.dev_word_inputs,
                           #                   self.dev_entity_inputs], self.dev_labels),
                           verbose=2)

            results = self.model.predict({'sentence_input': self.test_word_inputs,
                                          'entity_type_input': self.test_entity_inputs},
                                         batch_size=64,
                                         verbose=0)

            results = e.get_true_label(label=results)
            results = e.process_bie(sen_label=results)
            e.get_true_prf(results, epoch=i + 1)


if __name__ == '__main__':

    s = SelfAttentionModel(max_len=125, class_num=73)
    s.build_model()
    s.train_model()

