"""
Copyright 2020 ICES, University of Manchester, Evenset Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from utils.spec_tokenizers import tokenize_fa
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Lambda, Input
from tensorflow.keras.layers import add, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os
PADWORD = "PADword"

# Code by Nikola Milosevic


class ElmoEmbeddingLayer(Layer):
    """
    ELMo embedding layer with trainable weights.
    """
    def __init__(self, elmo, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        self.elmo = elmo
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self._trainable_weights += tf.trainable_variables(
            scope="^{}_module/.*".format("elmo"))  # add trainable weights
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        result = self.elmo(inputs={
            "tokens": inputs[0],
            "sequence_len": inputs[1]
        },
            as_dict=True,
            signature='tokens',
        )['elmo']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs[0], PADWORD)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


class NER_BiLSTM_ELMo_i2b2_pad_mask(object):
    def __init__(self):
        """Implementation of initialization"""
        # load json and create model
        self.sess = tf.Session()
        K.set_session(self.sess)
        self.elmo_model = hub.Module(
            "https://tfhub.dev/google/elmo/2", trainable=True, name="{}_module".format("elmo"))
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())

        self.batch_size = 32
        self.n_tags = 9
        self.model = self.createModel()
        if os.path.exists("Models/NER_BiLSTM_ELMo_i2b2_pad_mask.h5"):
            print("Loading model")
            self.model.load_weights("Models/NER_BiLSTM_ELMo_i2b2_pad_mask.h5")
            print("Loaded model")
        self.tags = None

    def perform_NER(self, text):
        """
        Function that perform BiLSTM-based NER

        :param text: Text that should be analyzed and tagged
        :return: returns sequence of sequences with labels
        """
        # tokenize text and split into sentences
        sequences = tokenize_fa([text])
        word_sequences = [[word for word, _ in seq] for seq in sequences]
        X = pad_sequences(
            word_sequences,
            dtype=object,
            padding='post',
            value=PADWORD
        )  # pad sequences to have same length
        # do predictions
        seq_lens = np.array([len(s) for s in X])
        predictions = self.model.predict([X, seq_lens])
        index2tags = {0: 'O', 1: 'ID', 2: 'PHI', 3: 'NAME', 4: 'CONTACT',
                        5: 'DATE', 6: 'AGE', 7: 'PROFESSION', 8: 'LOCATION'}
        Y_pred_F = np.argmax(predictions, axis=2) # get class index from softmax output
        final_sequences = []
        for i, row in enumerate(X):
            length = len(word_sequences[i])  # restore original length, ignoring padding
            labels = [index2tags[i] for i in Y_pred_F[i]]
            final_sequences.append(list(zip(row[:length], labels[:length])))
        return final_sequences

    def createModel(self):

        input_text = Input(shape=(None,), dtype="string", name='tokens')
        input_len = Input(shape=[], dtype=tf.int32, name='seq_len') # sequence length
        embedding = ElmoEmbeddingLayer(
            self.elmo_model)([input_text, input_len])
        x = Bidirectional(LSTM(units=512, return_sequences=True,
                            recurrent_dropout=0.2, dropout=0.2))(embedding)
        x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                                recurrent_dropout=0.2, dropout=0.2))(x)
        x = add([x, x_rnn])  # residual connection to the first biLSTM
        out = TimeDistributed(Dense(self.n_tags, activation="softmax"))(x)
        self.model = Model(inputs=[input_text, input_len], outputs=out)
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                            metrics=["accuracy"])
        self.model.summary()
        return self.model

    def transform_sequences(self, token_sequences, max_len=50):
        """
        Transforms token sequences to padded/truncated sequences of fixed length = max_len.
        """
        X = []
        Y = []
        all_tags = []
        for tok_seq in token_sequences:
            X_seq = []
            Y_seq = []
            for i in range(0, max_len):
                try:
                    X_seq.append(tok_seq[i][0])
                    Y_seq.append(tok_seq[i][1])
                    all_tags.append(tok_seq[i][1])
                except:
                    X_seq.append(PADWORD)
                    Y_seq.append("O")
            X.append(X_seq)
            Y.append(Y_seq)
        self.n_tags = len(set(all_tags))
        self.tags = set(all_tags)
        tags2index = {'O': 0, 'ID': 1, 'PHI': 2, 'NAME': 3, 'CONTACT': 4,
                      'DATE': 5, 'AGE': 6, 'PROFESSION': 7, 'LOCATION': 8}

        Y = [[tags2index[w] for w in s] for s in Y]

        return X, Y

    def learn(self, X, Y, epochs=1):
        """
        Method for the training ELMo BiLSTM NER model
        :param X: Training sequences
        :param Y: Results of training sequences
        :param epochs: number of epochs
        :return:
        """
        first = int(np.floor(0.9*len(X)/self.batch_size))
        second = int(np.floor(0.1*len(X)/self.batch_size))
        X_tr, X_val = X[:first *
                        self.batch_size], X[-second * self.batch_size:]
        y_tr, y_val = Y[:first *
                        self.batch_size], Y[-second * self.batch_size:]
        y_tr = np.array(y_tr)
        y_val = np.array(y_val)
        y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
        y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)

        X_tr = np.array(X_tr)
        X_val = np.array(X_val)
        seq_lens_tr = np.array([len(s) for s in X_tr])
        seq_lens_val = np.array([len(s) for s in X_val])

        self.model.fit(x=[X_tr, seq_lens_tr], y=y_tr, validation_data=([X_val, seq_lens_val], y_val),
                       batch_size=self.batch_size, epochs=epochs)

    def evaluate(self, X, Y):
        """
        Function that evaluates the model and calculates precision, recall and F1-score
        :param X: sequences that should be evaluated
        :param Y: true positive predictions for evaluation
        :return: prints the table with precision,recall and f1-score
        """
        seq_lens = np.array([len(s) for s in X])
        Y_pred = self.model.predict([np.array(X), seq_lens])
        from sklearn import metrics
        index2tags = {0: 'O', 1: 'ID', 2: 'PHI', 3: 'NAME', 4: 'CONTACT',
                      5: 'DATE', 6: 'AGE', 7: 'PROFESSION', 8: 'LOCATION'}
        labels = ["O", "ID", "PHI", "NAME", "CONTACT", "DATE", "AGE",
                  "PROFESSION", "LOCATION"]
        Y_pred_F = np.argmax(Y_pred, axis=2).flatten()
        Y_pred_F = [index2tags[i] for i in Y_pred_F]
        Y_test_F = np.array(Y).flatten()
        Y_test_F = [index2tags[i] for i in Y_test_F]
        print(metrics.classification_report(Y_test_F, Y_pred_F, labels=labels))
        print(metrics.classification_report(
            Y_test_F, Y_pred_F, labels=labels[1:]))
        from matplotlib import pyplot as plt
        _, ax = plt.subplots(figsize=(10, 10))
        metrics.ConfusionMatrixDisplay.from_predictions(
            y_true=Y_test_F, y_pred=Y_pred_F, labels=labels, normalize='true', xticks_rotation='vertical', ax=ax)

    def save(self, model_path):
        """
        Function to save model. Models are saved as h5 files in Models directory. Name is passed as argument
        :param model_path: Name of the model file
        :return: Doesn't return anything
        """
        self.model.save("Models/"+model_path+".h5")
        print("Saved model to disk")
