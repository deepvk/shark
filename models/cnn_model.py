import ujson

from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import sequence

from keras.layers import Dense, Input, Activation, Conv1D, Embedding
from keras.layers import Dropout, GlobalMaxPooling1D
from keras.layers import concatenate, BatchNormalization


class ToxicDetector(object):
    """
    CNN-based classification model for toxic texts
    Trained on the dataset from:
    https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    for educational purposes
    The model is intentionally simple so we can then export it to tf-js
    """

    MAX_LEN = 100
    EMB_DIM = 50
    KERNEL_SIZES = [3, 5, 7]

    LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def __init__(self, model_weights_path, model_vocab_path):
        self.model_weights_path = model_weights_path
        self.vocab = self._load_vocab(model_vocab_path)
        num_words = len(self.vocab)
        self.model = self._load_model(num_words=num_words,
                                      embed_dim=self.EMB_DIM,
                                      kernel_sizes=self.KERNEL_SIZES,
                                      max_len=self.MAX_LEN,
                                      weights_path=model_weights_path)

    def _load_vocab(self, model_vocab_path):
        return ujson.load(open(model_vocab_path, "r"))

    def _load_model(self,
                    num_words,
                    embed_dim,
                    kernel_sizes,
                    max_len,
                    weights_path=None):

        sequence_input = Input(shape=(max_len,))

        x = Embedding(num_words + 1, embed_dim, trainable=False)(sequence_input)

        xs = []

        for k_s in kernel_sizes:
            x_i = Conv1D(64, kernel_size=k_s, padding="same", kernel_initializer="glorot_uniform")(x)
            x_i = BatchNormalization()(x_i)
            x_i = Activation('relu')(x_i)
            x_i = GlobalMaxPooling1D()(x_i)
            xs.append(x_i)

        x = concatenate(xs, axis=1)
        x = Dropout(rate=0.3)(x)
        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(rate=0.3)(x)
        preds = Dense(6, activation="sigmoid")(x)
        model = Model(sequence_input, preds)

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

        if weights_path is not None:
            model.load_weights(weights_path)

        return model

    def _text_to_sequence(self, text):
        ids = [self.vocab[x] for x in text.lower().split() if x in self.vocab]
        seq = sequence.pad_sequences([ids], self.MAX_LEN, padding="post")
        return seq

    def _predict_proba(self, text):
        seq = self._text_to_sequence(text)
        preds = self.model.predict(seq)
        return preds[0]

    def predict(self, text):
        probs = self._predict_proba(text)
        res = {}
        for i, label in enumerate(self.LABELS):
            res[label] = probs[i]
        return res