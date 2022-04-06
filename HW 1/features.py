from collections import OrderedDict, Counter
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np
import math


class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        list_ = Counter(' '.join(X).split())
        self.bow = []
        for v, k in list_.most_common(self.k):
            self.bow.append(v)

        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """

        result = []
        counter = Counter(text.split(' '))
        for elem in self.bow:
            result.append(counter[elem])

        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = OrderedDict()

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """

        self.build_idf(X)

        # fit method must always return self
        return self

    def build_tf(self, text):
        split_phrase = text.split()
        df_dict = dict.fromkeys(set(split_phrase), 0)
        freq_ = Counter(split_phrase)

        for word, val in freq_.most_common():
            df_dict[word] = val
        bag_size = len(df_dict)

        for word, val in df_dict.items():
            df_dict[word] /= bag_size

        return df_dict

    def build_idf(self, raw_text, k=None):
        text = ' '.join(raw_text).split()
        n_bag = len(set(text))
        list_ = Counter(text)

        bag_size = n_bag if self.k is None else self.k
        for word, val in list_.most_common(bag_size):
            self.idf[word] = 0
            for phrase in raw_text:
                if word in phrase.split():
                    self.idf[word] += 1

        nn = len(text)

        for word, val in self.idf.items():
            self.idf[word] = math.log(nn / self.idf[word])

    def build_tf_idf(self, phrase_tf):
        res = []
        for word, val in self.idf.items():
            if word in phrase_tf:
                res.append(phrase_tf[word] * val)
            else:
                res.append(0.)
        return np.array(res)

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """

        result = self.build_tf_idf(self.build_tf(text))
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])
