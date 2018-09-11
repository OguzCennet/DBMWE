from i_o import Dataset, Output
from data import DataPreparation
from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

import urllib
import collections
import os
import zipfile

de_corpora = "dataset/de_pud-ud-test.conllu"
tr_corpora = "dataset/tr_pud-ud-test.conllu"
en_corpora = "dataset/en_pud-ud-test.conllu"


def get_data():
	en_dataset = Dataset(en_corpora)
	print(en_dataset.sentences[0].arc_list)
	en_data = DataPreparation(en_dataset)


if __name__ == "__main__":
	get_data()