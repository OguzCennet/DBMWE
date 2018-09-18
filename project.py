from i_o import Dataset
from data import DataPreparation
from keras_model import run_model_with_dependencies, run_model_with_linear
from cosine_sim import cosine_similarities

from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

import urllib
import collections
import os
from random import shuffle
#import zipfil

de_corpora = "dataset/de_pud-ud-test.conllu"
tr_corpora = "dataset/tr_pud-ud-test.conllu"
en_corpora = "dataset/en_pud-ud-test.conllu"

if __name__ == "__main__":

	en_dataset = Dataset(en_corpora)
	en_data = DataPreparation(en_dataset)
	

	de_dataset = Dataset(de_corpora)
	de_data = DataPreparation(de_dataset)

	tr_dataset = Dataset(tr_corpora)
	tr_data = DataPreparation(tr_dataset)

	en_dependent_embeddings = run_model_with_dependencies(en_data)
	de_dependent_embeddings = run_model_with_dependencies(de_data)
	tr_dependent_embeddings = run_model_with_dependencies(tr_data)

	#en_linear_embeddings = run_model_with_linear(en_data)
	#de_linear_embeddings = run_model_with_linear(de_data)
	#tr_linear_embeddings = run_model_with_linear(tr_data)


	#cosine_similarities(en_data, en_linear_embeddings)

	#cosine_similarities(en_data, en_dependent_embeddings)




	print("here")