from i_o import Dataset
from sgns import SGNS
from model import TrainModel
from cosine_sim import cosine_similarities

#import zipfil

de_corpora = "dataset/de_pud-ud-test.conllu"
tr_corpora = "dataset/tr_pud-ud-test.conllu"
en_corpora = "dataset/en_pud-ud-test.conllu"

if __name__ == "__main__":

	en_dataset = Dataset(en_corpora)
	en_data = SGNS(en_dataset)
	en_dependent_embeddings = TrainModel(en_data.dependent_train_set, en_data.word2id)
	print(en_dependent_embeddings.embeddings['transition'])
	en_linear_bow_embeddings = TrainModel(en_data.linear_bow_train_set, en_data.word2id)
	print(en_dependent_embeddings.embeddings['transition'])
	

	de_dataset = Dataset(de_corpora)
	de_data = SGNS(de_dataset)
	de_dependent_embeddings = TrainModel(de_data.dependent_train_set, de_data.word2id)
	print(de_dependent_embeddings.embeddings['übergangs'])
	de_linear_bow_embeddings = TrainModel(de_data.linear_bow_train_set, de_data.word2id)
	print(de_linear_bow_embeddings.embeddings['übergangs'])

	tr_dataset = Dataset(tr_corpora)
	tr_data = SGNS(tr_dataset)
	tr_dependent_embeddings = TrainModel(tr_data.dependent_train_set, tr_data.word2id)
	print(tr_dependent_embeddings.embeddings['dönüşümün'])
	tr_linear_bow_embeddings = TrainModel(tr_data.linear_bow_train_set, tr_data.word2id)
	print(tr_linear_bow_embeddings.embeddings['dönüşümün'])