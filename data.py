import numpy as np

class DataPreparation:
	def __init__(self, dataset):
		self.dataset = dataset
		self.word2id = self.get_word2id()
		self.onehotvectors = self.get_onehotvector()

	def get_word2id(self):
		word_list = list()
		for sentence in self.dataset.sentences:
			word_list+= sentence.lemma_list
		unique_words = list(set(word_list))
		return dict((c, i) for i, c in enumerate(unique_words))

	def get_onehotvector(self):
		onehot_matrix = np.zeros((len(self.word2id), len(self.word2id)),  dtype = int)
		for word, id in self.word2id.items():
			onehot_matrix[id][id] = 1
		return onehot_matrix



