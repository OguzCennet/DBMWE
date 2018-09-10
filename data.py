import numpy as np

class DataPreparation:
	def __init__(self, dataset):
		self.dataset = dataset
		self.onehotvectors = self.get_onehotvector()

	def get_onehotvector(self):
		print("get_onehotvector")
		word_list = list()
		for sentence in self.dataset.sentences:
			word_list+= sentence.lemma_list
		unique_words = list(set(word_list))
		word2id = dict((c, i) for i, c in enumerate(unique_words))
		onehot_matrix = np.zeros((len(word2id), len(word2id)),  dtype = int)
		for word, id in word2id.items():
			onehot_matrix[id][id] = 1
		print(len(word2id))
		print(onehot_matrix[30])
		print(onehot_matrix.shape)



