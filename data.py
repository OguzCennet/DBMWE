import numpy as np

class DataPreparation:
	def __init__(self, dataset):
		self.dataset = dataset
		self.labels = list()
		self.word_target = []
		self.word_context = []
		self.unique_words = []
		self.word2id = self.get_word2id()
		self.onehotvectors = self.get_onehotvector()
		self.couples = self.get_couples()

	def get_word2id(self):
		word_list = list()
		for sentence in self.dataset.sentences:
			word_list+= sentence.form_list
		unique_words = list(set(word_list))
		self.unique_words = unique_words
		print(len(unique_words))
		return dict((c, i) for i, c in enumerate(unique_words))

	def get_onehotvector(self):
		onehot_matrix = np.zeros((len(self.word2id), len(self.word2id)),  dtype = int)
		for word, id in self.word2id.items():
			onehot_matrix[id][id] = 1
		return onehot_matrix

	def get_couples(self):
		couples = list()
		for sentence in self.dataset.sentences:
			for arc in sentence.arc_list:
				if 0 not in arc:
					couples.append(np.array([self.word2id[sentence.form_list[arc[0]]],self.word2id[sentence.form_list[arc[1]]]]))
					self.labels.append(1)

		word_target, word_context = zip(*couples)
		self.word_target = np.array(word_target, dtype="int32")
		self.word_context = np.array(word_context, dtype="int32")
		return couples




