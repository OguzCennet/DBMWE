import numpy as np
import random



class DataPreparation:
	def __init__(self, dataset):
		self.dataset = dataset
		self.unique_words = []
		self.dependent_couples_str = []
		self.linear_couples_str = []
		self.word2id = self.get_word2id()
		self.dependent_couples_int = self.get_dependent_couple()
		self.dependent_negative_samples = self.get_dependent_negative_samples()
		self.dependent_train_set = self.get_dependent_train_set()
		#self.onehotvectors = self.get_onehotvector()
		#self.linear_couples_int = self.get_linear_couples()
		

	def get_word2id(self):
		word_list = list()
		for sentence in self.dataset.sentences:
			word_list+= sentence.form_list
		unique_words = list(set(word_list))
		self.unique_words = unique_words
		print(len(unique_words))
		return dict((c, i) for i, c in enumerate(unique_words))

	def get_dependent_couple(self):
		couples_int = list()
		couples_str = list()

		for sentence in self.dataset.sentences:
			for arc in sentence.arc_list:
				if 0 not in arc:
					couples_str.append((sentence.form_list[arc[1]], sentence.form_list[arc[0]]))
					couples_int.append(np.array([self.word2id[sentence.form_list[arc[1]]],self.word2id[sentence.form_list[arc[0]]], 1]))

		word_target, word_context, labels = zip(*couples_int)
		self.dependent_word_target = np.array(word_target, dtype="int32")
		self.dependent_word_context = np.array(word_context, dtype="int32")
		self.dependent_couples_str = couples_str
		return couples_int

	def get_dependent_negative_samples(self):
		negative_couples = list()
		negative_sample_count = 3
		word_negative_sample_dict = dict()
		word_target, word_context = zip(*self.dependent_couples_str)

		for word in self.unique_words:
			context_word_list = list()
			for t_word in word_target:
				if word == t_word and not word_context[word_target.index(t_word)] in context_word_list:
					context_word_list.append(word_context[word_target.index(t_word)])
			word_negative_sample_dict[word] = context_word_list

		for unique in self.unique_words:
			for i in range(0, negative_sample_count-1):
				negative_sample = random.choice(self.unique_words)
				if negative_sample not in word_negative_sample_dict[unique]:
					negative_couples.append(np.array([self.word2id[unique], self.word2id[negative_sample], 0]))
				else:
					negative_couples.append(np.array([self.word2id[unique], self.word2id[negative_sample], 1]))

		return negative_couples

	def get_dependent_train_set(self):
		train_list = self.dependent_couples_int + self.dependent_negative_samples
		random.shuffle(train_list)
		return train_list

'''
		def get_linear_couples(self):
			couples = list()
			window_size = 2
			for sentence in self.dataset.sentences:
				for idx, form in enumerate(sentence.form_list):
					if sentence.form_list.index(form) != 0:
						right = idx - window_size 
						left  = idx + window_size + 1
						right_list = sentence.form_list[right : idx]
						left_list = sentence.form_list[idx+1  : left]

						if right_list:
							for right_word in right_list:
								couples.append(np.array([self.word2id[form],self.word2id[right_word]]))
								self.linear_labels.append(1)
						if left_list:
							for left_word in left_list:
								couples.append(np.array([self.word2id[form],self.word2id[left_word]]))
								self.linear_labels.append(1)

			word_target, word_context = zip(*couples)
			self.linear_word_target = np.array(word_target, dtype="int32")
			self.linear_word_context = np.array(word_context, dtype="int32")
			return couples

	def get_onehotvector(self):
		onehot_matrix = np.zeros((len(self.word2id), len(self.word2id)),  dtype = int)
		for word, id in self.word2id.items():
			onehot_matrix[id][id] = 1
		return onehot_matrix

'''














