import numpy as np
import random
import math
from collections import Counter



class SGNS:
	def __init__(self, dataset):
		self.dataset = dataset
		self.unique_words = []
		self.word2id = {}
		self.words2frequency = {}
		self.corpus_frequency  = self.get_data_frequencies()
		self.subsampling_probabilities = self.get_subsampling_probabilities()
		self.negative_sampling_probabilities = self.get_negative_sampling_probabilities()


		self.dependent_context_unique_words = []
		self.dependent_couples= self.get_dependent_couple()
		self.dependent_negative_samples = self.get_dependent_negative_samples()
		self.dependent_train_set = self.get_dependent_train_set()

		self.linear_bow_context_unique_words = []
		self.linear_bow_couples = self.get_linear_bow_couples()
		self.linear_bow_negative_samples =self.get_linear_bow_negative_samples()
		self.linear_bow_train_set = self.get_linear_bow_train_set()
		
	def get_data_frequencies(self):
		word_list = list()
		for sentence in self.dataset.sentences:
			word_list+= sentence.form_list
		self.unique_words = list(set(word_list))
		self.word2id = dict((c, i) for i, c in enumerate(self.unique_words))
		self.words2freqs = Counter(word_list)
		return len(word_list)

	def get_subsampling_probabilities(self):
		sample = 1e-3 #from Mikolov's paper 
		word_prob_dict = dict()
		for word, freq in self.words2freqs.items():
			word_fraction = freq / self.corpus_frequency
			prob = (math.sqrt( word_fraction / sample ) +1 )*( sample / word_fraction)
			word_prob_dict[word] = prob
		return word_prob_dict

	def get_negative_sampling_probabilities(self):
		word_prob_dict = dict()
		denominator = 0
		for word_d, freq_d in self.words2freqs.items():
			denominator += math.pow(freq_d,3/4)
		for word_n, freq_n in self.words2freqs.items():
			word_prob_dict[word_n] = (math.pow(freq_n,3/4)) / denominator
		return word_prob_dict

	def get_dependent_couple(self):
		couples = list()
		dependent_context_unique_words = list()
		for sentence in self.dataset.sentences:
			for arc in sentence.arc_list:
				if 0 not in arc:
					subsampling_probability = self.subsampling_probabilities[sentence.form_list[arc[1]]]
					if subsampling_probability >= 1:
						dependent_context_unique_words.append(sentence.form_list[arc[1]])
						couples.append(np.array([self.word2id[sentence.form_list[arc[1]]],self.word2id[sentence.form_list[arc[0]]], 1]))
					else:
						decision = np.random.choice([1,0], p=[subsampling_probability, 1-subsampling_probability])
						if decision == 1:
							dependent_context_unique_words.append(sentence.form_list[arc[1]])
							couples.append(np.array([self.word2id[sentence.form_list[arc[1]]],self.word2id[sentence.form_list[arc[0]]], 1]))

		self.dependent_context_unique_words = list(set(dependent_context_unique_words))
		return couples

	def get_dependent_negative_samples(self):
		ngs_constant = 1000000
		negative_sample_count = 20
		negative_sample_table = list()
		negative_sample_couples = list()

		#it is our negative sampling table by its negative sampling probability
		for word, ngs_prob in self.negative_sampling_probabilities.items():
			times = ngs_prob * ngs_constant
			for i in range(0, int(times)-1):
				negative_sample_table.append(self.word2id.get(word))

		for unique in self.dependent_context_unique_words:
			for i in range(1,negative_sample_count):
				negative_sample = random.choice(negative_sample_table)
				negative_sample_couples.append(np.array([self.word2id[unique], negative_sample, 0]))

		return negative_sample_couples

	def get_dependent_train_set(self):
		train_list = self.dependent_couples + self.dependent_negative_samples
		random.shuffle(train_list)
		return train_list


	def get_linear_bow_couples(self):
		window_size = 2
		couples = list()
		sentences = []
		linear_bow_context_unique_words = list()

		#create a new sentence sequence according to subsampling probabilities
		for sentence in self.dataset.sentences:
			new_form_list = []
			for form in sentence.form_list:
				subsampling_probability = self.subsampling_probabilities[form]
				if subsampling_probability >= 1:
					linear_bow_context_unique_words.append(form)
					new_form_list.append(form)
				else:
					decision = np.random.choice([1,0], p=[subsampling_probability, 1-subsampling_probability])
					if decision == 1:
						linear_bow_context_unique_words.append(form)
						new_form_list.append(form)
			sentences.append(new_form_list)

		for new_form_list in sentences:
			for idx, form in enumerate(new_form_list):
				right = idx - window_size 
				left  = idx + window_size + 1
				right_list = sentence.form_list[right : idx]
				left_list = sentence.form_list[idx+1  : left]

				if right_list:
					for right_word in right_list:
						couples.append(np.array([self.word2id[form],self.word2id[right_word],1]))
				if left_list:
					for left_word in left_list:
						couples.append(np.array([self.word2id[form],self.word2id[left_word],1]))

		self.linear_bow_context_unique_words = list(set(linear_bow_context_unique_words))
		return couples

	def get_linear_bow_negative_samples(self):
		ngs_constant = 1000000
		negative_sample_count = 20
		negative_sample_table = list()
		negative_sample_couples = list()

		#it is our negative sampling table by its negative sampling probability
		for word, ngs_prob in self.negative_sampling_probabilities.items():
			times = ngs_prob * ngs_constant
			for i in range(0, int(times)-1):
				negative_sample_table.append(self.word2id.get(word))

		for unique in self.linear_bow_context_unique_words:
			for i in range(1,negative_sample_count):
				negative_sample = random.choice(negative_sample_table)
				negative_sample_couples.append(np.array([self.word2id[unique], negative_sample, 0]))

		return negative_sample_couples 

	def get_linear_bow_train_set(self):
		train_list = self.linear_bow_couples + self.linear_bow_negative_samples
		random.shuffle(train_list)
		return train_list



















