from keras.models import Model
from keras.layers import Input, Dense, Reshape, Add, dot
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

import numpy as np

class TrainModel:
	vector_dim = 50
	epochs = 5000

	def __init__(self, train_data, word2id):
		self.train_data = train_data
		self.word2id = word2id
		self.embeddings = self.run_model()

	def run_model(self):
		vocab_size = len(self.word2id)
		input_target = Input((1,))
		input_context = Input((1,))

		embedding = Embedding(vocab_size, self.vector_dim, input_length=1, name='embedding')

		target = embedding(input_target)
		target = Reshape((self.vector_dim, 1))(target)

		context = embedding(input_context)
		context = Reshape((self.vector_dim, 1))(context)

		dot_product = dot([target, context], axes=1, normalize=False)
		dot_product = Reshape((1,))(dot_product)
		#output = Dense(1, activation='sigmoid')(dot_product)

		model = Model(input=[input_target, input_context], output=dot_product)
		model.compile(loss='binary_crossentropy', optimizer='rmsprop')


		target_arr = np.zeros((1,))
		context_arr = np.zeros((1,))
		label_arr = np.zeros((1,))
		word_target, word_context, labels = zip(*self.train_data)
		for cnt in range(self.epochs):
		    idx = np.random.randint(0, len(labels)-1)
		    target_arr[0,] = word_target[idx]
		    context_arr[0,] = word_context[idx]
		    label_arr[0,] = labels[idx]
		    loss = model.train_on_batch([target_arr, context_arr], label_arr)
		    if cnt % 100 == 0:
		        print("Iteration {}, loss={}".format(cnt, loss))

		weights = embedding.get_weights()[0]
		words_embeddings = {w:weights[idx] for w, idx in self.word2id.items()}
		return words_embeddings