from keras.models import Model
from keras.layers import Input, Dense, Reshape, Add, dot
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

import numpy as np

window_size = 3
vector_dim = 50
epochs = 5000

def run_model_with_dependencies(data):
	print("nn_model")
	vocab_size = len(data.word2id)
	input_target = Input((1,))
	input_context = Input((1,))

	embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')

	target = embedding(input_target)
	target = Reshape((vector_dim, 1))(target)

	context = embedding(input_context)
	context = Reshape((vector_dim, 1))(context)

	dot_product = dot([target, context], axes=1, normalize=False)
	dot_product = Reshape((1,))(dot_product)
	#output = Dense(1, activation='sigmoid')(dot_product)

	model = Model(input=[input_target, input_context], output=dot_product)
	model.compile(loss='binary_crossentropy', optimizer='rmsprop')

	target_arr = np.zeros((1,))
	context_arr = np.zeros((1,))
	label_arr = np.zeros((1,))
	word_target, word_context, labels = zip(*data.dependent_train_set)
	for cnt in range(epochs):
	    idx = np.random.randint(0, len(labels)-1)
	    target_arr[0,] = word_target[idx]
	    context_arr[0,] = word_context[idx]
	    label_arr[0,] = labels[idx]
	    loss = model.train_on_batch([target_arr, context_arr], label_arr)
	    if cnt % 100 == 0:
	        print("Iteration {}, loss={}".format(cnt, loss))

	weights = embedding.get_weights()[0]
	words_embeddings = {w:weights[idx] for w, idx in data.word2id.items()}
	return words_embeddings

def run_model_with_linear(data):
	print("nn_model")
	vocab_size = len(data.word2id)
	input_target = Input((1,))
	input_context = Input((1,))

	embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')

	target = embedding(input_target)
	target = Reshape((vector_dim, 1))(target)

	context = embedding(input_context)
	context = Reshape((vector_dim, 1))(context)

	dot_product = dot([target, context], axes=1, normalize=True)
	dot_product = Reshape((1,))(dot_product)
	output = Dense(1, activation='sigmoid')(dot_product)

	model = Model(input=[input_target, input_context], output=output)
	model.compile(loss='binary_crossentropy', optimizer='rmsprop')
	
	target_arr = np.zeros((1,))
	context_arr = np.zeros((1,))
	label_arr = np.zeros((1,))
	for cnt in range(epochs):
	    idx = np.random.randint(0, len(data.linear_labels)-1)
	    target_arr[0,] = data.linear_word_target[idx]
	    context_arr[0,] = data.linear_word_context[idx]
	    label_arr[0,] = data.linear_labels[idx]
	    loss = model.train_on_batch([target_arr, context_arr], label_arr)
	    if cnt % 100 == 0:
	        print("Iteration {}, loss={}".format(cnt, loss))

	weights = embedding.get_weights()[0]
	words_embeddings = {w:weights[idx] for w, idx in data.word2id.items()}
	return words_embeddings