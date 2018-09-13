from keras.models import Model
from keras.layers import Input, Dense, Reshape, Add, dot
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

import numpy as np

window_size = 3
vector_dim = 50
epochs = 5000

class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        for i in range(vocab_size):
            in_arr1[0,] = valid_word_idx
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim

def nn_model(data):
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

	#similarity = merge([target, context], mode='cos', dot_axes=0)
	#validation_model = Model(input=[input_target, input_context], output=similarity)


	target_arr = np.zeros((1,))
	context_arr = np.zeros((1,))
	label_arr = np.zeros((1,))
	for cnt in range(epochs):
	    idx = np.random.randint(0, len(data.labels)-1)
	    target_arr[0,] = data.word_target[idx]
	    context_arr[0,] = data.word_context[idx]
	    label_arr[0,] = data.labels[idx]
	    loss = model.train_on_batch([target_arr, context_arr], label_arr)
	    if cnt % 100 == 0:
	        print("Iteration {}, loss={}".format(cnt, loss))

	weights = embedding.get_weights()[0]
	words_embeddings = {w:weights[idx] for w, idx in data.word2id.items()}
	#print(words_embeddings['peaceful'])