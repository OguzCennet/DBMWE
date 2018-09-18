from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


def cosine_distance(v1,v2):
	return (1-cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1)))/2

def cosine_similarities(data, embeddings):
	similarity_dict = dict()
	for word1 in data.unique_words:
		print(word1)
		for word2 in data.unique_words:
			if word2 != word1:
				similarity_dict[word1] = tuple((word2,cosine_distance(embeddings[word1],embeddings[word2])))
	print(similarity_dict)
