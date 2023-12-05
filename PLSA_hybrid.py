import string
import numpy as np
from nltk.corpus import stopwords
from scipy.linalg import svd
from numpy import array
from sklearn.preprocessing import normalize
import PLSA


def PLSA_hybrid(tokenized_docs, n):
    vocab = list(set([word for doc in tokenized_docs for word in doc]))
    def document_term_matrix():
        dc_matrix = [[0 for _ in range(len(vocab))] for _ in range(len(tokenized_docs))]
        for i, word in enumerate(vocab):
            for j, docs in enumerate(tokenized_docs):
                freq = docs.count(word)
                dc_matrix[j][i] = freq
        return dc_matrix

    final_matrix = array(document_term_matrix())
    # print("final_matrix: ", final_matrix)

    U, s, Vt = svd(final_matrix, full_matrices=False)
    k = n  # number of topics
    U_k = U[:, :k]
    s_k = np.diag(s[:k])
    Vt_k = Vt[:k, :]
    topic_term_matrix = np.dot(s_k, Vt_k)
    normalized_matrix = normalize(topic_term_matrix, norm='l2', axis=1)

    # Print the top terms for each topic
    n_top_words = 100
    feature_names = sorted(vocab)
    string_send = ""
    top_words = []
    for i, topic_vec in enumerate(normalized_matrix):
        top_words_idx = topic_vec.argsort()[:-n_top_words - 1:-1]
        top_words.append([feature_names[i] for i in top_words_idx])
    string_send = PLSA.PLSA(top_words, k)

    return string_send

