
import numpy as np


from scipy.linalg import svd

from numpy import array

from sklearn.preprocessing import normalize


def LSA(tokenized_docs ,n):
    vocab = list(set([word for doc in tokenized_docs for word in doc]))

    def document_term_matrix():
        dc_matrix = [[0 for _ in range(len(vocab))] for _ in range(len(tokenized_docs))]
        for i, word in enumerate(vocab):
            for j, docs in enumerate(tokenized_docs):
                freq = docs.count(word)
                dc_matrix[j][i] = freq
        return dc_matrix

    def calculate_idf():
        idf = np.zeros(len(vocab))
        total_docs = len(tokenized_docs)
        for i, word in enumerate(vocab):
            doc_count = sum(1 for doc in tokenized_docs if word in doc)
            idf[i] = np.log(total_docs / (1 + doc_count))
        return idf

    final_matrix = array(document_term_matrix())
    idf = calculate_idf()
    tfidf_matrix = final_matrix * idf
    # print("final_matrix: ", final_matrix)

    U, s, Vt = svd(tfidf_matrix, full_matrices=False)
    k = n  # number of topics
    U_k = U[:, :k]
    s_k = np.diag(s[:k])
    Vt_k = Vt[:k, :]
    topic_term_matrix = np.dot(s_k, Vt_k)
    normalized_matrix = normalize(topic_term_matrix, norm='l2', axis=1)

    # Print the top terms for each topic
    n_top_words = 10
    feature_names = sorted(vocab)

    string_send = ""
    for i, topic_vec in enumerate(normalized_matrix):
        top_words_idx = topic_vec.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        string_send += f"Topic {i + 1}: {' '.join(top_words)}" + "\n"
        # print(f"Topic {i + 1}: {' '.join(top_words)}")

    doc_topic_matrix = np.dot(U_k, s_k)
    normalized_doc_topic_matrix = normalize(doc_topic_matrix, norm='l1', axis=1)
    for doc_idx, doc in enumerate(tokenized_docs):
        string_send += f"The percentages of topics in doc no. {doc_idx}:" + "\n"
        # print(f"The percentages of topics in doc no. {doc_idx}:")
        for topic_idx, percentage in enumerate(normalized_doc_topic_matrix[doc_idx]):
            percentage = abs(percentage)
            string_send += f"Topic {topic_idx + 1}: {percentage:.2%}" + "\n"
            # print(f"Topic {topic_idx + 1}: {percentage:.2%}")
        string_send += "\n"
        # print()
    return string_send

# print(LSA(tokenized_docs, 6))