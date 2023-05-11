
import numpy as np



def LDA(tokenized_docs, n):
    vocab = list(set([word for doc in tokenized_docs for word in doc]))

    num_topics = n
    num_iterations = 100       # define the number of topics and iterations

    topic_word = np.zeros((len(vocab), num_topics))
    doc_topic = np.zeros((len(tokenized_docs), num_topics))

    word_topic = []
    for i, doc in enumerate(tokenized_docs):
        doc_topic_dist = np.random.dirichlet(np.ones(num_topics))
        for j, word in enumerate(doc):
            topic = np.random.choice(num_topics, p=doc_topic_dist)
            word_topic.append((i, j, topic))
            topic_word[vocab.index(word), topic] += 1
            doc_topic[i, topic] += 1

    alpha = 0.01
    beta = 0.1

    for iteration in range(num_iterations):
        for i, j, topic in word_topic:
            word = tokenized_docs[i][j]
            topic_word[vocab.index(word), topic] -= 1
            doc_topic[i, topic] -= 1
            topic_dist = (topic_word[vocab.index(word), :] + beta) * (doc_topic[i, :] + alpha) / (np.sum(topic_word, axis=0) + beta) / (np.sum(doc_topic[i, :]) + alpha)
            normalized_topic_dist = topic_dist / np.sum(topic_dist)
            new_topic = np.random.choice(num_topics, p=normalized_topic_dist)
            t = (i * len(tokenized_docs[i])) + j
            index = 0
            if i > 0:
                for p in range(i):
                    index += len(tokenized_docs[p])
            else:
                index = 0

            # word_topic[(i * len(tokenized_docs[i])) + j] = (i, j, new_topic)
            word_topic[(index) + j] = (i, j, new_topic)
            topic_word[vocab.index(word), new_topic] += 1
            doc_topic[i, new_topic] += 1
    string_send = ""
    # print the top words for each topic
    # print("The top words in the topics are:")
    # string_send += "The top words in the topics are:" + "\n"
    for topic in range(num_topics):
        top_words = [vocab[i] for i in np.argsort(topic_word[:, topic])[::-1][:10]]
        string_send += 'Topic {}: {}'.format(topic + 1, ' '.join(top_words)) + "\n"

    for i in range(len(doc_topic)):
        normalised_doc_topic = doc_topic[i] / np.sum(doc_topic[i]) * 100
        string_send += "The percentages of topics in doc no. " + str(i + 1) + "is: " + "\n"
        for j in range(len(doc_topic[i])):
            string_send += "Topic " + str(j + 1) + ": " + str(round(normalised_doc_topic[j], 1)) + " %" + "\n"
    return string_send

