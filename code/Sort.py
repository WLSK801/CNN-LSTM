import numpy as np
from numpy import linalg as LA
def get_sentence_vector_average(sentence, glove):
    sentence_tokens = tokenize(sentence)
    count = 0
    vectors = []
    for token in sentence_tokens:
        if token in glove:
            vectors.append(glove[token])
            count = count + 1
    hello = glove["hello"]
    value = np.zeros(hello.size)
    for vector in vectors:
        value = np.add(value, vector)
    return np.divide(value, count)
def cosine_similarity(x, y):
    norm_x = LA.norm(x)
    norm_y = LA.norm(y)
    if norm_x == 0 or norm_y == 0:
        return 0    
    return np.dot(x,y) / norm_x / norm_y 
def sentiment_sort(dict_list, standard_sentence, glove):
    standard_vector = get_sentence_vector_average(standard_sentence, glove)
    arr = []
    idx = 0
    for sentence_dict in dict_list:
        vector = get_sentence_vector_average(sentence_dict['text'], glove)
        cos_sim = cosine_similarity(vector, standard_vector)
        arr.append((idx, cos_sim))
        idx = idx + 1
    arr.sort(key=lambda tup: tup[1])
    new_batch = []
    for tup in arr:
        new_batch.append(dict_list[tup[0]])
    return new_batch
def get_sorted_batch(batch, standard_sentence, glove):
    # sort batch
    dict_list = sentiment_sort(batch[0]['data'], standard_sentence, glove)
    vectors = []
    label = batch[0]['label']
    for sentence in dict_list:
        vectors.append(sentence["text_index_sequence"])
    return vectors, label
def get_unsort_batch(batch, standard_sentence, glove):
    dict_list = batch[0]['data']
    vectors = []
    label = batch[0]['label']
    for sentence in dict_list:
        vectors.append(sentence["text_index_sequence"])
    return vectors, label