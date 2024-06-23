import numpy as np


def load_glove_vectors(glove_file):
    word_vectors = {}
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            split = line.split()
            if len(split) > 2:
                word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
    return word_vectors


def get_emb_matrix(pretrained, word_counts):
    vocab_size = len(word_counts) + 2
    emb_size = len(pretrained['.'])
    vocab_to_idx = {}
    vocab = ["UNK"]
    W = np.zeros((vocab_size, emb_size), dtype="float32")
    W[0] = np.zeros(emb_size, dtype='float32')  # padding
    W[1] = np.random.uniform(-0.25, 0.25, emb_size)  # unknown words
    vocab_to_idx["UNK"] = 1
    i = 2
    for word in word_counts:
        if word in pretrained:
            W[i] = pretrained[word]
        else:
            W[i] = np.random.uniform(-0.25, 0.25, emb_size)
        vocab_to_idx[word] = i
        vocab.append(word)
        i += 1
    return W, np.array(vocab), vocab_to_idx