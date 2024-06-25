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
    vocab = ["PAD", "UNK"]
    W = np.zeros((vocab_size, emb_size), dtype="float32")
    W[0] = np.zeros(emb_size, dtype='float32')  # padding

    if len(pretrained) > 0:
        pretrained_mean = np.mean(list(pretrained.values()), axis=0)
    else:
        pretrained_mean = np.zeros(emb_size, dtype='float32')
    W[1] = pretrained_mean
    vocab_to_idx["PAD"] = 0
    vocab_to_idx["UNK"] = 1
    i = 2
    for word in word_counts:
        if word in pretrained:
            W[i] = pretrained[word]
        else:
            W[i] = pretrained_mean
        vocab_to_idx[word] = i
        vocab.append(word)
        i += 1
    return W, np.array(vocab), vocab_to_idx