import torch
from torch.utils.data import Dataset


class TaggingDataset(Dataset):
    def __init__(self, sentences, labels, vocab_to_idx, label_to_idx):
        self.sentences = sentences
        self.labels = labels
        self.vocab_to_idx = vocab_to_idx
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        sentence_idx = [self.vocab_to_idx.get(word, 0) for word in sentence]
        label_idx = [self.label_to_idx[tag] for tag in label]
        return torch.tensor(sentence_idx), torch.tensor(label_idx)