import torch
from torch.utils.data import Dataset


class TaggingDataset(Dataset):
    def __init__(self, sentences, labels, vocab_to_idx, label_to_idx, device, max_length=None):
        self.sentences = sentences
        self.labels = labels
        self.vocab_to_idx = vocab_to_idx
        self.label_to_idx = label_to_idx
        self.device = device
        if max_length == None:
            self.max_length = max(len(sentence) for sentence in self.sentences)
        else:
            self.max_length = max_length
        print("max length ", self.max_length)
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        sentence_idx = [self.vocab_to_idx.get(word, 1) for word in sentence]
        label_idx = [self.label_to_idx[tag] for tag in label]

        # Padding
        if len(sentence_idx) < self.max_length:
            sentence_idx.extend([0] * (self.max_length - len(sentence_idx)))
            label_idx.extend([0] * (self.max_length - len(label_idx)))
        else:
            sentence_idx = sentence_idx[:self.max_length]
            label_idx = label_idx[:self.max_length]

        return torch.tensor(sentence_idx).to(self.device), torch.tensor(label_idx).to(self.device)
