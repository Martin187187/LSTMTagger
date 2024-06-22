
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
# Prepare data
from models.bilstm_model import LSTM_glove_vecs
from models.tagging_dataset import TaggingDataset
from utils.conll_reader import read_conll_file
from utils.glove_loader import load_glove_vectors, get_emb_matrix
from utils.metrics import precision_recall_f1
from utils.training_utils import print_embeddings_for_sentence, predict_tags
from torch.nn.utils import clip_grad_norm_

from utils.visualization import plot_f1_scores

train_file = 'data/train.conll'
dev_file = 'data/dev.conll'
test_file = 'data/test.conll'

train_sentences, train_labels = read_conll_file(train_file)
dev_sentences, dev_labels = read_conll_file(dev_file)
test_sentences, test_labels = read_conll_file(test_file)

# Count words and labels
word_counts = {}
label_counts = {}
for sentence, labels in zip(train_sentences + dev_sentences + test_sentences, train_labels + dev_labels + test_labels):
    for word in sentence:
        word_counts[word] = word_counts.get(word, 0) + 1
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

# Load GloVe vectors and create embedding matrix
glove_file = "glove.6B.50d.txt"
glove_vectors = load_glove_vectors(glove_file)
embedding_matrix, vocab, vocab_to_idx = get_emb_matrix(glove_vectors, word_counts)
label_to_idx = {label: idx for idx, label in enumerate(label_counts.keys())}

# Create idx_to_label for reverse mapping
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

# Create Dataset and DataLoader
train_dataset = TaggingDataset(train_sentences, train_labels, vocab_to_idx, label_to_idx)
dev_dataset = TaggingDataset(dev_sentences, dev_labels, vocab_to_idx, label_to_idx)
test_dataset = TaggingDataset(test_sentences, test_labels, vocab_to_idx, label_to_idx)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1)

# Initialize model parameters
vocab_size = embedding_matrix.shape[0]
embedding_dim = 50
hidden_dim = 50  # will get doubled due to bidirectional LSTM
num_labels = len(label_to_idx)

# Initialize the model
model = LSTM_glove_vecs(vocab_size, embedding_dim, hidden_dim, embedding_matrix, num_labels)

# Calculate class weights based on inverse frequency
class_counts = np.array(list(label_counts.values()))
class_weights = 1.0 / class_counts
class_weights = class_weights / np.sum(class_weights)
class_weights = torch.FloatTensor(class_weights)

# Print the weights for each class
for idx, weight in enumerate(class_weights):
    print(f"Class {idx_to_label[idx]}: Weight = {weight.item()}")

# Initialize loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)


# Function to monitor gradients
def monitor_gradients(model):
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    return grad_norms

# Print embeddings for a sample test sentence before training
print_embeddings_for_sentence(test_sentences[2], model, vocab_to_idx)

# Training the model
num_epochs = 20
train_f1_scores = []
dev_f1_scores = []

best_model = None
best_dev_f1 = 0
best_epoch = 0
for epoch in range(num_epochs):
    predict_tags([word.lower() for word in test_sentences[10]], model, vocab_to_idx, idx_to_label)
    model.train()
    total_loss = 0

    train_preds = []
    train_labels_flat = []
    for sentences, labels in train_loader:
        optimizer.zero_grad()
        output = model(sentences) # Get model predictions
        output = output.view(-1, num_labels) # Reshape to (sequence_length, num_labels)
        labels = labels.view(-1) # Reshape to (sequence_length,)
        loss = criterion(output, labels)
        loss.backward()

        # Clip gradients
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Check gradient norms
        #grad_norms = monitor_gradients(model)
        """
        for name, norm in grad_norms.items():
            if norm > 1e3:  # threshold for exploding gradient
                print(f"Exploding gradient detected in {name} with norm {norm}")
            if norm < 1e-6:  # threshold for vanishing gradient
                print(f"Vanishing gradient detected in {name} with norm {norm}")
        """

        optimizer.step()
        total_loss += loss.item()

        preds = torch.argmax(output, dim=1).cpu().numpy()
        train_preds.extend(preds)
        train_labels_flat.extend(labels.cpu().numpy())
    model.eval()
    _, _, train_f1 = precision_recall_f1(train_preds, train_labels_flat, num_labels)
    train_f1_scores.append(train_f1)

    # Evaluate on dev data
    dev_preds = []
    dev_labels_flat = []
    for sentences, labels in dev_loader:
        with torch.no_grad():
            output = model(sentences)
        preds = torch.argmax(output, dim=2).view(-1).cpu().numpy()
        labels_flat = labels.view(-1).cpu().numpy()
        dev_preds.extend(preds)
        dev_labels_flat.extend(labels_flat)

    _, _, dev_f1 = precision_recall_f1(dev_preds, dev_labels_flat, num_labels)
    dev_f1_scores.append(dev_f1)
    # Save the best model based on dev F1 score
    if dev_f1 > best_dev_f1:
        best_dev_f1 = dev_f1
        best_model = model.state_dict()
        best_epoch = epoch


    # Predict and print the tags for the test sentence
    print(f"Epoch {epoch + 1} - Train F1 Score: {train_f1}, Dev F1 Score: {dev_f1}")


model.load_state_dict(best_model)
model.eval()
# Evaluate on test data
test_preds = []
test_labels_flat = []
for sentences, labels in test_loader:
    with torch.no_grad():
        output = model(sentences)
    preds = torch.argmax(output, dim=2).view(-1).cpu().numpy()
    labels_flat = labels.view(-1).cpu().numpy()
    test_preds.extend(preds)
    test_labels_flat.extend(labels_flat)

_, _, test_f1 = precision_recall_f1(test_preds, test_labels_flat, num_labels)
print(f"Test F1 Score of the best model: {test_f1}")
plot_f1_scores(train_f1_scores, dev_f1_scores, test_f1, best_epoch, num_epochs)