import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from models.bilstm_model import LSTM_glove_vecs
from models.tagging_dataset import TaggingDataset
from utils.conll_reader import read_conll_file
from utils.glove_loader import load_glove_vectors, get_emb_matrix
from utils.metrics import precision_recall_f1
from utils.training_utils import print_embeddings_for_sentence, predict_tags, monitor_gradients
from torch.nn.utils import clip_grad_norm_

from utils.visualization import plot_f1_scores
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Import the scheduler

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a BiLSTM model with GloVe embeddings for sequence tagging.")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate")
parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
args = parser.parse_args()
print(f"Running with parameters: seed={args.seed}, batch_size={args.batch_size}, dropout_rate={args.dropout_rate}, learning_rate={args.learning_rate}")

SEED = args.seed
embedding_dim = 50
hidden_dim = 50  # will get doubled due to bidirectional LSTM
learning_rate = args.learning_rate
dropout_rate = args.dropout_rate
batch_size = args.batch_size

# Set seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set a seed value
set_seed(SEED)
# Check for CUDA availability and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Prepare data
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
label_to_idx = {label: idx for idx, label in enumerate(label_counts.keys(), 1)}
label_to_idx["PAD"] = 0
# Create idx_to_label for reverse mapping
idx_to_label = {idx: label for label, idx in label_to_idx.items()}


def collate_fn(batch):
    # Sort batch by sequence length (descending order) for efficient packing
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    sentences, labels = zip(*batch)

    # Find the max length in the batch
    max_length = max(len(sentence) for sentence in sentences)

    # Pad sequences to the max length
    padded_sentences = [sentence + [0] * (max_length - len(sentence)) for sentence in sentences]
    padded_labels = [label + [0] * (max_length - len(label)) for label in labels]

    return torch.tensor(padded_sentences).to(device), torch.tensor(padded_labels).to(device)


# Create Dataset and DataLoader with dynamic batching
train_dataset = TaggingDataset(train_sentences, train_labels, vocab_to_idx, label_to_idx, device, None)
dev_dataset = TaggingDataset(dev_sentences, dev_labels, vocab_to_idx, label_to_idx, device, None)
test_dataset = TaggingDataset(test_sentences, test_labels, vocab_to_idx, label_to_idx, device, None)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Initialize model parameters
vocab_size = embedding_matrix.shape[0]
num_labels = len(label_to_idx)

# Initialize the model
model = LSTM_glove_vecs(embedding_dim, hidden_dim, embedding_matrix, num_labels, dropout_rate)

model.to(device)

# Initialize loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# Training the model
num_epochs = 20
train_f1_scores = []
dev_f1_scores = []

best_model = None
best_dev_f1 = 0
best_epoch = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    train_preds = []
    train_labels_flat = []
    expl = 0
    van = 0
    normal = 0
    for sentences, labels in train_loader:
        sentences, labels = sentences, labels
        optimizer.zero_grad()
        output = model(sentences, teacher_forcing_ratio=0.0)
        output = output.view(-1, num_labels)
        labels = labels.view(-1)
        loss = criterion(output, labels)
        loss.backward()

        # Clip gradients
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check gradient norms

        grad_norms = monitor_gradients(model)
        for name, norm in grad_norms.items():
            if norm > 1e3:  # threshold for exploding gradient
                expl = expl + 1
            elif norm < 1e-6:  # threshold for vanishing gradient
                van = van + 1
            else:
                normal = normal + 1
        optimizer.step()
        total_loss += loss.item()

        preds = torch.argmax(output, dim=1).cpu().numpy()
        train_preds.extend(preds)
        train_labels_flat.extend(labels.cpu().numpy())

    total_gradients = expl + van + normal
    percent_exploding = (expl / total_gradients) * 100
    percent_vanishing = (van / total_gradients) * 100
    percent_normal = (normal / total_gradients) * 100

    print(f"Summary of gradient norms:")
    print(f"Exploding gradients: {expl} ({percent_exploding:.2f}%)")
    print(f"Vanishing gradients: {van} ({percent_vanishing:.2f}%)")
    print(f"Normal gradients: {normal} ({percent_normal:.2f}%)")

    model.eval()
    _, _, train_f1 = precision_recall_f1(train_preds, train_labels_flat, num_labels)
    train_f1_scores.append(train_f1)

    # Evaluate on dev data
    dev_preds = []
    dev_labels_flat = []
    for sentences, labels in dev_loader:
        sentences, labels = sentences, labels
        with torch.no_grad():
            output = model(sentences)
        preds = torch.argmax(output, dim=2).view(-1).cpu().numpy()
        labels_flat = labels.view(-1).cpu().numpy()
        dev_preds.extend(preds)
        dev_labels_flat.extend(labels_flat)

    _, _, dev_f1 = precision_recall_f1(dev_preds, dev_labels_flat, num_labels)
    dev_f1_scores.append(dev_f1)

    # Update the scheduler
    scheduler.step(dev_f1)

    # Save the best model based on dev F1 score
    if dev_f1 > best_dev_f1:
        best_dev_f1 = dev_f1
        best_model = model.state_dict()
        best_epoch = epoch

    print(f"Epoch {epoch + 1} - Train F1 Score: {train_f1}, Dev F1 Score: {dev_f1}")

# Load best model and evaluate on test data
model.load_state_dict(best_model)
model.eval()

test_preds = []
test_labels_flat = []
for sentences, labels in test_loader:
    sentences, labels = sentences, labels
    with torch.no_grad():
        output = model(sentences)
    preds = torch.argmax(output, dim=2).view(-1).cpu().numpy()
    labels_flat = labels.view(-1).cpu().numpy()
    test_preds.extend(preds)
    test_labels_flat.extend(labels_flat)

_, _, test_f1 = precision_recall_f1(test_preds, test_labels_flat, num_labels)
print(f"Test F1 Score of the best model: {test_f1}")

# Plot F1 scores
plot_f1_scores(train_f1_scores, dev_f1_scores, test_f1, best_epoch, num_epochs, learning_rate, dropout_rate, batch_size, SEED)
