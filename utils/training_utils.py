import torch


def monitor_gradients(model):
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    return grad_norms

def predict_tags(sentence, model, vocab_to_idx, idx_to_label):
    model.eval()
    sentence_idx = [vocab_to_idx.get(word, 0) for word in sentence]
    sentence_tensor = torch.tensor([sentence_idx])
    print("Sentence: ", " ".join(sentence))
    with torch.no_grad():
        output = model(sentence_tensor)
    preds = torch.argmax(output, dim=2).squeeze().cpu().numpy()
    predicted_tags = [idx_to_label[idx] for idx in preds[:len(sentence)]]
    print("Predicted Tags: ", " ".join(predicted_tags))

# Function to print embeddings for a given sentence
def print_embeddings_for_sentence(sentence, model, vocab_to_idx):
    model.eval()
    sentence_idx = [vocab_to_idx.get(word.lower(), vocab_to_idx['UNK']) for word in sentence]
    sentence_tensor = torch.tensor(sentence_idx)
    with torch.no_grad():
        embeddings = model.embeddings(sentence_tensor)
    print("Input sentence:", sentence)
    print("Token indices:", sentence_idx)
    print("Embeddings:", embeddings)
