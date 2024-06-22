def read_conll_file(file_path):
    sentences, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence, label = [], []
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append([word.lower() for word in sentence])
                    labels.append(label)
                    sentence, label = [], []
            elif not line.startswith("-DOCSTART-"):
                parts = line.split()
                sentence.append(parts[0])
                label.append(parts[-1])
        if sentence:
            sentences.append([word.lower() for word in sentence])
            labels.append(label)
    return sentences, labels