def read_conll_file(file_path):
    sentences, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence, sentence_labels = [], []
        for line in f:
            if line.startswith("-DOCSTART-"):
                continue
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    labels.append(sentence_labels)
                    sentence = []
                    sentence_labels = []
                continue
            parts = line.strip().split()
            token = parts[0].lower()  # Lowercase the token
            label = parts[-1]
            sentence.append(token)
            sentence_labels.append(label)

        # Append the last sentence if any
        if sentence:
            sentences.append(sentence)
            labels.append(sentence_labels)

        return sentences, labels
