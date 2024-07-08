import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import pandas as pd
import torch.nn.functional as F

from torchtext.data import get_tokenizer
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        assert len(texts) == len(labels)
        self.texts = [torch.tensor([vocab[token] for token in text], dtype=torch.long) for text in texts]
        self.labels = labels
        self.max_len = max_len if max_len is not None else max(len(t) for t in self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if len(text) < self.max_len:
            # Pad the sequence if it's shorter than max_len
            text = F.pad(text, (0, self.max_len - len(text)), 'constant', vocab['<pad>'])
        return text, self.labels[idx]

    def collate_fn(self, batch):
        texts, labels = zip(*batch)
        # If all texts are already padded to max_len, no need for dynamic padding
        return torch.stack(texts), torch.tensor(labels, dtype=torch.long)

def get_training_data(filename, vocab):
    # implement preprocessing from https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/reports/6880837.pdf

    df = pd.read_csv(filename, low_memory=False)

    tokenizer = get_tokenizer("basic_english")

    df["pros"] = df["pros"].apply(lambda x: tokenizer(str(x))[:100])
    df["cons"] = df["cons"].apply(lambda x: tokenizer(str(x))[:100])

    df["text"] = df["pros"] + df["cons"]
    df["text"].apply(lambda x: x.extend(["<pad>"] * (200 - len(x))))

    labels = torch.tensor(df['overall_rating'].values - 1)
    dataset = ReviewDataset(df['text'].tolist(), labels.tolist(), vocab)

    return labels, dataset
