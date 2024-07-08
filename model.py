import math
import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()

from torch import nn
from torchtext.vocab import GloVe, vocab

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, device="cpu"):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).reshape(max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]#.detach()

class TransformerClassifier(nn.Module):
    def __init__(self, embeddings, d_model, nhead, num_encoder_layers,
                 num_classes, dropout=0.0, transformer_dropout=0.0, device="cpu"):
        super(TransformerClassifier, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=torch.float32), freeze=False)
        self.embedding_dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(d_model, device=device)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model, transformer_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.embedding_dropout(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        # output = self.output_dropout(output)
        return self.fc(output)

def get_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpu_ok = device.type != "cpu" and torch.cuda.get_device_capability() in ((7, 0), (8, 0), (9, 0))
    if gpu_ok and "A100" in torch.cuda.get_device_name(0):
        torch.set_float32_matmul_precision('high')

    glove_vectors = GloVe(name="6B", dim=300)
    glove_vocab = vocab(glove_vectors.stoi)
    glove_vocab.insert_token("<unk>", 0)
    glove_vocab.set_default_index(0)
    pretrained_embeddings = glove_vectors.vectors
    pretrained_embeddings = torch.cat((torch.zeros(1,pretrained_embeddings.shape[1]), pretrained_embeddings))

    embeddings = pretrained_embeddings
    d_model = 300 # Embedding dimension
    nhead = 6  # Number of heads in MultiHeadAttention
    num_encoder_layers = 4  # Number of TransformerEncoder layers
    num_classes = 5  # Number of output classes

    # Instantiate the model
    torch.manual_seed(0)
    model = TransformerClassifier(
            embeddings, 
            d_model, 
            nhead,
            num_encoder_layers, 
            num_classes, 
            device=device
    )

    model = model.to(device)
    if gpu_ok:
        model = torch.compile(model)

    return model, glove_vocab, device
