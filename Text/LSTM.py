import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class FeatureExtractor(nn.Module):
    def __init__(self, voacb_size, embedding_dim=300, hidden_dim=300):

        super(FeatureExtractor, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(voacb_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, sentence):
        x = self.embedding(sentence)
        lstm_out, lstm_hidden = self.lstm(x)
        return lstm_out


class Classifier(nn.Module):
    def __init__(self, target_size=2, hidden_dim=300):
        super(Classifier, self).__init__()
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 150),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(150, target_size)
        )

    def forward(self, x, sentence_length):
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        lstm1_out, lstm1_hidden = self.lstm1(x)
        lstm2_out, lstm2_hidden = self.lstm2(lstm1_out)
        out = torch.stack([lstm2_out[i, sentence_length[i] - 1] for i in range(len(lstm2_out))], dim=0)
        out = self.fc(out)
        return out


class MutlInfo(nn.Module):
    def __init__(self, voacb_size, target_size=2, embedding_dim=300, hidden_dim=300):
        super(MutlInfo, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(voacb_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_dim + target_size, 150),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(150, 1)
        )

    def forward(self, sentence, z, u, sentence_length):
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()
        x = self.embedding(sentence)
        lstm1_out, lstm1_hidden = self.lstm1(x)
        lstm2_out, lstm2_hidden = self.lstm2(lstm1_out)
        lstm3_out, lstm3_hidden = self.lstm3(lstm2_out)
        x_new = torch.stack([lstm3_out[i, sentence_length[i]-1] for i in range(len(lstm3_out))], dim=0)

        lstm2_out_z, lstm2_hidden_z = self.lstm2(z)
        lstm3_out_z, lstm3_hidden_z = self.lstm3(lstm2_out_z)
        z_new = torch.stack([lstm3_out_z[i, sentence_length[i]-1] for i in range(len(lstm3_out_z))], dim=0)

        out = torch.cat((x_new, z_new, u), dim=1)
        out = self.fc(out)
        return out


def info_loss(MI, x, x_length, z, u, x_prime, x_prime_length):
    Ej = -F.softplus(-MI(x, z, u, x_length)).mean()
    Em = F.softplus(MI(x_prime, z, u, x_prime_length)).mean()
    return Ej - Em
