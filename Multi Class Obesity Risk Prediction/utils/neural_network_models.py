import torch
import torch.nn as nn

class NeuralNetworkV1(nn.Module):
    def __init__(self, in_features , num_classes):
        super(NeuralNetworkV1, self).__init__()

        self.LinearBlock1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.LinearBlock2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.LinearBlock3 = nn.Sequential(
            nn.Linear(in_features=32, out_features=64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.FC = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=num_classes),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.LinearBlock1(x)
        x = self.LinearBlock2(x)
        x = self.LinearBlock3(x)
        x = self.FC(x)
        return x

class RNN(nn.Module):
    def __init__(self,vocab_size , embed_size , hidden_size ,num_classes , num_layers):
        super(RNN,self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_size,1)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.emb(x)
        out, hn = self.lstm(out)
        out = self.relu(out[:,-1,:])
        out = self.linear(out)
        return out
    
VOCAB_SIZE = 31
EMBED_SIZE= 128
NUM_CLASSES = 7
NUM_LAYERS = 4
HIDDEN_SIZE = 256