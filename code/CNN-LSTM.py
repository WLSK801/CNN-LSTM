import torch.autograd as autograd
import torch
import torch.nn as nn
class TextCNNLSTM(nn.Module):
    def __init__(self, input_size, embedding_dim, window_size, n_filters, cnn_out_size, dropout_prob, hidden_dim, num_labels):
        super(TextCNNLSTM, self).__init__()
        # CNN
        self.embed = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p = dropout_prob)
        self.dropout2 = nn.Dropout(p = dropout_prob)
        self.conv1 = nn.Conv2d(1, n_filters, (window_size, embedding_dim)) 
        self.fc1 = nn.Linear(n_filters, cnn_out_size)
        self.init_weights()
        # LSTM
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=cnn_out_size, hidden_size=hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, num_labels)
        self.hidden = self.init_hidden()
        
    def forward(self, x):
        # Pass the input through your layers in order
        out = self.embed(x)
        out = self.dropout(out)
        out = out.unsqueeze(1)
        out = self.conv1(out).squeeze(3)
        out = F.relu(out)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        cnn_out = self.fc1(self.dropout2(out))
        cnn_out_size = cnn_out.size()
        cnn_out = cnn_out.view(cnn_out_size[0],1, cnn_out_size[1])
        lstm_out, self.hidden = self.lstm(cnn_out, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y, dim = 1)
        return log_probs

    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.fc1]
        em_layer = [self.embed]
     
        for layer in lin_layers+em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)
    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))