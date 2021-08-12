import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional = False, drop_p = 0.5):
        super(BaseLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
    
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, 
                            hidden_size = hidden_dim,
                            num_layers= n_layers,
                            dropout= drop_p,
                            bidirectional = bidirectional,
                            batch_first= True)

        self.dropout = nn.Dropout(0.3)
        self.maxpool = nn.MaxPool1d(4)
        self.full_layer_1 = nn.Linear(hidden_dim//2, hidden_dim//2)
        self.fc = nn.Linear(hidden_dim//2 , output_size)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, text, hidden):
        #[batch_size, seq_len]
        batch_size = text.size(0)
        embeded = self.embedding(text)
        #[batch_size, seq_len, embedding_dim]
        
        #packed_embeds = nn.utils.rnn.pack_padded_sequence(embeded, text_length, batch_first = True)
        
        #hidden = None
        #init_hidden = ([numlayer, batch_size, hidden_dim], [numlayer, batch_size, hidden_dim])
        h, c = hidden
        lstm_out, (h,c) = self.lstm(embeded, (h,c))
        #hidden = ([numlayer, batch_size, hidden_dim], [numlayer, batch_size, hidden_dim]) bidirectional, numlayer *2
        #lstm_out = [batch_size, seq_len, hidden_dim]
        
        #hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        
        lstm_out = self.dropout(lstm_out)
        #dense_outputs = F.relu(self.full_layer_1(lstm_out))
        #dense_outputs = F.relu(self.full_layer_2(dense_outputs))
        lstm_out = self.maxpool(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim//2)
        #[batch_size*seq_len, hidden_dim]
        dense_outputs = torch.relu(self.full_layer_1(lstm_out))
        dense_outputs = self.fc(dense_outputs)
        #[Batch_size*seq_len, output_dim]
        
        #outputs = self.sigmoid(dense_outputs)
        #[Batch_size*seq_len, 1]
        
        #outputs = outputs.view(text.shape[0], -1)
        #[Batch_size, seq_len]
        
        
        #outputs = outputs[:, -1]

        return dense_outputs, hidden
      
    def init_hidden(self, batch_size):
        if self.bidirectional:
          c0 = torch.zeros(self.n_layers *2, batch_size, self.hidden_dim)
          h0 = torch.zeros(self.n_layers *2, batch_size, self.hidden_dim)
        else:
          c0 = torch.zeros(self.n_layers , batch_size, self.hidden_dim)
          h0 = torch.zeros(self.n_layers , batch_size, self.hidden_dim)
          
        
        return (h0, c0)
      
#define model 
INPUT_DIM = 14831
EMBEDDING_DIM = 100
OUTPUT_DIM = 4
HIDDEN_DIM = 256
N_LAYER = 4

model_base = BaseLSTM(vocab_size = INPUT_DIM, 
                    output_size = OUTPUT_DIM, 
                    hidden_dim = HIDDEN_DIM, 
                    embedding_dim = EMBEDDING_DIM,
                    n_layers = N_LAYER,
                    bidirectional = True)

model_base.load_state_dict(torch.load("text_modelling/model/base_model.pt"))
      

#create model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional, drop_p = 0.5):
        super(SentimentLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
    
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, 
                            hidden_size = hidden_dim,
                            num_layers= n_layers,
                            dropout= drop_p,
                            bidirectional = bidirectional,
                            batch_first= True)
        
        self.dropout = nn.Dropout(0.3)
        self.maxpool = nn.MaxPool1d(4)

        self.fc = nn.Linear(hidden_dim//4 , output_size)
  

    def forward(self, text, hidden):
        #[batch_size, seq_len]
        batch_size = text.size(0)
        embeded = self.embedding(text)
        #[batch_size, seq_len, embedding_dim]
        
        #packed_embeds = nn.utils.rnn.pack_padded_sequence(embeded, text_length, batch_first = True)
        
        #hidden = None
        #init_hidden = ([numlayer, batch_size, hidden_dim], [numlayer, batch_size, hidden_dim])
        lstm_out, hidden = self.lstm(embeded, hidden)
        #hidden = ([numlayer, batch_size, hidden_dim], [numlayer, batch_size, hidden_dim])
        #lstm_out = [batch_size, seq_len, hidden_dim]
        
        #hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.maxpool(lstm_out)
        
        
       
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim//4)
        #[batch_size*seq_len, hidden_dim]
        
        
        dense_outputs = self.fc(lstm_out)
        #[Batch_size*seq_len, 1]
        
        
        #outputs = self.sigmoid(dense_outputs)
        #[Batch_size*seq_len, 1]
        
        #outputs = outputs.view(text.shape[0], -1)
        #[Batch_size, seq_len]
        
        
        #outputs = outputs[:, -1]

        return dense_outputs, hidden
      
    def init_hidden(self, batch_size):
        if self.bidirectional:
          c0 = torch.zeros(self.n_layers *2, batch_size, self.hidden_dim)
          h0 = torch.zeros(self.n_layers *2, batch_size, self.hidden_dim)
        else:
          c0 = torch.zeros(self.n_layers , batch_size, self.hidden_dim)
          h0 = torch.zeros(self.n_layers , batch_size, self.hidden_dim)
          
        
        return (h0, c0)
        
    
#define model 
INPUT_DIM = 16079
EMBEDDING_DIM = 100
BIN_OUTPUT_DIM = 1
HIDDEN_DIM = 256
BIN_N_LAYER = 2


bin_model_base = SentimentLSTM(vocab_size = INPUT_DIM, 
                    output_size = BIN_OUTPUT_DIM, 
                    hidden_dim = HIDDEN_DIM, 
                    embedding_dim = EMBEDDING_DIM,
                    n_layers = BIN_N_LAYER,
                    bidirectional = False)


bin_model_base.load_state_dict(torch.load("text_modelling/model/sentiment_model.pt"))




class SentimentLSTMPrev(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional, drop_p = 0.5):
        super(SentimentLSTMPrev, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
    
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, 
                            hidden_size = hidden_dim,
                            num_layers= n_layers,
                            dropout= drop_p,
                            bidirectional = False,
                            batch_first= True)


        self.fc = nn.Linear(hidden_dim , output_size)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, text, hidden):
        #[batch_size, seq_len]
        batch_size = text.size(0)
        embeded = self.embedding(text)
        #[batch_size, seq_len, embedding_dim]
        
        #packed_embeds = nn.utils.rnn.pack_padded_sequence(embeded, text_length, batch_first = True)
        
        #hidden = None
        #init_hidden = ([numlayer, batch_size, hidden_dim], [numlayer, batch_size, hidden_dim])
        lstm_out, hidden = self.lstm(embeded, hidden)
        #hidden = ([numlayer, batch_size, hidden_dim], [numlayer, batch_size, hidden_dim])
        #lstm_out = [batch_size, seq_len, hidden_dim]
        
        #hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        #[batch_size*seq_len, hidden_dim]
        
        dense_outputs = self.fc(lstm_out)
        #[Batch_size*seq_len, 1]
        
        #outputs = self.sigmoid(dense_outputs)
        #[Batch_size*seq_len, 1]
        
        #outputs = outputs.view(text.shape[0], -1)
        #[Batch_size, seq_len]
        
        
        #outputs = outputs[:, -1]

        return dense_outputs, hidden
      
    def init_hidden(self, batch_size):
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        
        return (h0, c0)
      
      
      
INPUT_DIM_PREV = 10002
EMBEDDING_DIM_PREV = 100
OUTPUT_DIM_PREV = 1
HIDDEN_DIM_PREV = 256
N_LAYER_PREV = 2

bin_model_prev = SentimentLSTMPrev(INPUT_DIM_PREV, OUTPUT_DIM_PREV, EMBEDDING_DIM_PREV, HIDDEN_DIM_PREV, N_LAYER_PREV, bidirectional = False, drop_p = 0.5)

  
bin_model_prev.load_state_dict(torch.load("text_modelling/model/save_model.pt"))   





