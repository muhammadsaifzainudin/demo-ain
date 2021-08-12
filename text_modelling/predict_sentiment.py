from text_modelling.text_preprocessor import text_preprocessor
from text_modelling import base_model
from torchtext.legacy import data
import torchtext.vocab as vocab
import nltk
import torch


#import model weigth
sentiment_model = base_model.bin_model_base


#import dataset
REVIEW_TEXT = data.Field(
    tokenize = 'toktok',
    tokenizer_language = 'en',
    include_lengths = True,
    batch_first = True
)
SENTIMENT = data.LabelField(dtype = torch.float, batch_first = True)
fields = {"Review_Text": ("Text", REVIEW_TEXT),"Sentiment": ("Label", SENTIMENT)}

train_data, validation_data, test_data = data.TabularDataset.splits(
    path = 'text_modelling/dataset',
    train = 'train_data.json',
    validation = 'valid_data.json',
    test = 'test_data.json',
    format = 'json',
    fields = fields,
    skip_header = True
)


#import embedding

custom_embedding = vocab.Vectors(name = "text_modelling/embedding/custom_embedding_myunifi_100d.txt",
                                cache = 'custom_embedding',
                                unk_init =  torch.Tensor.normal_
                                )


#create vector
MAX_VOCAB_SIZE = 20000
REVIEW_TEXT.build_vocab(train_data,
                        max_size = MAX_VOCAB_SIZE,
                        vectors = custom_embedding
                        )

SENTIMENT.build_vocab(train_data)




N_LAYER = sentiment_model.n_layers
HIDDEN_DIM = sentiment_model.hidden_dim



def predict(sentence, model = sentiment_model):
  model.eval()
  processed = text_preprocessor(sentence)
  if(processed == ""):
    predictions = "nan"
    verdict = "Unspecified Comments"
  else:
    tokenized = [word for word in nltk.word_tokenize(processed)]
    indexed = [REVIEW_TEXT.vocab.stoi[word] for word in tokenized]
    tensor = torch.tensor(indexed)
    tensor = tensor.unsqueeze(1).T
    hidden = torch.zeros(N_LAYER, 1, HIDDEN_DIM), torch.zeros(N_LAYER, 1, HIDDEN_DIM)
    hidden = tuple([each.data for each in hidden])
  
    with torch.no_grad():
      predictions, hidden = model(tensor, hidden)
      #print(predictions)
      predictions = torch.sigmoid(predictions)
      #print(predictions)
      predictions = predictions.view(1, 1,  -1)
      #print(predictions)
      predictions = predictions[:, :, -1].squeeze()
      #print(predictions)
      predictions = predictions.item()
  

    if predictions > 0.55:
      verdict = 'positive'
    elif predictions < 0.51:
      verdict = 'negative'
    else:
      verdict = "Unspecified Comments"
    
  
  return predictions , verdict


  
