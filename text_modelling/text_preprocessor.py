import re
import unicodedata


def unicodeToAscii(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def text_preprocessor(text):
    text = text.lower().strip() #convert to ascii
    text = re.sub(r"\S*@\S*\s?", r"", text) #remove email and username
    text = re.sub(r"[.!?,]", r"", text) #remove . ? ! , 
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text) #remove non-alphabet character
    text = ' '.join(word for word in text.split() if len(word) > 1) #remove 1 character word
    text = unicodeToAscii(text) #convert to ascii
    return text
  
  
text = "i"
text_preprocessor(text)


  