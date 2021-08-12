from sklearn.pipeline import Pipeline
from text_modelling import text_preprocessor as tp
import pickle
import nltk
nltk.download('punkt')
import numpy as np

file_path = "text_modelling/model/community_topic_model.pickle"

with open(file_path, 'rb') as f:
    model = pickle.load(f)
    

topic_dict = { "Touchpoints" : 0
             , "Network and Services": 1
             , "Crowd Sourcing and Content" : 2
             , "Product Offering": 3
             , "Community UIUX": 4
             , "Responsiveness": 5} 

key_list = list(topic_dict.keys())

    
#def CheckProbabililty (array1, val):  
#       
#    for x in array1:  
#        if val <= x:  
#           return False
#R    return True

  
def predict_community_topic(text):
  processed_text = [tp.text_preprocessor(text)]
  probability = model.predict_proba(processed_text)[0]
  max_probability = np.max(probability)
  position = np.array([np.argmax(probability)])[0]
  
  if(max_probability > 0.55):
    result = key_list[position]
  else:
    result = "Unspecified Comments"
      
  return np.round(max_probability,2), result
    
    
