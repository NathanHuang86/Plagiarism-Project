from vector_database import *
from transformers import MarianMTModel, MarianTokenizer

# de = German, fr = French, el = Greek, ja = Japanese, ru = Russian
language_list = {'de', 'fr', 'el', 'ja', 'ru'}

from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

def translate_text(text, text_lang, target_lang='en'):
  # Get the name of the model
  model_name = f"Helsinki-NLP/opus-mt-{text_lang}-{target_lang}"
  # Get the tokenizer
  tokenizer = MarianTokenizer.from_pretrained(model_name)

 # Instantiate the model
  model = MarianMTModel.from_pretrained(model_name)
 
  # Translation of the text
  formated_text = ">>{}<< {}".format(text_lang, text)

  translation = model.generate(**tokenizer([formated_text], 
                                           return_tensors="pt",
                                           padding=True))
  
  translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translation][0]
 
  return translated_text