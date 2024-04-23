# Useful libraries
from preprocessing import *
import numpy as np
import torch
from keras.utils import pad_sequences
from transformers import BertTokenizer,  AutoModelForSequenceClassification
 
# Load bert model
model_path = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_path,
                                         do_lower_case=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                         output_attentions=False,
                                                         output_hidden_states=True)
                                                        
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
 
def create_vector_from_text(tokenizer, model, text, MAX_LEN = 510):
  
    input_ids = tokenizer.encode(
                        text,
                        add_special_tokens = True,
                        max_length = MAX_LEN,                          
    )   
    results = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long",
                              truncating="post", padding="post")
    # Remove the outer list.
    input_ids = results[0]
    # Create attention masks   
    attention_mask = [int(i>0) for i in input_ids]
    # Convert to tensors.
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    # Add an extra dimension for the "batch" (even though there is only one
    # input in this batch.)
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():       
        logits, encoded_layers = model(
                                    input_ids = input_ids,
                                    token_type_ids = None,
                                    attention_mask = attention_mask,
                                    return_dict=False)

    layer_i = 12 # The last BERT layer before the classifier.
    batch_i = 0 # Only one input in the batch.
    token_i = 0 # The first token, corresponding to [CLS]
      
    # Extract the vector.
    vector = encoded_layers[layer_i][batch_i][token_i]
    # Move to the CPU and convert to numpy ndarray.
    vector = vector.detach().cpu().numpy()
    return(vector)