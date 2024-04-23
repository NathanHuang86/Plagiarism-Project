from BERT_model import *

def create_vector_database(data):
  
   # The list of all the vectors
   vectors = []
  
   # Get overall text data
   source_data = data.abstract.values
  
   # Loop over all the comment and get the embeddings
   for text in tqdm(source_data):
      
       # Get the embedding
       vector = create_vector_from_text(tokenizer, model, text)
      
       #add it to the list
       vectors.append(vector)
  
   data["vectors"] = vectors
   data["vectors"] = data["vectors"].apply(lambda emb: np.array(emb))
   data["vectors"] = data["vectors"].apply(lambda emb: emb.reshape(1, -1))
   return data

# Create the vector index
vector_database = create_vector_database(source_data)
vector_database.sample(5)