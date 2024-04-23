import pandas as pd

def preprocess_data(data_path, sample_size):
 
  # Read the data from specific path
  data = pd.read_csv(data_path, low_memory=False)

  # Drop articles without Abstract
  data = data.dropna(subset = ['abstract']).reset_index(drop = True)

  # Get "sample_size" random articles
  data = data.sample(sample_size)[['abstract']]
  
return data

# Read data & preprocess it
data_path = "./data/cord19_source_data.csv"