import pandas as pd
from tqdm import tqdm

def preprocess_data(data_path, sample_size):
  
  # Read the data from specific path
  data = pd.read_csv(data_path, low_memory=False, encoding="utf-8")

  # Drop articles without Abstract
  data = data.dropna(subset = ['abstract']).reset_index(drop = True)

  # Get "sample_size" random articles
  data = data.sample(sample_size)[['abstract', 'paper_id']]
  
  return data

# Read data & preprocess it
data_path = "./document/source_data.csv"
source_data = preprocess_data(data_path, 5)