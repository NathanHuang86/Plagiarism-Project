from machine_translation import *
from sklearn.metrics.pairwise import cosine_similarity

def process_document(text):
  """
  Create a vector for given text and adjust it for cosine similarity search
  """
  text_vect = create_vector_from_text(tokenizer, model, text)
  text_vect = np.array(text_vect)
  text_vect = text_vect.reshape(1, -1)
  return text_vect

def is_plagiarism(similarity_score, plagiarism_threshold):
  is_plagiarism = False
  if(similarity_score >= plagiarism_threshold):
      is_plagiarism = True
  return is_plagiarism

def check_incoming_document(incoming_document):
  text_lang = detect(incoming_document)
  language_list = ['de', 'fr', 'el', 'ja', 'ru']
  final_result = ""
  if(text_lang == 'en'):
    final_result = incoming_document
  elif(text_lang not in language_list):
    final_result = None
  else:
    # Translate in English
    final_result = translate_text(incoming_document, text_lang)
  return final_result
 
def run_plagiarism_analysis(query_text, data, plagiarism_threshold=0.8):
  top_N=3
  # Check the language of the query/incoming text and translate if required.
  document_translation = check_incoming_document(query_text)
  if(document_translation is None):
    print("Only the following languages are supported: English, French, Russian, German, Greek and Japanese")
    exit(-1)
  else:
    # Preprocess the document to get the required vector for similarity analysis
    query_vect = process_document(document_translation)

    # Run similarity Search
    data["similarity"] = data["vectors"].apply(lambda x:
                                            cosine_similarity(query_vect, x))
    data["similarity"] = data["similarity"].apply(lambda x: x[0][0])
    similar_articles = data.sort_values(by='similarity',
                                        ascending=False)[1:top_N+1]
    formated_result = similar_articles[["abstract",
                                        "similarity"]].reset_index(drop = True)
    similarity_score = formated_result.iloc[0]["similarity"]
    most_similar_article = formated_result.iloc[0]["abstract"]
    is_plagiarism_bool = is_plagiarism(similarity_score, plagiarism_threshold)
    plagiarism_decision = {'similarity_score': similarity_score,
                          'is_plagiarism': is_plagiarism_bool,
                          'most_similar_article': most_similar_article,
                          'article_submitted': query_text
                          }
    return plagiarism_decision