from plagarism_analysis import *

english_article_to_check = "Ainsi, pour la grande majorité des lecteurs, l’article n’existe pas au-delà de son résumé. Pour les évaluateurs et les quelques lecteurs qui souhaitent lire au-delà du résumé, le résumé donne le ton au reste de l'article. Il est donc du devoir de l'auteur de s'assurer que le résumé est correctement représentatif de l'ensemble de l'article. Pour cela, le résumé doit posséder quelques qualités générales. Ceux-ci sont répertoriés dans le tableau 1."

# Select an existing article from the data
new_incoming_text = source_data.iloc[2]['abstract']
 
# Run the plagiarism detection
database_document = run_plagiarism_analysis(new_incoming_text, vector_database, plagiarism_threshold=0.2)
random_document = run_plagiarism_analysis(english_article_to_check, vector_database, plagiarism_threshold=0.2)

print(database_document)
print(random_document)