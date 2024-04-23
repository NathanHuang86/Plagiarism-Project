from plagarism_analysis import *

english_article_to_check = "Abstracts of scientific papers are sometimes poorly written, often lack important information, and occasionally convey a biased picture. This paper provides detailed suggestions, with examples, for writing the background, methods, results, and conclusions sections of a good abstract. The primary target of this paper is the young researcher; however, authors with all levels of experience may find useful ideas in the paper."

# Select an existing article from the data
new_incoming_text = source_data.iloc[0]['abstract']
 
# Run the plagiarism detection
analysis_result = run_plagiarism_analysis(english_article_to_check, vector_database, plagiarism_threshold=0.8)
print(analysis_result)