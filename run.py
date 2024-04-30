from plagarism_analysis import *

english_article_to_check = "Thus, for the vast majority of readers, the paper does not exist beyond its abstract. For the referees, and the few readers who wish to read beyond the abstract, the abstract sets the tone for the rest of the paper. It is therefore the duty of the author to ensure that the abstract is properly representative of the entire paper. For this, the abstract must have some general qualities. These are listed in Table 1."

# Select an existing article from the data
new_incoming_text = source_data.iloc[0]['abstract']
 
# Run the plagiarism detection
analysis_result = run_plagiarism_analysis(new_incoming_text, vector_database, plagiarism_threshold=0.8)

print(analysis_result)