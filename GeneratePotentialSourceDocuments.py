import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import pandas as pd
import xml.etree.ElementTree as ET

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def read_files(directory_path):
    documents = []
    file_names = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    documents.append(f.read())
                    file_names.append(file)
    return documents, file_names

def parse_xml_file(xml_file_path):
    # Set is used since there can be duplicate source file names in xml file
    source_references = set()
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    for feature in root.findall('feature'):
        if feature.get('name') == 'plagiarism':
            source_reference = feature.get('source_reference')
            source_references.add(source_reference)

    return list(source_references)

def check_matches(top_k_documents, xml_source_references):
    """
        This helper function checks the number of matches between top 5 potential sources with the actual xml file information and retruns the status

        Args:
         top_k_documents: top 5 source documents retrieved
         xml_source_references: list of correct source file names

        Returns:
            string: corresponding status
        """

    matches=[]
    for document in top_k_documents:
        if document in xml_source_references:
            matches.append(document)

    if len(matches) == len(xml_source_references):
        return "Pass"
    elif len(matches) > 0:
        return "Soft Pass"
    else:
        return "Fail"

def tokenize_into_sentences(document):
    return nltk.sent_tokenize(document)

def compute_tfidf(source_documents):
    """
        This helper function generates a tfidf matrix for all source documents

        Args:
           source_documents: array of source file names

        Returns:
            tfidf_matrix: where each element represents the importance of a term within that document
            vectorizer: instance of TF-IDF
    """
    stop_words = list(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(source_documents)
    return tfidf_matrix, vectorizer

def compute_similarity_scores(sentence, tfidf_matrix, vectorizer):
    """
        This helper function compares sentence with each source document and generates a similarity score

        Args:
            sentence: sentence of a suspicious document
            tfidf_matrix: 2D array where each row is source document and column is a term
            vectorizer: TF-IDF Vectorizer to apply on sentence

        Returns:
            list: each element represents a similarity score between sentence and each source document
            """
    query_tfidf = vectorizer.transform([sentence])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    return similarities

def combsum_merge(all_similarity_scores, total_sentences):
    """
        This helper function combines similarity scores from
        multiple sentences of a suspicious document into a single score for each source document

        Args:
            all_similarity_scores: 2D matrix where each row is a sentence and each column is a source document
            total_sentences: total number of sentences of a suspicious document

        Returns:
            list: each element represents a combined similarity scores for each source document
        """
    combined_scores = np.sum(all_similarity_scores, axis=0) / total_sentences
    return combined_scores


def find_potential_source_documents(suspicious_file_path, source_directory_path, top_q=10, top_k=5):
    """
    This function generates the top_k potential source documents for a suspicious file

    Args:
        suspicious_file_path(string)-: suspicious document file name
        source_directory_path-(string): source document file name
        top_q(int)-: number of sentences to compare with
        top_k(int)-: number of potential source documents it retrieves

    Returns:
        list[list]: top_k source documents
    """
    with open(suspicious_file_path, 'r', encoding='utf-8') as file:
        suspicious_document = file.read()
    print(suspicious_file_path)
    source_documents, file_names = read_files(source_directory_path)
    sentences = tokenize_into_sentences(suspicious_document)
    tfidf_matrix, vectorizer = compute_tfidf(source_documents)

    all_similarity_scores = np.zeros((len(sentences), tfidf_matrix.shape[0]))
    for i, sentence in enumerate(sentences):
        similarity_scores = compute_similarity_scores(sentence, tfidf_matrix, vectorizer)
        top_q_indices = np.argsort(similarity_scores)[-top_q:]
        all_similarity_scores[i, top_q_indices] = similarity_scores[top_q_indices]

    combined_scores = combsum_merge(all_similarity_scores, len(sentences))
    top_k_indices = np.argsort(combined_scores)[-top_k:][::-1]
    top_k_documents = [file_names[i] for i in top_k_indices]
    print(top_k_documents)
    return top_k_documents

def main():
    # Dataset Link-: https://zenodo.org/records/3250095
    source_directory_path = 'pan-plagiarism-corpus-2011/pan-plagiarism-corpus-2011/external-detection-corpus/source-document'
    suspicious_directory_path = "suspicious-documents"
    results = []
    for suspicious_file in os.listdir(suspicious_directory_path):
        if suspicious_file.endswith('.txt'):
            suspicious_file_path = os.path.join(suspicious_directory_path, suspicious_file)
            top_5_documents = find_potential_source_documents(suspicious_file_path, source_directory_path)
            xml_file = suspicious_file.replace('.txt', '.xml')
            xml_file_path = os.path.join(suspicious_directory_path, xml_file)
            xml_source_references = parse_xml_file(xml_file_path)
            status = check_matches(top_5_documents, xml_source_references)

            correct_sources = set(top_5_documents) & set(xml_source_references)
            true_positives = len(correct_sources)
            false_positives = len(set(top_5_documents) - set(xml_source_references))
            false_negatives = len(set(xml_source_references) - set(top_5_documents))

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            results.append((suspicious_file, ', '.join(top_5_documents), status, precision, recall, f1))

    df = pd.DataFrame(results,columns=['Suspicious File', 'Top 5 Candidate Documents', 'Status', 'Precision', 'Recall','F1 Score'])
    output_file = 'candidate_documents.csv'
    df.to_csv(output_file, index=False)

    average_precision = df['Precision'].mean()
    average_recall = df['Recall'].mean()
    average_f1_score = df['F1 Score'].mean()

    # Prints the overall averages
    print(f"Average Precision: {average_precision:.4f}")
    print(f"Average Recall: {average_recall:.4f}")
    print(f"Average F1 Score: {average_f1_score:.4f}")

if __name__ == "__main__":
    main()