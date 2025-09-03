import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import xml.etree.ElementTree as ET
import pandas as pd
import os

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    plagiarized_passages = []
    for feature in root.findall(".//feature[@name='plagiarism']"):
        start = int(feature.get('this_offset'))
        length = int(feature.get('this_length'))
        plagiarized_passages.append((start, start + length))
    return plagiarized_passages

def get_passages(text, window_size=3):
    """
          This function generates and returns the passages of length=3 sentences

          Args:
              text(string)-: textual content of the file
              window_size(int)-: sliding window size

          Returns:
              tuple-: passage,starting index of the passage
    """
    sentences = text.split('.')
    passages = []
    start_indices = []
    # Sliding Window Technique of size=3 sentences
    for i in range(len(sentences) - window_size + 1):
        passage = '. '.join(sentences[i:i + window_size]).strip()
        passages.append(passage)
        start_indices.append(sum(len(s) + 1 for s in sentences[:i]))
    return passages, start_indices

def find_most_similar_passages(suspicious_file, source_file, window_size=3, top_n=5, similarity_threshold=0.75):
    """
           This function generates and returns the top_n similar passages for a source file

           Args:
               suspicious_file(string)-: suspicious document file name
               source_file-(string)-: source document file name
               window_size-:  sliding window size
               top_n(int)-: number of topmost similar passages for a source file

           Returns:
               list: top_n similar pairs
    """
    text1 = read_file(suspicious_file)
    text2 = read_file(source_file)

    passages1, indices1 = get_passages(text1, window_size)
    passages2, indices2 = get_passages(text2, window_size)

    vectorizer = TfidfVectorizer().fit(passages1 + passages2)
    vectors1 = vectorizer.transform(passages1)
    vectors2 = vectorizer.transform(passages2)

    similarity_matrix = cosine_similarity(vectors1, vectors2)

    similar_pairs = []
    for _ in range(top_n):
        max_sim = np.max(similarity_matrix)
        if max_sim < similarity_threshold:
            break
        i, j = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        similar_pairs.append((passages1[i], passages2[j], max_sim, indices1[i], indices2[j]))
        similarity_matrix[i, j] = 0


    return similar_pairs


def process_source_files(suspicious_file, source_file_paths, window_size=3, top_n=5, similarity_threshold=0.75):
    """
            This function generates and returns the final source documents for every suspicious file

            Args:
                suspicious_file(string)-: suspicious document file name
                source_file_paths-(dict)-: source files
                window_size-:  sliding window size
                top_n(int)-: number of topmost similar passages for a source file
                similarity_threshold(float)-:75%

            Returns:
                list[list]: final source documents for suspicious files
    """
    results = []

    for file_name, source_file_path in source_file_paths.items():
        similar_passages = find_most_similar_passages(suspicious_file, source_file_path, window_size, top_n,
                                                      similarity_threshold)

        if similar_passages and similar_passages[0][2] >= similarity_threshold:
            # Append the file name and similar passages to the results
            results.append((file_name, similar_passages))

    # Sort results by highest similarity score
    results.sort(key=lambda x: x[1][0][2], reverse=True)

    return results

def highlight_passages(file_path, similar_passages, file_num, colors):
    """
        This function highlights the top similar passages in a given file

        Args:
            file_path(string):  file name
            similar_passages(list): list of topmost similar passages in file
            file_num(int)-: starting index of passage
            colors(list)-: list of colors

        Returns:
            string: highlighted file
    """
    text = read_file(file_path)
    reset_color = '\033[0m'
    sorted_passages = sorted(
        [(p[file_num - 1], p[file_num + 2], i % len(colors)) for i, p in enumerate(similar_passages)],
        key=lambda x: x[1]
    )

    color_indices = []
    for passage, start_index, color_index in sorted_passages:
        color_indices.append((start_index, True, colors[color_index]))
        color_indices.append((start_index + len(passage), False, colors[color_index]))

    color_indices.sort(key=lambda x: (x[0], not x[1]))

    highlighted_text = ""
    current_index = 0
    active_colors = []

    for index, is_start, color in color_indices:
        highlighted_text += text[current_index:index]

        if is_start:
            if not active_colors:
                highlighted_text += color
            active_colors.append(color)
        else:
            active_colors.remove(color)
            if active_colors:
                highlighted_text += active_colors[-1]
            else:
                highlighted_text += reset_color

        current_index = index
    highlighted_text += text[current_index:]

    highlighted_directory = os.path.join(os.getcwd(), 'highlighted-files')
    os.makedirs(highlighted_directory, exist_ok=True)

    highlighted_filename = f"color_{os.path.basename(file_path)}"
    highlighted_file_path = os.path.join(highlighted_directory, highlighted_filename)

    with open(highlighted_file_path, 'w', encoding='utf-8') as file:
        file.write(highlighted_text)

    return highlighted_filename

def highlight_all_passages(suspicious_file, plagiarism_results, source_file_paths,colors):
    # Highlight suspicious file
    all_passages = [passage for _, similar_passages in plagiarism_results for passage in similar_passages]
    highlight_passages(suspicious_file, all_passages, 1, colors)

    # Highlight source files
    for source_file_name, similar_passages in plagiarism_results:
        full_path = source_file_paths[source_file_name]
        highlight_passages(full_path, similar_passages, 2, colors)


def calculate_metrics(detected_passages, actual_passages):
    detected_chars = set()
    actual_chars = set()
    correct_chars = set()

    for start, end in detected_passages:
        detected_chars.update(range(start, end))

    for start, end in actual_passages:
        actual_chars.update(range(start, end))

    correct_chars = detected_chars.intersection(actual_chars)

    precision = len(correct_chars) / len(detected_chars) if detected_chars else 0
    recall = len(correct_chars) / len(actual_chars) if actual_chars else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    detections = len(detected_passages)
    overlapping = sum(1 for d in detected_passages if any(a[0] <= d[1] and d[0] <= a[1] for a in actual_passages))
    granularity = detections / overlapping if overlapping > 0 else 1

    plagdet = f1 / (np.log2(1 + granularity) if granularity > 1 else 1)

    return precision, recall, f1, granularity, plagdet

def calculate_average_metrics(metrics_list):
    num_pairs = len(metrics_list)
    avg_precision = sum(m[0] for m in metrics_list) / num_pairs
    avg_recall = sum(m[1] for m in metrics_list) / num_pairs
    avg_f1 = sum(m[2] for m in metrics_list) / num_pairs
    avg_granularity = sum(m[3] for m in metrics_list) / num_pairs
    avg_plagdet = sum(m[4] for m in metrics_list) / num_pairs
    return avg_precision, avg_recall, avg_f1, avg_granularity, avg_plagdet

def find_source_files_recursively(root_folder, file_names):
    source_files = {}
    for root, dirs, files in os.walk(root_folder):
        for file_name in file_names:
            if file_name in files:
                full_path = os.path.join(root, file_name)
                source_files[file_name] = full_path
    return source_files

def main():
    file_path = 'candidate_documents.csv'
    excel_data = pd.read_csv(file_path)
    current_directory = os.getcwd()
    suspicious_folder = os.path.join(current_directory, 'suspicious-documents')
    # Dataset Download link-: https://zenodo.org/records/3250095
    source_folder = r'pan-plagiarism-corpus-2011/pan-plagiarism-corpus-2011/external-detection-corpus/source-document'
    results = []
    colors = [
        '\033[91m',  # Red
        '\033[92m',  # Green
        '\033[93m',  # Yellow
        '\033[94m',  # Blue
        '\033[95m',  # Magenta
    ]

    for index, row in excel_data.iterrows():
        suspicious_file = row['Suspicious File']
        source_files = row['Top 5 Candidate Documents'].split(', ')

        # Get actual plagiarism passages for the suspicious file
        suspicious_file_path = os.path.join(suspicious_folder, suspicious_file)

        source_file_paths = find_source_files_recursively(source_folder, source_files)
        print(source_file_paths)
        xml_file = suspicious_file_path.replace('.txt', '.xml')
        try:
            actual_passages = parse_xml(xml_file)
        except FileNotFoundError:
            print(f"XML file not found for {suspicious_file}, skipping this file.")
            continue

        # Process each source file and calculate similarity metrics
        plagiarism_results = process_source_files(suspicious_file_path, source_file_paths)
        detected_passages = [
            (start_index, start_index + len(suspicious_passage))
            for _, similar_passages in plagiarism_results
            for suspicious_passage, _, _, start_index, _ in similar_passages
        ]

        # Calculate metrics for each detected passage
        metrics_list = []
        for _, similar_passages in plagiarism_results:
            metrics = calculate_metrics(detected_passages, actual_passages)
            metrics_list.append(metrics)

        # Calculate average metrics
        if metrics_list:
            avg_precision, avg_recall, avg_f1, avg_granularity, avg_plagdet = calculate_average_metrics(metrics_list)
        else:
            avg_precision = avg_recall = avg_f1 = avg_granularity = avg_plagdet = 0

        print(f"{suspicious_file_path}: {avg_precision},{avg_f1},{avg_granularity},{avg_plagdet}")
        # Highlight passages in the files
        for source_file, similar_passages in plagiarism_results:
            for suspicious_passage, source_passage, similarity, suspicious_start_index, source_start_index in similar_passages:
                print(f"\nSuspicious Passage:\n{suspicious_passage}\n")
                print(f"Source Passage:\n{source_passage}\n")
                print(f"Similarity: {similarity:.4f}")

        highlight_all_passages(suspicious_file_path, plagiarism_results, source_file_paths,colors)
        filtered_source_files = [file_name for file_name, _ in plagiarism_results]
        results.append([
            suspicious_file,
            ", ".join(filtered_source_files),
            avg_precision,
            avg_recall,
            avg_f1,
            avg_granularity,
            avg_plagdet
        ])

    # Save results to a CSV file
    output_csv_path = 'plagiarism_detection_results.csv'
    header = ['Suspicious File', 'Correct Source Files', 'Average Precision', 'Average Recall', 'Average F1 Score',
              'Average Granularity', 'Average Plagdet Score']

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        csvwriter.writerows(results)

    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    main()