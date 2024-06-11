import os
import json

# Constants for directory and file paths
DATASET_DIRECTORY = "./Dataset_1"
ANNOTATIONS_FILE = "map.json"

# Constants for identifying the sections in the code
VULNERABILITY_MARKER = "VULNERABLE LINES"

def load_json_annotations(filepath):
    """
    Load annotations from a JSON file.

    :param filepath: Path to the JSON file containing annotations.
    :return: Dictionary with loaded annotations.
    """
    with open(filepath, 'r') as file:
        return json.load(file)

def get_context_lines(file_content, line_number, context_range=5):
    """
    Extract context lines around a specific line in a file.

    :param file_content: List of all lines in the file.
    :param line_number: The line number around which context is extracted.
    :param context_range: Number of lines before and after to include as context.
    :return: List of context lines with basic whitespace tokenization applied.
    """
    start = max(0, line_number - context_range - 1)
    end = min(len(file_content), line_number + context_range)
    return [' '.join(line.strip().split()) for line in file_content[start:end]]

import random

def enhance_annotations_with_negatives(annotations, dataset_directory, context_range=5, neg_samples_per_positive=1):
    """
    Enhance annotations with context lines for each file in the dataset and add negative samples.

    :param annotations: The original annotations without context.
    :param dataset_directory: Directory containing the dataset files.
    :param context_range: The number of lines before and after the vulnerable line to include as context.
    :param neg_samples_per_positive: Number of non-vulnerable (negative) samples to include per vulnerable line.
    :return: A dictionary with enhanced annotations including both vulnerable and non-vulnerable lines.
    """
    enhanced_annotations = {}
    for filename in os.listdir(dataset_directory):
        with open(os.path.join(dataset_directory, filename), 'r') as file:
            file_lines = file.readlines()
            file_annotations = annotations.get(filename, {})
            enhanced_annotations[filename] = {}

            for line_num, char_ranges in file_annotations.items():
                line_num_int = int(line_num)
                # Add vulnerable line with context
                context = get_context_lines(file_lines, line_num_int, context_range)
                enhanced_annotations[filename][line_num] = {
                    'context': context,
                    'char_ranges': char_ranges,
                    'is_vulnerable': 1
                }

                # Add non-vulnerable lines
                all_line_nums = range(len(file_lines))
                non_vul_lines = set(all_line_nums) - set(range(max(0, line_num_int - context_range), min(len(file_lines), line_num_int + context_range + 1)))

                for _ in range(neg_samples_per_positive):
                    non_vul_line_num = random.choice(list(non_vul_lines))
                    non_vul_context = get_context_lines(file_lines, non_vul_line_num, context_range)
                    enhanced_annotations[filename][str(non_vul_line_num)] = {
                        'context': non_vul_context,
                        'char_ranges': [],
                        'is_vulnerable': 0
                    }
                    # Update non_vul_lines to avoid picking the same line again
                    non_vul_lines.remove(non_vul_line_num)

    return enhanced_annotations


def save_enhanced_annotations(annotations, filepath):
    """
    Save the enhanced annotations to a JSON file.

    :param annotations: The enhanced annotations to save.
    :param filepath: Path where the enhanced annotations JSON should be saved.
    """
    with open(filepath, 'w') as file:
        json.dump(annotations, file, indent=4)

# Main script execution
if __name__ == "__main__":
    annotations = load_json_annotations(ANNOTATIONS_FILE)
    enhanced_annotations = enhance_annotations(annotations, DATASET_DIRECTORY)
    save_enhanced_annotations(enhanced_annotations, 'enhanced_annotations.json')

    # Print a sample of the enhanced annotations for demonstration
    for key, value in list(enhanced_annotations.items())[:1]:  # Only the first file's annotation for brevity
        print(key, "\n\t", json.dumps(value, indent=4))
