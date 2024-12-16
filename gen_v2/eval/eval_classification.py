import os
import argparse
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from tqdm import tqdm
import re
import numpy as np
from scipy.stats import pearsonr

def parse_text(input_text, file_path, error_files):
    parsed_data = {}

    input_text = input_text.lower()
    input_text = input_text.replace('*', '').replace('**', '').replace(' & ', '').replace('#', '')
    input_text = input_text.replace("'", '').replace('"', '').replace('<', '').replace('>', '')

    input_text = re.sub(r"label\s*:\s*", "label:", input_text)
    input_text = re.sub(r"confidence score\s*:\s*", "confidence score:", input_text)

    label_match = re.search(r"label:\s*(\[.*?\]|\w+)", input_text)
    if label_match:
        label_value = label_match.group(1)
        if label_value.startswith('['):
            parsed_data['Label'] = [item.strip().strip("'\"") for item in label_value.strip('[]').split(',')]
        else:
            parsed_data['Label'] = label_value

    try:
        confidence_score_match = re.search(r"confidence score:\s*([\d.]+)", input_text)
        if confidence_score_match:
            parsed_data['Confidence Score'] = float(confidence_score_match.group(1))
        else:
            raise ValueError("Confidence Score missing or invalid")
    except ValueError:
        error_files.append(os.path.basename(file_path))
        return None

    true_answer_match = re.search(r"trueanswer:\s*(\[[^\]]+\]|\w+)", input_text)
    if true_answer_match:
        true_answer_value = true_answer_match.group(1)
        if true_answer_value.startswith('['):
            parsed_data['TrueAnswer'] = \
            [item.strip().strip("'\"") for item in true_answer_value.strip('[]').split(',')][0]
        else:
            parsed_data['TrueAnswer'] = true_answer_value

    true_label_list_match = re.search(r"truelabellist:\[(.*?)\]", input_text)
    if true_label_list_match:
        parsed_data['TrueLabellist'] = [label.strip().strip("'\"") for label in
                                        true_label_list_match.group(1).split(',')]

    return parsed_data

def parser_txt(file_path, error_files):
    with open(file_path, 'r', encoding='utf-8') as file:
        input_text = file.read()

    parsed_data = parse_text(input_text, file_path, error_files)
    return parsed_data

def process_result_files(folder_path):
    all_files = os.listdir(folder_path)
    result_files = [file for file in all_files if 'answer' in file and file.endswith('.txt')]
    return result_files

def calculate_correlations(confidence_scores, accuracies):
    accuracy_corr, _ = pearsonr(confidence_scores, accuracies)
    return accuracy_corr


def save_results(folder_path, models, aggregated_results, label_names, folder_info):
    results_file = os.path.join(folder_path, 'aggregated_results.md')
    with open(results_file, 'w') as f:
        f.write(f"# Folder Info: SI={folder_info['SI']}, TQ={folder_info['TQ']}, PS={folder_info['PS']}, SHOT={folder_info['SHOT']}\n\n")
        for model, result in aggregated_results.items():
            f.write(f"# Model: {model}\n\n")
            f.write(f"**F1-macro:** {result['f1_macro']:.4f}\n\n")
            f.write(f"**F1-weight:** {result['f1_weighted']:.4f}\n\n")
            f.write(f"**Precision-macro:** {result['precision_macro']:.4f}\n\n")
            f.write(f"**Recall-macro:** {result['recall_macro']:.4f}\n\n")
            f.write(f"**Accuracy:** {result['accuracy']:.4f}\n\n")
            f.write(f"**Avg Confidence Score:** {result['avg_confidence']:.4f}\n\n")
            f.write(f"**Correlation between Confidence Score and Accuracy:** {result['accuracy_corr']:.4f}\n\n")
            f.write(f"**Processed files:** {result['processed_count']}\n\n")
            f.write(f"**Failed files:** {result['failed_count']:.4f}\n\n")
            if result['error_files']:
                f.write(f"**Error processing files:** {result['error_files']}\n\n")

            f.write("\n## Classification Report\n\n")

            f.write("```\n")
            report_lines = result['classification_report'].split('\n')
            for line in report_lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) == 5:
                        f.write(f"{parts[0]:<15} {parts[1]:<10} {parts[2]:<10} {parts[3]:<10} {parts[4]:<10}\n")
                    else:
                        f.write(f"{line}\n")
            f.write("```\n")
            f.write('\n---\n\n')


def main(args):
    path = os.path.join(args.base_folder_path, args.folder_path)
    models = ['GPT4o','Gemini','Ollama_Qwen','Ollama_Mistral']
    aggregated_results = {}

    label_names = []

    folder_info = args.folder_info

    for model in models:
        result_list = []
        label_list = []
        confidence_scores = []
        valid_confidence_scores = []
        valid_accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        accuracies = []
        processed_count = 0
        failed_count = 0
        error_files = []
        folder_path = os.path.join(path, model)
        files = process_result_files(folder_path)

        for answer_path in tqdm(files, desc=f"Processing {model} files"):
            labels = parser_txt(os.path.join(folder_path, answer_path), error_files)
            if labels is not None:
                try:
                    label = labels['Label']
                    if isinstance(label, list):
                        label = label[0]
                    true_label = labels['TrueAnswer']
                    label_list_ = labels['TrueLabellist']
                    confidence_score = labels['Confidence Score']

                    if not label_names:
                        label_names = label_list_

                    label_list_dict = {label: i for i, label in enumerate(label_list_)}

                    predicted_label = int(label_list_dict[label])
                    true_label_index = int(label_list_dict[true_label])

                    result_list.append(predicted_label)
                    label_list.append(true_label_index)

                    precision_value = precision_score([true_label_index], [predicted_label], average='macro', zero_division=0)
                    recall_value = recall_score([true_label_index], [predicted_label], average='macro', zero_division=0)
                    f1_score_value = f1_score([true_label_index], [predicted_label], average='macro', zero_division=0)
                    accuracy_value = accuracy_score([true_label_index], [predicted_label])

                    precisions.append(precision_value)
                    recalls.append(recall_value)
                    f1_scores.append(f1_score_value)
                    accuracies.append(accuracy_value)

                    valid_confidence_scores.append(confidence_score)
                    valid_accuracies.append(accuracy_value)
                    confidence_scores.append(confidence_score)
                    processed_count += 1

                except Exception as e:
                    error_files.append(answer_path)
                    failed_count += 1
            else:
                failed_count += 1

        if processed_count > 0:
            f1_macro = f1_score(label_list, result_list, average='macro', zero_division=0)
            f1_weighted = f1_score(label_list, result_list, average='weighted', zero_division=0)
            precision_macro = precision_score(label_list, result_list, average='macro', zero_division=0)
            recall_macro = recall_score(label_list, result_list, average='macro', zero_division=0)
            accuracy = accuracy_score(label_list, result_list)
            avg_confidence = np.mean(confidence_scores)

            try:
                accuracy_corr = calculate_correlations(valid_confidence_scores, valid_accuracies)
            except ValueError:
                accuracy_corr = None

            class_report = classification_report(label_list, result_list, target_names=label_names, zero_division=0, digits=4)
            conf_matrix = "N/A"  # confusion_matrix(label_list, result_list, labels=label_list)

        else:
            f1_macro = f1_weighted = precision_macro = recall_macro = accuracy = avg_confidence = accuracy_corr = 0.0
            conf_matrix = "N/A"
            class_report = "N/A"

        aggregated_results[model] = {
            'processed_count': processed_count,
            'failed_count': failed_count,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'accuracy_corr': accuracy_corr,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'error_files': f"[{','.join(error_files)}]" if error_files else "None"
        }

        print(f"Model: {model}")
        print(f"F1-macro: {f1_macro:.4f}")
        print(f"F1-weight: {f1_weighted:.4f}")
        print(f"Precision-macro: {precision_macro:.4f}")
        print(f"Recall-macro: {recall_macro:.4f}")
        print(f"ACC: {accuracy:.4f}")
        print(f"Avg Confidence Score: {avg_confidence:.4f}")
        print(f"Correlation between Confidence Score and ACC: {accuracy_corr:.4f}")
        print(f"Processed files: {processed_count}")
        print(f"Failed files: {failed_count}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Classification Report:\n{class_report}")
        if error_files:
            print(f"Error processing files: {aggregated_results[model]['error_files']}")
        print('-' * 50)

    save_results(path, models, aggregated_results, label_names, folder_info)


def extract_folder_info(folder_path):
    parts = folder_path.split('\\')
    if len(parts) >= 4:
        SI = parts[-3]
        TQ = parts[-2]
        PS_SHOT = parts[-1].split('-')
        PS = PS_SHOT[1] if len(PS_SHOT) > 1 else "Unknown"
        SHOT = PS_SHOT[-1]
    else:
        SI = TQ = PS = SHOT = "Unknown"

    return {"SI": SI, "TQ": TQ, "PS": PS, "SHOT": SHOT}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process result files and calculate metrics.')

    parser.add_argument('--base_folder_path', type=str, required=False,
                        default='../../results/cssrs/Classification',
                        help='Base Path to the folder containing result files')

    args = parser.parse_args()

    base_folder_path = os.path.abspath(args.base_folder_path)

    first_level_subfolders = [os.path.join(base_folder_path, f.name) for f in os.scandir(base_folder_path) if
                              f.is_dir()]

    for first_level_folder in first_level_subfolders:
        second_level_subfolders = [os.path.join(first_level_folder, f.name) for f in os.scandir(first_level_folder) if
                                   f.is_dir()]

        for second_level_folder in second_level_subfolders:
            third_level_subfolders = [os.path.join(second_level_folder, f.name) for f in os.scandir(second_level_folder) if
                                      f.is_dir()]

            for third_level_folder in third_level_subfolders:
                folder_info = extract_folder_info(third_level_folder)
                args.folder_path = third_level_folder
                args.folder_info = folder_info
                print(f"Processing folder: {args.folder_path}")
                main(args)
