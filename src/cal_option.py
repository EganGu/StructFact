import re
import pickle
import argparse
import pandas as pd
import numpy as np
from pandas import DataFrame, merge
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from common import OPTION_MAP
from sklearn.metrics import accuracy_score
# from sklearn.metrics import balanced_accuracy_score as accuracy_score


SOURCE_COLS = ['ANSWER', 'TABLE', 'CLAIM', 'CHALLENGE', 'LABEL', 'QUESTION', 'SOURCE', 'EVIDENCE', 'len_evidence', 'SHUFFLED_ROWS', "DENSITY", 'NEW_EVIDENCE']


def extract_abc_letters(input_string):
    # 将字符串按照空格符和换行符分割成单词
    words = input_string.split()

    # 初始化一个空列表用于存储候选字母
    candidates = []

    # 遍历每个单词，检查是否仅包含一个字母且为A、B、C之一
    for word in words:
        # 使用正则表达式去除非字母字符
        clean_word = re.sub(r'[^A-Za-z]', '', word)
        if len(clean_word) == 1 and clean_word in 'ABC':
            candidates.append(clean_word)

    # 返回所有候选字母的列表
    return candidates


# Function to find the label from the response
def find_option_forward(response: str):
    if type(response) != str:
        return 'NONE'
    words = extract_abc_letters(response)
    for word in words:
        for key in ['A', 'B', 'C']:
            for char in word:
                if char == key:
                    return key
    return 'NONE'

def find_option_backward(response: str):
    # remove the repeated part
    if type(response) != str:
        return 'NONE'

    words = extract_abc_letters(response)
    for word in words[::-1]:
        for key in ['A', 'B', 'C']:
            for char in word:
                if char == key:
                    return key
    # print(response)
    return 'NONE'

def load_data(data_path):
    data = pickle.load(open(data_path, 'rb'))
    data_df = DataFrame(data.values())
    return data_df


def process_data(args):
    # Load data into a DataFrame
    data_df = load_data(args.data_path)

    map_cols = [c for c in data_df.columns if c not in SOURCE_COLS]
    map_cols_rename = [f"{c}_label" for c in map_cols]

    # Print data distribution
    if args.output_distribution:
        print('Data distribution:')
        print(data_df['LABEL'].value_counts())

    # Calculate and store metrics for each column
    metrics = []
    distribution_data = []

    option_map = {k.split('_')[-1].upper():v.upper() for k, v in OPTION_MAP[args.option_map].items()}
    option_map['NONE'] = 'NONE'
    for col, col_rename in zip(map_cols, map_cols_rename):
        if 'cot' in col or 'refine' in args.data_path:
            options = data_df[col].apply(find_option_backward)
        else:
            options = data_df[col].apply(find_option_forward)

        data_df[col_rename] = options.map(option_map)
        accuracy = accuracy_score(data_df['LABEL'], data_df[col_rename]) * 100
        f1 = f1_score(data_df['LABEL'], data_df[col_rename], average=args.f1_average, zero_division=0) * 100
        precision = precision_score(data_df['LABEL'], data_df[col_rename], average=args.f1_average, zero_division=0) * 100
        recall = recall_score(data_df['LABEL'], data_df[col_rename], average=args.f1_average, zero_division=0) * 100
        metrics.append({
            'Challenge': 'Overall',
            'Model': col_rename,
            'Accuracy (%)': round(accuracy, 2),
            'F1 Score (%)': round(f1, 2),
            'Precision (%)': round(precision, 2),
            'Recall (%)': round(recall, 2)
        })
        if args.output_distribution:
            distribution = data_df[col_rename].value_counts(dropna=False).reset_index()
            distribution.columns = ['Label', f'{col_rename} Count']
            distribution_data.append(distribution)
    # Merge all distribution data into one DataFrame
    if args.output_distribution:
        distribution_df = distribution_data[0]
        for dist in distribution_data[1:]:
            distribution_df = merge(distribution_df, dist, on='Label', how='outer')

        # Fill NaN values with zeros
        distribution_df.fillna(0, inplace=True)

        # Print distribution table
        print("\nDistribution Table:")
        print(distribution_df)

    # Calculate and print metrics for each challenge
    challenge_metrics = []

    for challenge in data_df['CHALLENGE'].unique():
        challenge_data = data_df[data_df['CHALLENGE'] == challenge]
        for col, col_rename in zip(map_cols, map_cols_rename):
            accuracy = accuracy_score(challenge_data['LABEL'], challenge_data[col_rename]) * 100
            f1 = f1_score(challenge_data['LABEL'], challenge_data[col_rename], average=args.f1_average, zero_division=0) * 100
            precision = precision_score(challenge_data['LABEL'], challenge_data[col_rename], average=args.f1_average, zero_division=0) * 100
            recall = recall_score(challenge_data['LABEL'], challenge_data[col_rename], average=args.f1_average, zero_division=0) * 100
            challenge_metrics.append({
                'Challenge': challenge,
                'Model': col_rename,
                'Accuracy (%)': round(accuracy, 2),
                'F1 Score (%)': round(f1, 2),
                'Precision (%)': round(precision, 2),
                'Recall (%)': round(recall, 2)
            })


    # Combine overall and challenge metrics
    combined_metrics_df = DataFrame(metrics + challenge_metrics)
    # Print combined metrics
    print("\nCombined Metrics Table:")
    print(combined_metrics_df)
    # breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and calculate metrics.")
    parser.add_argument("data_path", type=str, help="Path to the data file.")
    parser.add_argument("--target", type=str, default="main", help="The target of analysis.")
    parser.add_argument("--f1_average", type=str, default="weighted", help="F1 score average method. Default is 'weighted'.")
    parser.add_argument("--output_distribution", action="store_true", help="Flag to output data distribution.")
    parser.add_argument(
        "--option_map", type=int, default=0, help="option mapping."
    )

    args = parser.parse_args()
    if args.target == 'main':
        process_data(args)
