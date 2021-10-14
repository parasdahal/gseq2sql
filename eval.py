from pathlib import Path
from datasets.schema_info import SchemaInfo
from transformers import BertTokenizer
import pandas as pd
import Levenshtein
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import os
import csv
import sqlite3
import sys
import json

dataset_path = os.path.join(Path(__file__).parent.absolute(), './data/spider')
# dataset_path = "./datasets/spider"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Add schema tokens to tokenizer
schema_info = SchemaInfo(os.path.join(dataset_path, 'tables.json'))
tokenizer.add_tokens(['[T]', '[C]'])
tokenizer.add_tokens(schema_info.get_tokens())


# Function which summarizes 3 metrics for results
# Prints and returns exact match accuracy, exact set match accuracy
# and average Levenshtein similarity
def summarize_query_results(csv_fname):
    exact_match_acc = eval_exact_match_accuracy(csv_fname)
    print(f'Exact Match Accuracy: {exact_match_acc}')

    set_match_acc = eval_set_match_accuracy(csv_fname)
    print(f'Exact Set Match Accuracy: {set_match_acc}')

    sim_score = eval_query_similarity(csv_fname)
    print(f'Average Levenshtein similarity: {sim_score}')

    execution_success, execution_accuracy = eval_execution_accuracy(csv_fname)
    print(f'Execution success: {execution_success}')
    print(f'Execution accuracy: {execution_accuracy}')

    return exact_match_acc, set_match_acc, sim_score

def eval_query_similarity(csv_fname, split='validation'):
    """
    Evaluates the similarity by means of Levenshtein distance between 
    predicted and ground-truth SQL queries (Lower == better)
        Input:
            csv_fname: filename of the csv to read for predicted SQL queries
            split: which split to use for the ground-truth SQL queries
        Returns:
            Average Levenshtein distance between all pairs
    """
    # Read predicted and gt queries
    pred_queries, _, gt_queries = read_csv(csv_fname)

    # Loop over each instance of the dataset
    similarities = []
    for i in range(len(gt_queries)):
        # Read gt and predicted query
        gt_query = gt_queries[i]
        pred_query = pred_queries[i]

        # Convert ids to string, calculate levenshtein distance between the pred and gt query
        pred_query = ids_to_string(pred_query)
        similarity = Levenshtein.distance(gt_query, pred_query)

        similarities.append(similarity)

    # Return the average similarity of all queries
    avg_similarity = np.mean(similarities)
    return avg_similarity

def eval_exact_match_accuracy(csv_fname, split='validation'):
    """
    Evaluates the exact matches between predicted and ground-truth
    SQL queries
        Input:
            csv_fname: filename of the csv to read for predicted SQL queries
            split: which split to use for the ground-truth SQL queries
        Returns:
            Accuracy of exact matches of predicted SQL queries to the gt queries
            wrt the specified split
    """
    # Read predicted and gt queries
    pred_queries, gt_queries, _ = read_csv(csv_fname)

    # N = number of queries in total to evaluate
    N = len(pred_queries)
    # L = max length of sequences for padding pred queries
    L = len(gt_queries[0])

    # Convert queries to tensors
    gt_queries = [torch.tensor(query) for query in gt_queries]
    gt_queries = pad_sequence(gt_queries, batch_first=True)
    pred_queries = [torch.tensor(query) for query in pred_queries]
    pred_queries = pad_sequence(pred_queries, batch_first=True)

    # Pad the shorter tensor to match the size of the longer tensor
    if gt_queries.shape[1] > pred_queries.shape[1]:
        L = gt_queries.shape[1]
        pred_queries = F.pad(pred_queries, (0, (L - pred_queries.shape[1])))
    elif gt_queries.shape[1] < pred_queries.shape[1]:
        L = pred_queries.shape[1]
        gt_queries = F.pad(gt_queries, (0, (L - gt_queries.shape[1])))

    # Create tensor with 1's for incorrect predictions, 0's for correct predictions of SQL queries
    incorrect_pred = torch.sum(pred_queries != gt_queries, dim=1)

    # Count number of correct prediction for all queries, and calulate the overall accuracy
    correct_pred = torch.sum(incorrect_pred == 0)
    accuracy = correct_pred/N

    return accuracy

def eval_set_match_accuracy(csv_fname, split='validation'):
    """
    Evaluates the exact set matches between predicted and ground-truth
    SQL queries. This calculate which tokens are correctly predicted,
    irregardless of the order.
        Input:
            csv_fname: filename of the csv to read for predicted SQL queries
            split: which split to use for the ground-truth SQL queries
        Returns:
            Accuracy of exact set matches of predicted SQL queries to the gt queries
            wrt the specified split
    """
    # Read predicted and gt queries
    pred_queries, _, original_queries = read_csv(csv_fname)

    # Loop over each instance of the dataset
    set_accuracies = []
    for i in range(len(original_queries)):
        # Read gt and predicted query
        gt_query = original_queries[i]
        pred_query = pred_queries[i]

        # Convert to string, then split and convert to a set
        pred_query = ids_to_string(pred_query)
        gt_query, pred_query = set(gt_query.split(' ')), set(pred_query.split(' '))

        # Calculate accuracy by (intersection / union) of the two sets
        set_accuracy = len(gt_query.intersection(pred_query)) / len(gt_query.union(pred_query))
        set_accuracies.append(set_accuracy)

    # Return the average set match accuracy
    avg_set_match_accuracy = np.mean(set_accuracies)
    return avg_set_match_accuracy

def eval_execution_accuracy(csv_fname):
    total = 0
    fails = 0
    accurate = 0

    with open(csv_fname) as f:
        reader = csv.reader(f)
        for row in reader:
            total += 1
            db_file = os.path.join(dataset_path, 'database', row[2], f'{row[2]}.sqlite')
            con = sqlite3.connect(db_file)
            cur = con.cursor()
            cur.execute(row[4])
            target = cur.fetchall()[0]
            try:
                cur.execute(row[3])
                pred = cur.fetchall()[0]
                if target == pred:
                    accurate += 1
            except:
                fails += 1

    execution_success = (total-fails)/total
    execution_accuracy = accurate/total

    return execution_success, execution_accuracy

# Helper function which uses tokenizer to convert ids to strings
def ids_to_string(ids):
    return tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True))

# Helper function which reads CSV, returns query ids of shape (dataset_size x sentence length)
def read_csv(csv_fname):
    results_csv = pd.read_csv(csv_fname, header=None)
    pred_queries = results_csv.iloc[:,0]
    pred_queries = [row.replace('[','').replace(']','').split(', ') for i, row in pred_queries.iteritems()]
    pred_queries = [[int(id) for id in query] for query in pred_queries]

    gt_queries = results_csv.iloc[:,1]
    gt_queries = [row.replace('[','').replace(']','').split(', ') for i, row in gt_queries.iteritems()]
    gt_queries = [[int(id) for id in query] for query in gt_queries]

    original_queries = results_csv.iloc[:,4]
    original_queries = [row for i, row in original_queries.iteritems()]

    return pred_queries, gt_queries, original_queries

def save_string_csv(csv_fname):
    pred_queries, gt_queries, _ = read_csv(csv_fname)

    query_strings = []
    for pred_q, gt_q in zip(pred_queries, gt_queries):
        pred_q, gt_q = ids_to_string(pred_q), ids_to_string(gt_q)

        query_strings.append((pred_q, gt_q))
        
    df = pd.DataFrame(query_strings)
    df.to_csv('queries_as_strings.csv')

if __name__ == '__main__':
    # from pprint import pprint
    # pprint(get_query_id_dictionary(['train_spider.json', 'dev.json']))
    if len(sys.argv) < 1:
        print('Use csv file path as an argument')
    summarize_query_results(sys.argv[1])