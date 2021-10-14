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
    _, _, _, pred_queries, gt_queries = read_csv(csv_fname)

    # Loop over each instance of the dataset
    similarities = []
    for i in range(len(gt_queries)):
        # Read gt and predicted query
        gt_query = gt_queries[i].lower()
        pred_query = pred_queries[i]

        # Convert ids to string, calculate levenshtein distance between the pred and gt query
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
    pred_ids, gt_ids, _, _, _ = read_csv(csv_fname)

    # N = number of queries in total to evaluate
    N = len(pred_ids)
    # L = max length of sequences for padding pred queries
    L = len(gt_ids[0])

    # Convert queries to tensors
    gt_ids = [torch.tensor(query) for query in gt_ids]
    gt_ids = pad_sequence(gt_ids, batch_first=True)
    pred_ids = [torch.tensor(query) for query in pred_ids]
    pred_ids = pad_sequence(pred_ids, batch_first=True)

    # Pad the shorter tensor to match the size of the longer tensor
    if gt_ids.shape[1] > pred_ids.shape[1]:
        L = gt_ids.shape[1]
        pred_ids = F.pad(pred_ids, (0, (L - pred_ids.shape[1])))
    elif gt_ids.shape[1] < pred_ids.shape[1]:
        L = pred_ids.shape[1]
        gt_ids = F.pad(gt_ids, (0, (L - gt_ids.shape[1])))

    # Create tensor with 1's for incorrect predictions, 0's for correct predictions of SQL queries
    incorrect_pred = torch.sum(pred_ids != gt_ids, dim=1)

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
    _, _, _, pred_queries, gt_queries = read_csv(csv_fname)

    # Loop over each instance of the dataset
    set_accuracies = []
    for i in range(len(gt_queries)):
        # Read gt and predicted query
        gt_query = gt_queries[i].lower()
        pred_query = pred_queries[i]

        # Convert to string, then split and convert to a set
        gt_query, pred_query = set(gt_query.split(' ')), set(pred_query.split(' '))

        # Calculate accuracy by (intersection / union) of the two sets
        set_accuracy = len(gt_query.intersection(pred_query)) / len(gt_query.union(pred_query))
        set_accuracies.append(set_accuracy)

    # Return the average set match accuracy
    avg_set_match_accuracy = np.mean(set_accuracies)
    return avg_set_match_accuracy

def eval_execution_accuracy(csv_fname):
    _, _, db_ids, pred_queries, gt_queries = read_csv(csv_fname)

    total = 0
    fails = 0
    accurate = 0

    for db_id, pred_query, gt_query in zip(db_ids, pred_queries, gt_queries):
        db_file = os.path.join(dataset_path, 'database', db_id, f'{db_id}.sqlite')
        con = sqlite3.connect(db_file)
        try:
            cur = con.cursor()
            cur.execute(gt_query)
            result = cur.fetchall()
            target = result[0] if len(result) >= 1 else None
            total += 1
        except:
            continue

        try:
            cur.execute(pred_query)
            result = cur.fetchall()
            pred = result[0] if len(result) >= 1 else None
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

    pred_ids = results_csv.iloc[:,0]
    pred_ids = [row.replace('[','').replace(']','').split(', ') for i, row in pred_ids.iteritems()]
    pred_ids = [[int(id) for id in query] for query in pred_ids]

    gt_ids = results_csv.iloc[:,1]
    gt_ids = [row.replace('[','').replace(']','').split(', ') for i, row in gt_ids.iteritems()]
    gt_ids = [[int(id) for id in query] for query in gt_ids]

    db_ids = results_csv.iloc[:,2]
    db_ids = [row for i, row in db_ids.iteritems()]

    pred_queries = results_csv.iloc[:,3]
    pred_queries = [row for i, row in pred_queries.iteritems()]

    gt_queries = results_csv.iloc[:,4]
    gt_queries = [row for i, row in gt_queries.iteritems()]

    return pred_ids, gt_ids, db_ids, pred_queries, gt_queries

def save_string_csv(csv_fname):
    pred_queries, gt_queries, _ = read_csv(csv_fname)

    query_strings = []
    for pred_q, gt_q in zip(pred_queries, gt_queries):
        pred_q, gt_q = ids_to_string(pred_q), ids_to_string(gt_q)

        query_strings.append((pred_q, gt_q))
        
    df = pd.DataFrame(query_strings)
    df.to_csv('queries_as_strings.csv')

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Use csv file path as an argument')
        exit()
    summarize_query_results(sys.argv[1])
