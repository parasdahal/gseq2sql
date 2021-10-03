from data.dataset import SpiderDataset
from transformers import BertTokenizer
import pandas as pd
import Levenshtein
import numpy as np
import torch

dataset_path = './data/spider'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
    pred_queries = read_csv(csv_fname)
    valid_dataset = SpiderDataset(dataset_path,'dev.json')

    # Loop over each instance of the dataset
    similarities = []
    for i in range(valid_dataset.__len__()):
        # Read gt and predicted query
        ids, mask, gt_query, _ = valid_dataset.__getitem__(i)
        pred_query = pred_queries[i]

        # Convert ids to string, calculate levenshtein distance between the pred and gt query
        gt_query, pred_query = ids_to_string(gt_query), ids_to_string(pred_query)
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
    pred_queries = read_csv(csv_fname)
    valid_dataset = SpiderDataset(dataset_path,'dev.json')
    gt_queries = valid_dataset.queries

    # N = number of queries in total to evaluate
    N = len(pred_queries)

    # Convert queries to tensors
    pred_queries, gt_queries = torch.tensor(pred_queries), torch.tensor(gt_queries)

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
    pred_queries = read_csv(csv_fname)
    dataset = SpiderDataset(dataset_path,'dev.json')

    # Loop over each instance of the dataset
    set_accuracies = []
    for i in range(dataset.__len__()):
        # Read gt and predicted query
        ids, mask, gt_query, _ = dataset.__getitem__(i)
        pred_query = pred_queries[i]

        # Convert to string, then split and convert to a set
        gt_query, pred_query = ids_to_string(gt_query), ids_to_string(pred_query)
        gt_query, pred_query = set(gt_query.split(' ')), set(pred_query.split(' '))

        # Calculate accuracy by (intersection / union) of the two sets
        set_accuracy = len(gt_query.intersection(pred_query)) / len(gt_query.union(pred_query))
        set_accuracies.append(set_accuracy)

    # Return the average set match accuracy
    avg_set_match_accuracy = np.mean(set_accuracies)
    return avg_set_match_accuracy

# Helper function which uses tokenizer to convert ids to strings
def ids_to_string(ids):
    return tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True))

# Helper function which reads CSV, returns query ids of shape (dataset_size x sentence length)
def read_csv(csv_fname):
    pred_queries = pd.read_csv(csv_fname)
    pred_queries = [row[1].split(' ') for idx, row in pred_queries.iterrows()]
    pred_queries = [[int(id) for id in query] for query in pred_queries]

    return pred_queries

# Only for testing before having access to actual decoder output CSVs
# Loads the validation set ids into a csv
def dump_test_csv():
    valid_dataset = SpiderDataset(dataset_path,'dev.json')
    queries_ids = valid_dataset.queries
    queries_ids = [[str(id) for id in query] for query in queries_ids]
    queries_ids = [" ".join(query) for query in queries_ids]
    df = pd.DataFrame(queries_ids)
    df.to_csv('dev_ids.csv')


# dump_test_csv()

# eval_query_similarity('dev_ids.csv')

# eval_exact_match_accuracy('dev_ids.csv')

eval_set_match_accuracy('dev_ids.csv')