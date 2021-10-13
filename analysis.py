from eval import read_csv
from transformers import BertTokenizer

sql_tokens = ["from", "where", "by", "as", "select"]

sql_freqs = {}
tok = BertTokenizer.from_pretrained('bert-base-uncased')

csv_fname = "outputs.csv"
pred_queries, gt_queries = read_csv(csv_fname)

for pred, gt in zip(pred_queries, gt_queries):
    if pred!=gt:
        count = 0
        grnd = tok.convert_ids_to_tokens(gt[2:])
        for tkn in sql_tokens:
            if tkn in grnd:
                # counts how many special SQL tokens exist in the query
                count+=1
        try:
            sql_freqs[str(count)]+=1
        except:
            sql_freqs[str(count)]=1

print(sql_freqs)