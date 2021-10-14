from eval import read_csv
from transformers import BertTokenizer

sql_tokens = ["from", "where", "by", "as", "select"]
wrongs = {}
sql_freqs = {}
tok = BertTokenizer.from_pretrained('bert-base-uncased')

csv_fname = "outputs-17.csv"
pred_queries, gt_queries = read_csv(csv_fname)

for pred, gt in zip(pred_queries, gt_queries):
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

    if pred!=gt:
        try:
            wrongs[str(count)]+=1
        except:
            wrongs[str(count)]=1

for key in wrongs:
    wrongs[key] /= sql_freqs[key]

print(sql_freqs)
print(wrongs)