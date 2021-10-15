# DB Schema + Query Sequence to SQL

Code for DL4NLP project by Paras Dahal, Koen Gommers, Kevin Waller, Danai Xezonaki. Using custom seq2seq model for the Text2SQL task. Baseline model consists of BERT as an encoder and a LSTM with attention as a decoder. We extend our baseline model by concatenating schema information to the question input.

## Instructions

[Download Spider dataset](https://yale-lily.github.io/spider) and put the data in a `data/spider` directory.

Install required packages 

    pip install -r requirements.txt

Training (all arguments can be found in `utils.py`)

    python train.py --seed=42 --use_schema=False

By default, output will be saved in the `logs` directory.

Output CSVs can be evaluated using the `eval.py`

    python eval.py outputs.csv
