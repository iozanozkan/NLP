import re
import pandas as pd

real=pd.read_csv('True.csv')

df=real["text"]

def generate_ngrams(s, n):
    s = s.lower()
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    
    tokens = [token for token in s.split(" ") if token != ""]
    
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

for x in df:
  print(generate_ngrams(x, n=3))