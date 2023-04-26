import pandas as pd

data = pd.read_json("./x1.json")
data.head()
headlines = list(data['headline'])
labels = list(data['is_sarcastic'])

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(oov_token="<oov>")
tokenizer.fit_on_texts(headlines)

word_index = tokenizer.word_index
# print(word_index)

seqs = tokenizer.texts_to_sequences(headlines)
padded_seqs = pad_sequences(seqs, padding="post")

print(padded_seqs[0])