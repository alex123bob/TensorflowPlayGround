from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_sentences = [
    'It will rain',
    'The weather is cloudy!',
    'Will it be raining today?',
    'It is a super hot day!'
]

tokenizer = Tokenizer(num_words=100, oov_token="<oov>")
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(train_sentences)
padded_seqs = pad_sequences(sequences)

print(word_index)
print(train_sentences)
print(sequences)
print(padded_seqs)

padded_seqs = pad_sequences(sequences, padding="post", maxlen=5, truncating="post")

print(padded_seqs)