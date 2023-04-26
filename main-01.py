from tensorflow.keras.preprocessing.text import Tokenizer

train_sentences = [
    'It is a sunny day',
    'It is a cloudy day',
    'Will it rain today?'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(train_sentences)

print(f"Word index ---> {word_index}")
print(f"Sequences of words ---> {sequences}")

print(train_sentences[0])
print(sequences[0])

new_setences = [
    'Will it be raining today?',
    'It is a pleasant day.'
]

new_sequences = tokenizer.texts_to_sequences(new_setences)
print(new_setences)
print(new_sequences)

tokenizer = Tokenizer(num_words=100, oov_token="<oov>")
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
new_sequences = tokenizer.texts_to_sequences(new_setences)
print(word_index)
print(new_setences)
print(new_sequences)