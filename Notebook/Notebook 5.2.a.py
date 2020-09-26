#denfine the right version of tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io
print(tf.__version__)

#-------------------------------------------------------------------------------
#Import imdb dataset form tensorflow_datasets
###########################################
#                 Get Data                #
###########################################
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s,l in train_data:
  training_sentences.append(str(s.numpy()))
  training_labels.append(l.numpy())

for s,l in test_data:
  testing_sentences.append(str(s.numpy()))
  testing_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

#-------------------------------------------------------------------------------
#Tokenize, Sequences, Padded
###########################################
#             Data Pre-processing         #
###########################################
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
print(word_index)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
print(training_sentences[2])
print(padded[2])
print(padded.shape)
#Do similar tokenizer and sequences for testing datasets
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)
print(testing_sentences[2])
print(testing_padded[2])
print(testing_padded.shape)

#-------------------------------------------------------------------------------
#Reverse_word_index to decode it back to word
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
  return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[2]))
print(training_sentences[2])
#-------------------------------------------------------------------------------
#Build the NLP Model
###########################################
#              Data Processing            #
###########################################
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.GlobalAveragePooling1D()
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.summary()
#-------------------------------------------------------------------------------
#Train the Model
num_epochs = 10
model.fit(padded, training_labels_final,
          epochs=num_epochs,
          validation_data=(testing_padded, testing_labels_final))
#-------------------------------------------------------------------------------
#Print the shape
###########################################
#             Display Output              #
###########################################
e = model.layers[0]
weight = e.get_weight()[0]
print(weight.shape) #Shape: (vocab_size, embedding_dim)
#-------------------------------------------------------------------------------

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
#-------------------------------------------------------------------------------
try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')
#-----------------------------------------------------------------------------
sentence = "I really think this is amazing. honest"
print(sequence)
