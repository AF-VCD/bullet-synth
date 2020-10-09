#!/usr/bin/env python
# coding: utf-8

#this is the python script version of the jupyter notebooks that I used to create the models. 

# It's a lot easier to iterate in a jupyter notebook than in a script.



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# stuff for this notebook to work in kaggle
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.





from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import time





physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)





path_to_file = './bullets.txt'
# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))

# The unique characters in the file
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))

# Take a look at the first 250 characters in text
print(text[:250])

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

# Show how the first 13 characters from the text are mapped to integers
print ('{} ---- MAPS TO ---- > {}'.format(repr(text[:13]), text_as_int[:13]))





# The maximum length sentence we want for a single input in characters
seq_length = 250

examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets. 
# char_dataset is one basically one long 1d array with every element in there.
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(10):
  print(idx2char[i.numpy()])





sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
  print(repr(''.join(idx2char[item.numpy()])))





def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)





for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))





for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))





# Batch size - the number of simultaneous samples to evaluate on each training step. 
#  So for a given model step, BATCH_SIZE number examples are run through the model at that step, and the results for those 
#  BATCH_SIZE examples are compared to their respective "correct answers," and the resulting averaged or weighted averaged delta
#  is used to adjust the model for the next training set. 
BATCH_SIZE = 128

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset





# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 1024

# Number of RNN units
rnn_units = 1024





def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                              return_sequences=True,
                              stateful=True,
                              recurrent_initializer='glorot_uniform'),
    tf.keras.layers.LSTM(rnn_units,
                              return_sequences=True,
                              stateful=True,
                              recurrent_initializer='glorot_uniform'),
    tf.keras.layers.LSTM(rnn_units,
                              return_sequences=True,
                              stateful=True,
                              recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model





model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)





for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")





model.summary()





sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
sampled_indices





print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))





def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())





model.compile(optimizer='adam', loss=loss)





# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)





EPOCHS=75
# for previous model, around it converged to .23 at around 75 epochs. went all the way to 90 and it only dropped to like .21 or so.





with tf.device('/GPU:0'):   # GPU:/1 uses my GTX 970, it's the opposite of what is listed in task manager. With two LSTM layers, 970 couldn't handle I think.
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])





tf.train.latest_checkpoint(checkpoint_dir)





model.save('bullets-lstm.h5')





# For the purposes of evaluating the model, setting the batch size to 1 so you can train one thing at a time.

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))





model.summary()





model.save('bullets-lstm_built.h5')





def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 0.5

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      #print(input_eval)
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      
      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))





print(generate_text(model, start_string=u"- "))





print(generate_text(model, start_string=u"- Led "))





print(generate_text(model, start_string=u"- Led 6 org test event; executed 250k runs--"))





import json
json.dumps(char2idx)





json.dumps(vocab)


# In[ ]:




