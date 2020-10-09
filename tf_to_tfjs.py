# for the tfjs conversion, I used a completely separate conda environment that just had tensorflow and tensorflowjs installed (no GPU versions).
#   I forget the reason why I did this, I don't think it worked otherwise.

# python tf_to_tfjs.py -i  .\bullets-lstm.h5 -o  ./tfjs

import tensorflowjs as tfjs
import tensorflow as tf

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

import sys, getopt

inputfile = ''
outfolder = ''

helpstring = 'tf_to_tfjs.py -i <input-model> -o <output-folder>'

try:
  opts, args = getopt.getopt(sys.argv[1:],"hi:o:")
except getopt.GetoptError:
  print(helpstring) 
  sys.exit(2)

for opt,arg in opts:
  if opt == '-h':
    print(helpstring)
  elif opt == '-i':
    inputfile = arg
  elif opt == "-o":
    outfolder = arg
  else:
    print(helpstring)
    sys.exit(2)

import os
if not os.path.exists(outfolder):
  os.makedirs(outfolder)

model = tf.keras.models.load_model(inputfile, custom_objects={"loss":loss})
tfjs.converters.save_keras_model(model, outfolder)

