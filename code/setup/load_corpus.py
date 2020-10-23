import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from .params import *


train_examples = open(os.path.join(BASE_DIR, 'training/Guarani-Portugues.txt'), encoding='utf8').readlines()
val_examples = open(os.path.join(BASE_DIR, 'validation/Guarani-Portugues.txt'), encoding='utf8').readlines()

train_examples_pt = [ex.strip().split('\t')[0] for ex in train_examples]
train_examples_gn = [ex.strip().split('\t')[1] for ex in train_examples]

val_examples_pt = [ex.strip().split('\t')[0] for ex in val_examples]
val_examples_gn = [ex.strip().split('\t')[1] for ex in val_examples]

train_examples = tf.data.Dataset.from_tensor_slices((np.array(train_examples_pt),
                                                     np.array(train_examples_gn)))

val_examples = tf.data.Dataset.from_tensor_slices((val_examples_pt, val_examples_gn))

tokenizer_inp = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

tokenizer_targ = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)


def encode(lang1, lang2):
    lang1 = [tokenizer_targ.vocab_size] + tokenizer_targ.encode(
        lang1.numpy()) + [tokenizer_targ.vocab_size + 1]

    lang2 = [tokenizer_inp.vocab_size] + tokenizer_inp.encode(
        lang2.numpy()) + [tokenizer_inp.vocab_size + 1]

    return lang1, lang2


def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)

pt_batch, en_batch = next(iter(val_dataset))

input_vocab_size = tokenizer_targ.vocab_size + 2
target_vocab_size = tokenizer_inp.vocab_size + 2