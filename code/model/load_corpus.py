import os

from model.params import BASE_DIR, MAX_LENGTH, BUFFER_SIZE, BATCH_SIZE, META_INFO_PATH
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from .util import preprocess_sentence

train_examples = open(BASE_DIR + '/training/Guarani-Portugues.txt', encoding='utf8').readlines()
val_examples = open(BASE_DIR + '/validation/Guarani-Portugues.txt', encoding='utf8').readlines()

print(train_examples[:1])

train_examples_inp = [preprocess_sentence(ex.strip().split('\t')[0]) for ex in train_examples]
train_examples_tgt = [preprocess_sentence(ex.strip().split('\t')[1]) for ex in train_examples]

val_examples_inp = [preprocess_sentence(ex.strip().split('\t')[0]) for ex in val_examples]
val_examples_tgt = [preprocess_sentence(ex.strip().split('\t')[1]) for ex in val_examples]
print(train_examples_inp[:1], len(train_examples_inp))
print(train_examples_tgt[:1], len(train_examples_inp))

train_examples = tf.data.Dataset.from_tensor_slices((np.array(train_examples_inp),
                                                     np.array(train_examples_tgt)))

val_examples = tf.data.Dataset.from_tensor_slices((val_examples_inp, val_examples_tgt))

vocab_fname = os.path.join(META_INFO_PATH, 'GUARANI/GUARANI')
# tokenizer_inp = tfds.deprecated.text.SubwordTextEncoder.load_from_file(vocab_fname)
vocab_fname = os.path.join(META_INFO_PATH, 'PORTUGUES/PORTUGUES')
# tokenizer_tgt = tfds.deprecated.text.SubwordTextEncoder.load_from_file(vocab_fname)
tokenizer_inp = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)

tokenizer_tgt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2 ** 13)


def encode(lang1, lang2):
    lang1 = [tokenizer_tgt.vocab_size] + tokenizer_tgt.encode(
        lang1.numpy()) + [tokenizer_tgt.vocab_size + 1]

    lang2 = [tokenizer_inp.vocab_size] + tokenizer_inp.encode(
        lang2.numpy()) + [tokenizer_inp.vocab_size + 1]

    return lang1, lang2


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en


train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)

pt_batch, en_batch = next(iter(val_dataset))

input_vocab_size = tokenizer_tgt.vocab_size + 2
target_vocab_size = tokenizer_inp.vocab_size + 2
print("input_vocab_size: ", input_vocab_size)
print("target_vocab_size: ", target_vocab_size)
