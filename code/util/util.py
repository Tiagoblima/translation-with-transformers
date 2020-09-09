import io
import re
import numpy as np
import tensorflow as tf
import unicodedata
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')

    return tensor, lang_tokenizer


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping
    # -punctuation
    w = re.sub(r"([?.!,Â¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,Â¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples, start=0):
    lines = io.open(path, encoding='utf8').read().strip().split('\n')

    if num_examples is None:
        num_examples = len(lines)

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[start:start + num_examples]]

    return zip(*word_pairs)


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def max_length(tensor):
    return max(len(t) for t in tensor)


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def bleu_accuracy(original, sentences):
    translation = []
    unknow_words = []
    for sentence in sentences:
        unknow_words.append([i for i in sentence.split(' ') if i not in inp_lang.word_index.keys()])
        result, sentence, attention_plot = evaluate(sentence, True)
        result = result.split()
        result.pop()
        translation.append(result)

    references = []

    for ref in original:
        references.append(ref.split())

    scores = []

    smoth = SmoothingFunction()
    for hyp, ref in zip(translation, references):
        try:
            score = sentence_bleu([ref], hypothesis=hyp, smoothing_function=smoth.method4)

            scores.append(score)
        except ZeroDivisionError:
            pass

    return np.mean(scores), np.sqrt(np.var(scores)), list(itertools.chain.from_iterable(unknow_words))
