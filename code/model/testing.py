from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import sys
import os
import json
import numpy as np
from model.params import BASE_DIR, BATCH_SIZE
from model.training import translate

smoth = SmoothingFunction()


def test_sentence(args):
    sentence, references = args
    predicted = translate(sentence)

    return sentence_bleu(references, predicted, smoothing_function=smoth.method4)


def split_batches(corpus, batch_size):
    previous = 0
    batches = []
    for i in range(batch_size, len(corpus), batch_size):
        batches.append(corpus[previous:i])
        previous = i
    return batches


def test():
    test_file = os.path.join(BASE_DIR, 'testing/test_corpus_gn-pt.json')

    test_corpus = list(json.load(open(test_file, encoding='utf-8')).values())
    batches = split_batches(test_corpus, BATCH_SIZE)
    history = {}
    for i, batch in enumerate(batches):
        results = list(map(test_sentence, batch))

        history[i] = {'inputs': batch,
                      'scores': results}
        print("\nCorpus score: ", np.mean(results), np.std(results))
        if i % 5 == 0:
            json.dump(history, open('scores_history.json', 'w'))
