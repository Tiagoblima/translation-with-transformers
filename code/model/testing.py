from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import sys
import os
import json

from model.params import BASE_DIR, BATCH_SIZE
from model.training import translate


def test_sentence(args):
    sentence, references = args
    predicted = translate(sentence)

    return predicted.lower().split(), references


def split_batches(corpus, batch_size):
    previous = 0
    for i in range(len(corpus), batch_size):
        yield corpus[previous:i]
        previous = i


def test():
    test_file = os.path.join(BASE_DIR, 'testing/test_corpus_gn-pt.json')

    test_corpus = list(json.load(open(test_file, encoding='utf-8')).values())[:64]
    batches = split_batches(test_corpus, BATCH_SIZE)
    history = {}
    for i, batch in enumerate(batches):
        results = list(map(test_sentence, batch))
        predictions = [res[0] for res in results]
        true_translations = [res[1] for res in results]
        smoth = SmoothingFunction()
        score = corpus_bleu(true_translations, predictions, smoothing_function=smoth.method4)
        history[i] = {'inputs': batch,
                      'predictions': predictions}
        print("\nCorpus score: ", score)
        if i % 5 == 0:
            json.dump(history, open('scores_history.json', 'w'))