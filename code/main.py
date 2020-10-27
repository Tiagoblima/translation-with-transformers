from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os

from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu
from setup.setup import train, test_sentence
from .setup.params import BASE_DIR

EPOCHS = 100


def test():
    test_file = os.path.join(BASE_DIR, 'testing/test_corpus_gn-pt.json')
    test_corpus = list(json.load(open(test_file, 'r', encoding='utf8')).values())

    results = list(map(test_sentence, test_corpus))
    predictions = [res[0] for res in results]
    true_translations = [res[1] for res in results]
    smoth = SmoothingFunction()
    score = corpus_bleu(true_translations, predictions, smoothing_function=smoth.method4)

    print("\nCorpus score: ", score)


def main():
    train(EPOCHS)
    test()
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
