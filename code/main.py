from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
import sys
import time

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

from setup.load_corpus import train_dataset
from setup.setup import BASE_DIR, test_sentence
from setup.setup import train_loss, train_accuracy, train_step, ckpt_manager, translate
from util.util import preprocess_sentence

EPOCHS = 100


def train():
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> guarani, tar -> portugues
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


def test():
    train()

    test_file = os.path.join(BASE_DIR, 'testing/test_corpus_gn-pt.json')
    test_corpus = list(json.load(open(test_file, encoding='utf-8')).values())

    results = list(map(test_sentence, test_corpus))
    predictions = [res[0] for res in results]
    true_translations = [res[1] for res in results]
    smoth = SmoothingFunction()
    score = corpus_bleu(true_translations, predictions, smoothing_function=smoth.method4)

    print("\nCorpus score: ", score)


def main():
    train()
    test()
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
