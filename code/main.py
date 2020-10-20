from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import time

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu

from setup.setup import BASE_DIR
from setup.setup import train_loss, train_accuracy, train_step, train_dataset, ckpt_manager, translate
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


def main():
    # train()
    testing_lang = ['Guarani.txt']
    references = []
    ref_langs = ['NTLH.txt', 'acf.txt', 'NVI.txt', 'aa.txt']
    test_dir = os.path.join(BASE_DIR, 'testing/')
    test_corpus = open(os.path.join(test_dir, testing_lang[0]), encoding='utf-8').readlines()
    print(os.listdir(test_dir))
    for filename in os.listdir(test_dir):
        if filename in ref_langs:
            try:
                references.append(open(os.path.join(test_dir, filename), encoding='utf-8').readlines())
            except IsADirectoryError:
                pass



    print()
    total = len(test_corpus)
    print("Total test examples: ", total)
    scores = np.zeros(shape=(total, 1))
    for i, text in enumerate(test_corpus):
        refs = [ref[i].lower().split() for ref in references]
        translation = translate(preprocess_sentence(text)).lower().split()

        smoth = SmoothingFunction()
        scores[i] = sentence_bleu(references=refs, hypothesis=translation, smoothing_function=smoth.method4)
        if i % 10 == 0:
            sys.stdout.write('\r' + 'Loading: {:.2f}% Score Mean: {:.2f} STD: {:.2f}'.format(((i + 1) / total) * 100,
                                                                                             np.mean(scores[:i]),
                                                                                             np.std(scores)))
            sys.stdout.flush()

    print("Mean: {} STD: {}".format(np.mean(scores), np.std(scores)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
