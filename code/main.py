from __future__ import absolute_import, division, print_function, unicode_literals

from model.testing import test
from model.training import train_model


def main():
    train_model()
    test()
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
