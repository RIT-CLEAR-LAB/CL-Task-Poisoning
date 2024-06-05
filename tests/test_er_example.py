from utils.main import main, parse_args
import pytest
import sys


@pytest.mark.parametrize('dataset', ['seq-cifar10', ])  # 'seq-mnist', 'seq-cifar100', 'seq-tinyimg', 'rot-mnist', 'perm-mnist', 'mnist-360'
def test_er(dataset):
    sys.argv = ['mammoth',
                '--model',
                'er',
                '--dataset',
                dataset,
                '--buffer_size',
                '10',
                '--lr',
                '1e-4',
                '--n_epochs',
                '1',
                '--debug_mode',
                '1']
    a = parse_args()

    main(a)
