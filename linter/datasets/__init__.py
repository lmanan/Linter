from linter.datasets.ctc import CTCDataset


def get_dataset(data_dir, patch_size):
    return CTCDataset(data_dir, patch_size)
