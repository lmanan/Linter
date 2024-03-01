from linter.datasets.ctc import CTCDataset


def get_dataset(data_dir, crop_size):
    return CTCDataset(data_dir=data_dir, crop_size=crop_size)
