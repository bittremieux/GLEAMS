import logging
import math
from typing import List, Tuple

import numpy as np
from keras.utils import Sequence


logger = logging.getLogger('gleams')


class PairSequence(Sequence):

    def __init__(self, filename_feat: str,
                 filename_pairs_pos: str, filename_pairs_neg: str,
                 batch_size: int, feature_split: Tuple[int, int],
                 max_num_pairs: int = None, shuffle: bool = True):
        """
        Initialize the PairSequence generator.

        The number of pairs that will be used for training will be twice the
        minimum number of positive and negative pairs, to ensure a balanced
        dataset.

        Parameters
        ----------
        filename_feat : str
            A NumPy binary file containing the encoded spectrum features.
        filename_pairs_pos : str
            The file name of the positive pair indexes. Comma-separated file
            with feature/spectrum indexes corresponding to the metadata file.
        filename_pairs_neg : str
            The file name of the negative pair indexes. Comma-separated file
            with feature/spectrum indexes corresponding to the metadata file.
        batch_size : int
            The (maximum) size of each batch. Batch sizes can sometimes be
            smaller than this maximum size in case of missing feature vectors.
        feature_split : Tuple[int, int]
            Indexes on which the feature vectors are split into individual
            inputs to the separate parts of the neural network (precursor
            features, fragment features, reference spectra features).
        max_num_pairs : int
            Maximum number of pairs to include.
        shuffle : bool
            Whether to shuffle the order of the batches at the beginning of
            each epoch.
        """
        self.features = np.load(filename_feat)

        pairs_pos = np.loadtxt(filename_pairs_pos, np.uint32, delimiter=',')
        np.random.shuffle(pairs_pos)
        pairs_neg = np.loadtxt(filename_pairs_neg, np.uint32, delimiter=',')
        np.random.shuffle(pairs_neg)
        num_pairs = min(len(pairs_pos), len(pairs_neg))
        if max_num_pairs is not None:
            num_pairs = min(num_pairs, max_num_pairs // 2)
        logger.info('Using %d feature pairs from file %s', num_pairs,
                    filename_feat)
        self.pairs_pos = pairs_pos[:num_pairs]
        self.pairs_neg = pairs_neg[:num_pairs]

        self.batch_size = batch_size
        self.feature_split = feature_split
        self.shuffle = shuffle
        self.epoch_count = 0

    def __len__(self) -> int:
        """
        Gives the total number of batches.

        Returns
        -------
        int
            The number of batches.
        """
        return int(math.ceil(2 * len(self.pairs_pos) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Get the batch with the given index.

        Parameters
        ----------
        idx : int
            Index of the requested batch.

        Returns
        -------
        Tuple[List[np.ndarray], np.ndarray]
            A tuple of features and class labels. The features consist of three
            NumPy arrays for the three input elements of the neural network.
            The class labels are 1 for positive pairs and 0 for negative pairs.
        """
        batch_pairs_pos = self.pairs_pos[idx * self.batch_size // 2:
                                         (idx + 1) * self.batch_size // 2]
        batch_pairs_neg = self.pairs_neg[idx * self.batch_size // 2:
                                         (idx + 1) * self.batch_size // 2]
        batch_pairs_idx = np.vstack((batch_pairs_pos, batch_pairs_neg))

        batch_x1 = self.features[batch_pairs_idx[:, 0]]
        batch_x2 = self.features[batch_pairs_idx[:, 1]]
        batch_y = np.hstack((np.ones(len(batch_pairs_pos), np.uint8),
                             np.zeros(len(batch_pairs_neg), np.uint8)))

        return (_features_to_arrays(batch_x1, batch_x2, *self.feature_split),
                batch_y)

    def on_epoch_end(self):
        self.epoch_count += 1
        if self.shuffle and self.epoch_count % len(self) == 0:
            logger.debug('Shuffle the features because all pairs have been '
                         'processed after epoch %d', self.epoch_count)
            np.random.shuffle(self.pairs_pos)
            np.random.shuffle(self.pairs_neg)


def _features_to_arrays(x1: List[np.ndarray], x2: List[np.ndarray],
                        idx1: int, idx2: int) -> List[np.ndarray]:
    """
    Convert individual features to arrays corresponding to the three inputs for
    the neural network.

    Parameters
    ----------
    x1 : List[np.ndarray]
        List of left feature arrays.
    x2 : List[np.ndarray]
        List of right feature arrays.
    idx1 : int
        First index to split the feature arrays.
    idx2 : int
        Second index to split the feature arrays.

    Returns
    -------
    List[np.ndarray]
        A list of six arrays: both the left and right features are split in
        three arrays according to the two split indexes.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return [x1[:, :idx1], x1[:, idx1:idx2], x1[:, idx2:],
            x2[:, :idx1], x2[:, idx1:idx2], x2[:, idx2:]]
