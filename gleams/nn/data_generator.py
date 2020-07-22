import logging
import math
from typing import List, Tuple

import numpy as np
import scipy.sparse as ss
from tensorflow.keras.utils import Sequence


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
            A SciPy sparse file containing the encoded spectrum features.
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
        self.features = ss.load_npz(filename_feat)

        pairs_pos = np.load(filename_pairs_pos, mmap_mode='r')
        pairs_neg = np.load(filename_pairs_neg, mmap_mode='r')
        num_pairs = min(len(pairs_pos), len(pairs_neg))
        if max_num_pairs is not None:
            num_pairs = min(num_pairs, max_num_pairs // 2)
        logger.info('Using %d positive and negative feature pairs each from '
                    'file %s', num_pairs, filename_feat)
        idx_pos = np.random.choice(pairs_pos.shape[0], num_pairs, False)
        self.pairs_pos = pairs_pos[idx_pos]
        idx_neg = np.random.choice(pairs_neg.shape[0], num_pairs, False)
        self.pairs_neg = pairs_neg[idx_neg]

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
        batch_pairs = np.vstack((batch_pairs_pos, batch_pairs_neg))

        batch_x1 = self.features[batch_pairs[:, 0]]
        batch_x2 = self.features[batch_pairs[:, 1]]
        batch_y = np.hstack((np.ones(len(batch_pairs_pos), np.uint8),
                             np.zeros(len(batch_pairs_neg), np.uint8)))

        return ([*_split_features_to_input(batch_x1, *self.feature_split),
                 *_split_features_to_input(batch_x2, *self.feature_split)],
                batch_y)

    def on_epoch_end(self):
        self.epoch_count += 1
        if self.shuffle and self.epoch_count % len(self) == 0:
            logger.debug('Shuffle the features because all pairs have been '
                         'processed after epoch %d', self.epoch_count)
            np.random.shuffle(self.pairs_pos)
            np.random.shuffle(self.pairs_neg)


class EncodingsSequence(Sequence):

    def __init__(self, encodings: ss.csr_matrix, batch_size: int,
                 feature_split: Tuple[int, int]):
        """
        Initialize the EncodingsSequence generator.

        Parameters
        ----------
        encodings : ss.csr_matrix
            Sparse SciPy array with encodings as rows.
        batch_size : int
            The (maximum) size of each batch. Batch sizes can sometimes be
            smaller than this maximum size in case of missing feature vectors.
        feature_split : Tuple[int, int]
            Indexes on which the feature vectors are split into individual
            inputs to the separate parts of the neural network (precursor
            features, fragment features, reference spectra features).
        """
        self.encodings = encodings
        self.batch_size = batch_size
        self.feature_split = feature_split

    def __len__(self) -> int:
        """
        Gives the total number of batches.

        Returns
        -------
        int
            The number of batches.
        """
        return int(math.ceil(len(self.encodings) / self.batch_size))

    def __getitem__(self, idx: int)\
            -> List[np.ndarray]:
        """
        Get the batch of encodings arrays with the given index.

        Parameters
        ----------
        idx : int
            Index of the requested batch.

        Returns
        -------
        List[np.ndarray]
            A batch of encodings consisting of three NumPy arrays for the three
            input elements of the neural network.
        """
        return list(_split_features_to_input(
            self.encodings[idx * self.batch_size:(idx + 1) * self.batch_size],
            *self.feature_split))


def _split_features_to_input(x: ss.csr_matrix, idx1: int, idx2: int)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert individual features to arrays corresponding to the three inputs for
    the neural network.

    Parameters
    ----------
    x : List[ss.csr_matrix]
        Sparse feature array with encodings as rows.
    idx1 : int
        First index to split the feature arrays.
    idx2 : int
        Second index to split the feature arrays.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        The features are split in three arrays according to the two split
        indexes.
    """
    x = x.toarray()
    return x[:, :idx1], x[:, idx1:idx2], x[:, idx2:]
