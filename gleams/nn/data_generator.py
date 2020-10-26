import logging
import math
import os
from typing import List, Tuple

import numpy as np
import scipy.sparse as ss
from tensorflow.keras.utils import Sequence


logger = logging.getLogger('gleams')


class PairSequence(Sequence):

    def __init__(self, filename_feat: str,
                 filenames_pairs_pos: List[str],
                 filenames_pairs_neg: List[str],
                 batch_size: int, feature_split: Tuple[int, int],
                 max_num_pairs: int = None, shuffle: bool = True):
        """
        Initialize the PairSequence generator.

        The number of pairs that will be used for training will be twice the
        minimum number of positive and negative pairs for the longest list of
        pairs, ensuring that a balanced dataset is used.

        Parameters
        ----------
        filename_feat : str
            A SciPy sparse file containing the encoded spectrum features.
        filenames_pairs_pos : Iterator[str]
            List of file names of the positive pair indexes. The pairs files
            should be comma-separated files with feature/spectrum indexes
            corresponding to the metadata file.
        filenames_pairs_neg : Iterator[str]
            List of file names of the negative pair indexes. The pairs files
            should be comma-separated file with feature/spectrum indexes
            corresponding to the metadata file.
        batch_size : int
            The (maximum) size of each batch. Batch sizes can sometimes be
            smaller than this maximum size in case of missing feature vectors.
            This should ideally be a multiple of the number of pairs lists and
            2 (positive and negative), otherwise the actual batch size might be
            slightly smaller than specified.
        feature_split : Tuple[int, int]
            Indexes on which the feature vectors are split into individual
            inputs to the separate parts of the neural network (precursor
            features, fragment features, reference spectra features).
        max_num_pairs : int
            Maximum number of pairs to include per combination of positive and
            negative file names.
        shuffle : bool
            Whether to shuffle the order of the batches at the beginning of
            each epoch.
        """
        self.features = ss.load_npz(filename_feat)

        if (len(filenames_pairs_pos) == 0 or
                len(filenames_pairs_pos) != len(filenames_pairs_neg)):
            raise ValueError('A positive number of matching pairs filenames '
                             'is required')

        if max_num_pairs is not None:
            max_num_pairs = max_num_pairs // len(filenames_pairs_pos) // 2

        self.pairs_pos, self.pairs_neg = [], []
        for filename_pairs_pos, filename_pairs_neg in zip(filenames_pairs_pos,
                                                          filenames_pairs_neg):
            pairs_pos = np.load(filename_pairs_pos, mmap_mode='r')
            pairs_neg = np.load(filename_pairs_neg, mmap_mode='r')
            num_pairs = min(len(pairs_pos), len(pairs_neg))
            if max_num_pairs is not None:
                num_pairs = min(num_pairs, max_num_pairs)
            logger.info('Using %d positive and negative feature pairs each '
                        'from pairs files %s and %s', num_pairs,
                        os.path.basename(filename_pairs_pos),
                        os.path.basename(filename_pairs_neg))
            idx_pos = np.random.choice(pairs_pos.shape[0], num_pairs, False)
            self.pairs_pos.append(pairs_pos[idx_pos])
            idx_neg = np.random.choice(pairs_neg.shape[0], num_pairs, False)
            self.pairs_neg.append(pairs_neg[idx_neg])

        self.batch_size_per_pairs = batch_size // len(self.pairs_pos) // 2
        self.feature_split = feature_split
        self.shuffle = shuffle
        self.epoch_count = 0

    def __len__(self) -> int:
        """
        Gives the total number of batches.

        The number of batches is determined based on the longest pairs list.
        Batches from the pairs lists that are shorter than the longest pairs
        list will be created using random sampling with repetition, until the
        longest pairs list runs out.

        Returns
        -------
        int
            The number of batches.
        """
        return math.ceil(max(map(lambda pairs: len(pairs), self.pairs_pos))
                         / self.batch_size_per_pairs)

    def __getitem__(self, idx: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Get the batch with the given index.

        Note: This function should only be called to retrieve consecutive
        batches in a loop! Otherwise unspecified randomization of the batches
        can occur.

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
        batch_pairs, batch_y = [], []
        for pairs_pos, pairs_neg in zip(self.pairs_pos, self.pairs_neg):
            batch_start_i = idx * self.batch_size_per_pairs
            if batch_start_i > len(pairs_pos):
                batch_start_i = batch_start_i % len(pairs_pos)
                # Shuffle if this is the first time we start a new loop through
                # these pairs.
                if self.shuffle and batch_start_i == 0:
                    np.random.shuffle(pairs_pos)
                    np.random.shuffle(pairs_neg)
            batch_stop_i = batch_start_i + self.batch_size_per_pairs
            batch_pairs_pos = pairs_pos[batch_start_i:batch_stop_i]
            batch_pairs_neg = pairs_neg[batch_start_i:batch_stop_i]
            batch_pairs.append(batch_pairs_pos)
            batch_pairs.append(batch_pairs_neg)
            batch_y.append(np.ones(len(batch_pairs_pos), np.uint8))
            batch_y.append(np.zeros(len(batch_pairs_neg), np.uint8))

        batch_pairs, batch_y = np.vstack(batch_pairs), np.hstack(batch_y)
        batch_x1 = self.features[batch_pairs[:, 0]]
        batch_x2 = self.features[batch_pairs[:, 1]]

        return ([*_split_features_to_input(batch_x1, *self.feature_split),
                 *_split_features_to_input(batch_x2, *self.feature_split)],
                batch_y)

    def on_epoch_end(self):
        self.epoch_count += 1
        if self.shuffle and self.epoch_count % len(self) == 0:
            logger.debug('Shuffle all pairs because all pairs of the longest '
                         'pairs list have been processed after epoch %d',
                         self.epoch_count)
            for pairs_pos, pairs_neg in zip(self.pairs_pos, self.pairs_neg):
                np.random.shuffle(pairs_pos)
                np.random.shuffle(pairs_neg)


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
        return int(math.ceil(self.encodings.shape[0] / self.batch_size))

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
