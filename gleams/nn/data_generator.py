import itertools
import math
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
from keras.utils import Sequence


class PairSequence(Sequence):

    def __init__(self, filename_metadata: str, filename_feat: str,
                 filename_pairs_pos: str, filename_pairs_neg: str,
                 batch_size: int, feature_split: Tuple[int, int],
                 max_num_pairs: int = None):
        """
        Initialize the PairSequence generator.

        The number of pairs that will be used for training will be twice the
        minimum number of positive and negative pairs, to ensure a balanced
        dataset.

        Parameters
        ----------
        filename_metadata : str
            The file name of the metadata file. This file needs to have ordered
            rows corresponding to the pair indexes (see below) and have the
            columns 'dataset', 'filename', and 'scan'.
        filename_feat : str
            The file name of the HDF5 feature file. Feature vectors for
            individual scans should be stored under a dataset/filename/scan
            key.
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
        """
        metadata = pd.read_csv(filename_metadata,
                               usecols=['dataset', 'filename', 'scan'],
                               dtype=str)
        self.spec_keys = (metadata['dataset'] + '/' + metadata['filename']
                          + '/' + metadata['scan'])

        self.f_feat = h5py.File(filename_feat, 'r')

        pairs_pos = np.loadtxt(filename_pairs_pos, np.uint32, delimiter=',')
        np.random.shuffle(pairs_pos)
        pairs_neg = np.loadtxt(filename_pairs_neg, np.uint32, delimiter=',')
        np.random.shuffle(pairs_neg)
        num_pairs = min(len(pairs_pos), len(pairs_neg))
        if max_num_pairs is not None:
            num_pairs = min(num_pairs, max_num_pairs // 2)
        self.pairs_pos = pairs_pos[:num_pairs]
        self.pairs_neg = pairs_neg[:num_pairs]

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
        return int(math.ceil(len(self.pairs_pos) / self.batch_size)) * 2

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

        batch_x1, batch_x2, batch_y = [], [], []
        for key1, key2, y in zip(
                self.spec_keys.iloc[batch_pairs_idx[:, 0]],
                self.spec_keys.iloc[batch_pairs_idx[:, 1]],
                itertools.chain(np.ones(len(batch_pairs_pos), np.uint8),
                                np.zeros(len(batch_pairs_neg), np.uint8))):
            if key1 in self.f_feat and key2 in self.f_feat:
                batch_x1.append(self.f_feat[key1][:])
                batch_x2.append(self.f_feat[key2][:])
                batch_y.append(y)

        return (_features_to_arrays(batch_x1, batch_x2, *self.feature_split),
                np.asarray(batch_y))


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
