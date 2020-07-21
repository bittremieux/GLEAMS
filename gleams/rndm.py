import os

# Limit annoying Tensorflow logging to only warnings and errors.
# 1: No FILTER logging.
# 2: No WARNING logging.
# 3: No ERROR logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# Fix logging hijacking by Tensorflow/abseil.
# FIXME: https://github.com/abseil/abseil-py/issues/99
# FIXME: https://github.com/tensorflow/tensorflow/issues/26691
try:
    import logging
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    pass

# https://github.com/NVIDIA/framework-determinism
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import random

import numpy as np
import tensorflow as tf


def set_seeds(my_seed=42):
    os.environ['PYTHONHASHSEED'] = str(my_seed)
    random.seed(my_seed)
    np.random.seed(my_seed)
    tf.random.set_seed(my_seed)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

