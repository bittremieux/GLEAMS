import logging
import os
import sys

import multiprocessing_logging


def init():
    # Configure logging.
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : '
        '{message}', style='{'))
    root.addHandler(handler)
    # Avoid logging conflicts during multiprocessing.
    multiprocessing_logging.install_mp_handler()
    # Limit annoying Tensorflow logging to only warnings and errors.
    # 1: No FILTER logging.
    # 2: No WARNING logging.
    # 3: No ERROR logging.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    # Disable dependency non-critical log messages.
    logging.getLogger('airflow').setLevel(logging.WARNING)
    logging.getLogger('faiss').setLevel(logging.WARNING)
    logging.getLogger('h5py').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
