import logging
import os
import sys
# Make sure all code is in the PATH.
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__),
                                              os.pardir, os.pardir)))
# Limit annoying Tensforflow logging to only warnings and errors.
# 1: No FILTER logging.
# 2: No WARNING logging.
# 3: No ERROR logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Initialize all random seeds before importing any packages.
from gleams import rndm
rndm.set_seeds()

import datetime

import multiprocessing_logging
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils import helpers
import tensorflow.compat.v1 as tf

from gleams import config
from gleams.feature import feature
from gleams.metadata import metadata
from gleams.nn import nn


# Fix logging hijacking by Tensorflow/abseil.
# FIXME: https://github.com/abseil/abseil-py/issues/99
# FIXME: https://github.com/tensorflow/tensorflow/issues/26691
try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    pass
# Disable Tensorflow v1 deprecation warnings.
tf.logging.set_verbosity(tf.logging.ERROR)


# Initialize logging.
logging.basicConfig(format='{asctime} [{levelname}/{processName}] '
                           '{module}.{funcName} : {message}',
                    style='{', level=logging.DEBUG, force=True)
logging.captureWarnings(True)
multiprocessing_logging.install_mp_handler()


default_args = {
    'owner': 'gleams',
    'depends_on_past': False,
    'start_date': datetime.datetime(2019, 1, 1),
    'email': ['wbittremieux@ucsd.edu'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': datetime.timedelta(minutes=5)
}

with DAG('gleams', default_args=default_args,
         schedule_interval=datetime.timedelta(weeks=1)) as dag:
    suffixes = ['train', 'val', 'test']

    feat_dir = os.path.join(os.environ['GLEAMS_HOME'], 'data', 'feature')

    t_metadata = PythonOperator(
        task_id='convert_massivekb_metadata',
        python_callable=metadata.convert_massivekb_metadata,
        op_kwargs={'massivekb_filename': config.massivekb_filename,
                   'metadata_filename': config.metadata_filename}
    )
    t_split_feat = PythonOperator(
        task_id='split_metadata_train_val_test',
        python_callable=metadata.split_metadata_train_val_test,
        op_kwargs={'metadata_filename': config.metadata_filename,
                   'val_ratio': config.val_ratio,
                   'test_ratio': config.test_ratio,
                   'rel_tol': config.split_ratio_tolerance}
    )
    t_download = PythonOperator(
        task_id='download_massivekb_peaks',
        python_callable=metadata.download_massivekb_peaks,
        op_kwargs={'massivekb_filename': config.massivekb_filename}
    )
    t_enc_feat = PythonOperator(
        task_id='convert_peaks_to_features',
        python_callable=feature.convert_peaks_to_features,
        op_kwargs={'metadata_filename': config.metadata_filename}
    )
    t_combine_feat = {
        suffix: PythonOperator(
            task_id=f'combine_features_{suffix}',
            python_callable=feature.combine_features,
            op_kwargs={'metadata_filename': config.metadata_filename.replace(
                           '.parquet', f'_{suffix}.parquet')})
        for suffix in suffixes
    }
    t_pairs_pos = {
        suffix: PythonOperator(
            task_id=f'generate_pairs_positive_{suffix}',
            python_callable=metadata.generate_pairs_positive,
            op_kwargs={
                'metadata_filename': os.path.join(
                    feat_dir,
                    f'feature_{config.massivekb_task_id}_{suffix}.parquet')})
        for suffix in suffixes
    }
    t_pairs_neg = {
        suffix: PythonOperator(
            task_id=f'generate_pairs_negative_{suffix}',
            python_callable=metadata.generate_pairs_negative,
            op_kwargs={
                'metadata_filename': os.path.join(
                    feat_dir,
                    f'feature_{config.massivekb_task_id}_{suffix}.parquet'),
                'mz_tolerance': config.pair_mz_tolerance})
        for suffix in suffixes
    }
    t_train = PythonOperator(
        task_id='train_nn',
        python_callable=nn.train_nn,
        op_kwargs={'filename_model': config.model_filename,
                   'filename_feat_train':
                       os.path.join(feat_dir,
                                    f'feature_{config.massivekb_task_id}_'
                                    f'train.npy'),
                   'filename_train_pairs_pos':
                       os.path.join(feat_dir,
                                    f'feature_{config.massivekb_task_id}_'
                                    f'train_pairs_pos.npy'),
                   'filename_train_pairs_neg':
                       os.path.join(feat_dir,
                                    f'feature_{config.massivekb_task_id}_'
                                    f'train_pairs_neg.npy'),
                   'filename_feat_val':
                       os.path.join(feat_dir,
                                    f'feature_{config.massivekb_task_id}_'
                                    f'val.npy'),
                   'filename_val_pairs_pos':
                       os.path.join(feat_dir,
                                    f'feature_{config.massivekb_task_id}_'
                                    f'val_pairs_pos.npy'),
                   'filename_val_pairs_neg':
                       os.path.join(feat_dir,
                                    f'feature_{config.massivekb_task_id}_'
                                    f'val_pairs_neg.npy')}
    )
    t_embed = PythonOperator(
        task_id='embed',
        python_callable=nn.embed,
        op_kwargs={'filename_model': config.model_filename}
    )

    t_metadata >> t_split_feat
    t_download >> t_enc_feat
    helpers.cross_downstream([t_split_feat, t_enc_feat],
                             t_combine_feat.values())
    for suffix in suffixes:
        t_combine_feat[suffix] >> [t_pairs_pos[suffix], t_pairs_neg[suffix]]
    [t_pairs_pos['train'], t_pairs_neg['train'],
     t_pairs_pos['val'], t_pairs_neg['val']] >> t_train
    t_train >> t_embed
