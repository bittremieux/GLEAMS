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
    t_pairs_pos = [
        PythonOperator(
            task_id=f'generate_pairs_positive_{suffix}',
            python_callable=metadata.generate_pairs_positive,
            op_kwargs={'metadata_filename': config.metadata_filename.replace(
                '.csv', f'_{suffix}.csv')})
        for suffix in ['train', 'val', 'test']
    ]
    t_pairs_neg = [
        PythonOperator(
            task_id=f'generate_pairs_negative_{suffix}',
            python_callable=metadata.generate_pairs_negative,
            op_kwargs={'metadata_filename': config.metadata_filename.replace(
                           '.csv', f'_{suffix}.csv'),
                       'mz_tolerance': config.pair_mz_tolerance})
        for suffix in ['train', 'val', 'test']
    ]
    t_enc_feat = PythonOperator(
        task_id='convert_peaks_to_features',
        python_callable=feature.convert_peaks_to_features,
        op_kwargs={'metadata_filename': config.metadata_filename,
                   'feat_dir': config.feat_dir}
    )
    t_combine_feat = [
        PythonOperator(
            task_id='combine_features',
            python_callable=feature.combine_features,
            op_kwargs={'metadata_filename': config.metadata_filename.replace(
                           '.csv', f'_{suffix}.csv'),
                       'feat_dir': config.feat_dir})
        for suffix in ['train', 'val', 'test']
    ]
    t_train = PythonOperator(
        task_id='train_nn',
        python_callable=nn.train_nn,
        op_kwargs={'feat_dir': config.feat_dir,
                   'filename_model': config.model_filename,
                   'filename_metadata_train':
                       config.metadata_filename.replace(
                           '.csv', '_train.csv'),
                   'filename_train_pairs_pos':
                       config.metadata_filename.replace(
                           '.csv', '_train_pairs_pos.csv'),
                   'filename_train_pairs_neg':
                       config.metadata_filename.replace(
                           '.csv', '_train_pairs_neg.csv'),
                   'filename_metadata_val':
                       config.metadata_filename.replace(
                           '.csv', '_val.csv'),
                   'filename_val_pairs_pos':
                       config.metadata_filename.replace(
                           '.csv', '_val_pairs_pos.csv'),
                   'filename_val_pairs_neg':
                       config.metadata_filename.replace(
                           '.csv', '_val_pairs_neg.csv')}
    )

    t_metadata >> t_split_feat >> [*t_pairs_pos, *t_pairs_neg]
    t_download >> t_enc_feat
    helpers.cross_downstream([t_split_feat, t_enc_feat], t_combine_feat)
    [*t_pairs_pos, *t_pairs_neg, *t_combine_feat] >> t_train
