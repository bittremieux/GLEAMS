import os
import sys
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__),
                                              os.pardir, os.pardir)))

import datetime

from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

from gleams import config
from gleams.feature import feature
from gleams.metadata import metadata


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
        op_args={'massivekb_filename': config.massivekb_filename,
                 'metadata_filename': config.metadata_filename}
    )
    t_split_feat = PythonOperator(
        task_id='split_metadata_train_val_test',
        python_callable=metadata.split_metadata_train_val_test,
        op_args={'metadata_filename': config.metadata_filename,
                 'val_ratio': config.val_ratio,
                 'test_ratio': config.test_ratio,
                 'rel_tol': config.split_ratio_tolerance}
    )
    t_download = PythonOperator(
        task_id='download_massivekb_peaks',
        python_callable=metadata.download_massivekb_peaks,
        op_args={'massivekb_filename': config.massivekb_filename}
    )
    t_pairs_pos = [
        PythonOperator(
            task_id=f'generate_pairs_positive_{suffix}',
            python_callable=metadata.generate_pairs_positive,
            op_args={'metadata_filename': config.metadata_filename.replace(
                '.csv', f'_{suffix}.csv')})
        for suffix in ['train', 'val', 'test']
    ]
    t_pairs_neg = [
        PythonOperator(
            task_id=f'generate_pairs_negative_{suffix}',
            python_callable=metadata.generate_pairs_negative,
            op_args={'metadata_filename': config.metadata_filename.replace(
                    '.csv', f'_{suffix}.csv'),
                'mz_tolerance': config.pair_mz_tolerance})
        for suffix in ['train', 'val', 'test']
    ]
    t_enc_feat = PythonOperator(
        task_id='convert_peaks_to_features',
        python_callable=feature.convert_peaks_to_features,
        op_args={'metadata_filename': config.metadata_filename}
    )
    t_feat_combine = [
        PythonOperator(
            task_id=f'merge_features_{suffix}',
            python_callable=feature.merge_features,
            op_args={'metadata_filename': config.metadata_filename.replace(
                '.csv', f'_{suffix}.csv')})
        for suffix in ['train', 'val', 'test']
    ]
    t_train = DummyOperator(
        task_id='train_model'
    )

    t_metadata >> t_split_feat >> [*t_pairs_pos, *t_pairs_neg]
    t_download >> t_enc_feat >> t_feat_combine
    [*t_pairs_pos, *t_pairs_neg, *t_feat_combine] >> t_train
