import os

from gleams.feature import spectrum


# MassIVE-KB metadata processing and pair generation.
massivekb_task_id = '82c0124b'  # Version 2018-06-15.
massivekb_filename = os.path.join(
    os.environ['GLEAMS_HOME'], 'data', 'metadata',
    f'LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-{massivekb_task_id}-'
    f'candidate_library_spectra-main.tsv')
metadata_filename = os.path.join(
    os.environ['GLEAMS_HOME'], 'data', 'metadata',
    f'metadata_{massivekb_task_id}.parquet')
model_filename = os.path.join(
    os.environ['GLEAMS_HOME'], 'data', 'model',
    f'gleams_{massivekb_task_id}.hdf5')
val_ratio = 0.1
test_ratio = 0.1
split_ratio_tolerance = 0.01
pair_mz_tolerance = 10  # ppm

# MS/MS spectrum preprocessing settings.

# Minimum number of peaks for an MS/MS spectrum to be considered.
min_peaks = 10
min_mz_range = 250.
remove_precursor_tolerance = 0.05  # Da
min_intensity = 0.01
max_peaks_used = 150
scaling = 'sqrt'


# Encoder settings.

# Precursor encoding.
precursor_mz_min = 50.5
precursor_mz_max = 2500.
num_bits_precursor_mz = 27
precursor_mass_min = 400.
precursor_mass_max = 6000.
num_bits_precursor_mass = 27
precursor_charge_max = 7
num_precursor_features = (num_bits_precursor_mz + num_bits_precursor_mass +
                          precursor_charge_max)

# Fragment encoding.
averagine_peak_separation_da = 1.0005079
fragment_mz_min = averagine_peak_separation_da * 50.5
fragment_mz_max = 2500.
bin_size = averagine_peak_separation_da
num_fragment_features = spectrum.get_num_bins(fragment_mz_min, fragment_mz_max,
                                              bin_size)

# Reference spectra encoding.
ref_spectra_filename = os.path.join(
    os.environ['GLEAMS_HOME'], 'src', 'data', 'gleams_reference_spectra.mgf')
num_ref_spectra = 500
fragment_mz_tol = 0.05  # Da


# Neural network settings.
embedding_size = 32

# Training hyperparameters.
loss_label_certainty = 0.99
margin = 1
lr = 0.0002
batch_size = 2048
steps_per_epoch = 5000  # 2048 * 5000 = 10,240,000 samples per epoch
num_epochs = 1000
max_num_pairs_train = None
max_num_pairs_val = 500000


# Clustering settings.

# Pairwise distances.
precursor_tol_mass = 10
precursor_tol_mode = 'ppm'
mz_interval = 1
batch_size_add = 2**14
batch_size_dist = 2**11
num_neighbors = 50
num_neighbors_ann = 1024
num_probe = 1024

# DBSCAN clustering.
# TODO: Figure out good hyperparameters.
eps = 0.5
min_samples = 5
