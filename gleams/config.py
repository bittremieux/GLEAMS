from gleams.embed import utils

# MS/MS spectrum preprocessing settings.

# Minimum number of peaks for an MS/MS spectrum to be considered.
min_peaks = 10
min_mz_range = 250.
remove_precursor_tolerance = 0.05
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

# Fragment encoding.
fragment_mz_min = utils.averagine_peak_separation_da * 50.5
fragment_mz_max = 2500.
bin_size = utils.averagine_peak_separation_da

# Reference spectra encoding.
ref_spectra_filename = utils.get_data_path('gleams_reference_spectra.mgf')
num_ref_spectra = 500
fragment_mz_tol = 0.05

# Theoretical spectrum simulation.
ms2pip_model = 'HCDch2'
ms2pip_batch_size = 100000


# Neural network settings.

# Training hyperparameters.
loss_label_certainty = 0.99
margin = 1
lr = 0.0002
batch_size = 1024
num_epochs = 60
