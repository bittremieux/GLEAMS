import logging
import math
import operator
import os
import sys
import tempfile
from typing import Iterator, List

import numpy as np
import pandas as pd
import tqdm
from pyteomics.mass import fast_mass
from spectrum_utils.spectrum import MsmsSpectrum

from gleams import config
from gleams.embed import spectrum, utils

# We need to explicitly add the MS2PIP directory to the path because we're not
# using MS2PIP as a stand-alone script.
ms2pip_dir = os.path.realpath(os.path.join(__file__, '../../../ms2pip_c'))
if ms2pip_dir not in sys.path:
    sys.path.append(ms2pip_dir)

from ms2pip_c import ms2pipC


logger = logging.getLogger('gleams')


class SpectrumSimulator:
    """
    Simulate theoretical spectra using MS2PIP.

    Current MS2PIP release: v20190624.
    """

    def __init__(self, model: str = 'HCDch2'):
        """
        Instantiate a SpectrumSimulator.

        Parameters
        ----------
        model : str (default: 'HCDch2')
            The MS2PIP model used to predict theoretical spectra. Allowed
            values are 'HCD', 'CID', 'TTOF5600', 'TMT', 'iTRAQ',
            'iTRAQphospho', 'HCDch2'.
        """
        models = ('HCD', 'CID', 'TTOF5600', 'TMT', 'iTRAQ', 'iTRAQphospho',
                  'HCDch2')
        if model not in models:
            logger.error('Incorrect MS2PIP model, should be one of: %s '
                         '(default: %s)', ', '.join(models),
                         config.ms2pip_model)
            raise ValueError(f'Incorrect MS2PIP model, should be one of: '
                             f'{", ".join(models)} (default: '
                             f'{config.ms2pip_model})')
        else:
            self.model = model

    def simulate(self, peptides: List[str], charges: List[int],
                 decoy: bool = False) -> Iterator[MsmsSpectrum]:
        """
        Generate simulated spectra for the given peptides with the given
        charge,

        Parameters
        ----------
        peptides : List[str]
            The peptide sequences for which spectra are simulated. The peptide
            sequences can have modifications specified as mass differences.
        charges : List[int]
            The precursor charge to simulate the spectra.
        decoy : bool, optional
            Convert the peptides to decoy peptides by shuffling them before
            simulating their spectra.

        Returns
        -------
        Iterator[MsmsSpectrum]
            An iterator over the simulated spectra.
        """
        logger.info('Predict spectra for %d peptides using MS2PIP',
                    len(peptides))
        for batch_start in range(0, len(peptides), config.ms2pip_batch_size):
            batch_stop = min(batch_start + config.ms2pip_batch_size,
                             len(peptides))
            peptides_batch = peptides[batch_start:batch_stop]
            charges_batch = charges[batch_start:batch_stop]

            peptides_no_mod, peptides_mods_md = [], []
            with tempfile.NamedTemporaryFile('w+', delete=False) as f_config,\
                    tempfile.NamedTemporaryFile('w+', delete=False) as f_pep,\
                    tempfile.NamedTemporaryFile('w+', delete=False) as f_out:
                # Create suitable config and PEPREC files.
                f_config.write(f'model={self.model}\n')
                f_config.write(f'frag_error={config.fragment_mz_tol}\n')
                logger.debug('MS2PIP configuration file written to %s',
                             f_config.name)

                peprec = [['spec_id', 'peptide', 'modifications', 'charge']]
                # Process the peptide sequences and collect all possible
                # modifications.
                mods, mod_keys = {}, set()
                for i, (peptide, charge) in enumerate(zip(peptides_batch,
                                                          charges_batch)):
                    peptide_no_mod = utils.normalize_peptide(peptide)
                    # Shuffle the peptide to generate a decoy if necessary.
                    aa_order = np.arange(len(peptide_no_mod) + 1)
                    if decoy:
                        # Keep the N-terminal and C-terminal amino acids in
                        # place.
                        np.random.shuffle(aa_order[2:-1])
                        peptide_no_mod = ''.join(
                            np.asarray(list(peptide_no_mod))[aa_order[1:] - 1])
                    # Specify the positions of the modifications.
                    pep_mods, peptide_mods_md = [], 0.
                    for md, aa, pos in utils.get_peptide_modifications(
                            peptide):
                        if (md, aa) not in mods:
                            name = utils.generate_random_string(3)
                            while name in mod_keys:
                                name = utils.generate_random_string(3)
                            mods[(md, aa)] = name
                            mod_keys.add(name)
                        pep_mods.append((aa_order[pos],
                                         f'{aa_order[pos]}|{mods[(md, aa)]}'))
                        peptide_mods_md += md
                    if decoy:
                        pep_mods = sorted(pep_mods, key=operator.itemgetter(0))
                    pep_mods = [pm[1] for pm in pep_mods]
                    # Store the peptide in the PEPREC format.
                    peprec.append([str(i), peptide_no_mod,
                                   ('|'.join(pep_mods)
                                    if len(pep_mods) > 0 else '-'),
                                   str(charge)])
                    peptides_no_mod.append(peptide_no_mod)
                    peptides_mods_md.append(peptide_mods_md)

                for (md, aa), name in mods.items():
                    f_config.write(f'ptm={name},{md},opt,{aa}\n')

                f_pep.write('\n'.join([' '.join(row) for row in peprec]))
                logger.debug('MS2PIP PEPREC file written to %s', f_pep.name)

                # Run MS2PIP.
                f_pep.seek(0)
                f_config.seek(0)
                logger.debug('Write MS2PIP spectrum predictions to '
                             '%s_predictions.csv', f_out.name)
                ms2pipC.run(f_pep.name, config_file=f_config.name,
                            num_cpu=os.cpu_count(),
                            output_filename=f_out.name)

                # Collect the MS2PIP predictions.
                predictions = pd.read_csv(f'{f_out.name}_predictions.csv',
                                          index_col='spec_id')
                for spec_id, peptide, charge, md in zip(
                        sorted(predictions.index.unique()), peptides_no_mod,
                        charges, peptides_mods_md):
                    precursor_mz = fast_mass(peptide, charge=charge) + md
                    intensity = np.asarray(
                        2 ** predictions.loc[spec_id, 'prediction'] - 0.001)
                    intensity_mask = intensity > 0
                    intensity = spectrum._norm_intensity(
                        intensity[intensity_mask])
                    mz = (np.asarray(predictions.loc[spec_id, 'mz'])
                          [intensity_mask])
                    spec = MsmsSpectrum(f'ms2pip_{peptide}', precursor_mz,
                                        charge, mz, intensity, peptide=peptide)
                    # Do some minimal spectrum preprocessing similar to the
                    # experimental spectra.
                    # Explicitly preprocess like this to avoid throwing out
                    # "low-quality" simulated spectra.
                    spec = spec.set_mz_range(config.fragment_mz_min,
                                             config.fragment_mz_max)
                    scaling = ('root' if config.scaling == 'sqrt' else
                               config.scaling)
                    if scaling is not None:
                        spec = spec.scale_intensity(
                            scaling, max_rank=config.max_peaks_used)
                    yield spec
