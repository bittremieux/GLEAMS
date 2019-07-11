import logging
import os
import sys
import tempfile
from typing import Iterator, List

import numpy as np
import pandas as pd
from pyteomics.mass import fast_mass
from spectrum_utils.spectrum import MsmsSpectrum

from gleams import config
from gleams.embed import utils

# We need to explicitly add the MS2Pip directory to the path because we're not
# using MS2Pip as a stand-alone script.
ms2pip_dir = os.path.realpath(os.path.join(__file__, '../../../ms2pip_c'))
if ms2pip_dir not in sys.path:
    sys.path.append(ms2pip_dir)

from ms2pip_c import ms2pipC


logger = logging.getLogger('gleams')


class PrintSuppressor:

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout

    def write(self, x):
        pass


class SpectrumSimulator:
    """
    Simulate theoretical spectra using MS2Pip.
    """

    def __init__(self):
        """
        Instantiate a SpectrumSimulator.
        """
        pass

    def simulate(self, peptides: List[str], charges: List[int])\
            -> Iterator[MsmsSpectrum]:
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

        Returns
        -------

        """
        # TODO: Batch the spectrum predictions.
        logger.info('Predict spectra for %d peptides using MS2Pip',
                    len(peptides))
        peptides_no_mod, peptides_mods_md = [], []
        with tempfile.NamedTemporaryFile('w+', delete=False) as f_config,\
                tempfile.NamedTemporaryFile('w+', delete=False) as f_pep,\
                tempfile.NamedTemporaryFile('w+', delete=False) as f_out:
            # Create suitable config and PEPREC files.
            f_config.write('model=HCDch2\n')
            f_config.write(f'frag_error={config.fragment_mz_tol}\n')
            logger.debug('MS2Pip configuration file written to %s',
                         f_config.name)

            peprec = [['spec_id', 'peptide', 'modifications', 'charge']]
            # Process the peptide sequences and collect all possible
            # modifications.
            mods, mod_keys = {}, set()
            for i, (peptide, charge) in enumerate(zip(peptides, charges)):
                pep_mods, peptide_mods_md = [], 0.
                for md, aa, pos in utils.get_peptide_modifications(peptide):
                    if (md, aa) not in mods:
                        name = utils.generate_random_string(3)
                        while name in mod_keys:
                            name = utils.generate_random_string(3)
                        mods[(md, aa)] = name
                        mod_keys.add(name)
                    pep_mods.append(f'{pos}|{mods[(md, aa)]}')
                    peptide_mods_md += md
                peptide_no_mod = utils.normalize_peptide(peptide)
                peprec.append([str(i), peptide_no_mod,
                               ('|'.join(pep_mods)
                                if len(pep_mods) > 0 else '-'),
                               str(charge)])
                peptides_no_mod.append(peptide_no_mod)
                peptides_mods_md.append(peptide_mods_md)

            for (md, aa), name in mods.items():
                f_config.write(f'ptm={name},{md},opt,{aa}\n')

            f_pep.write('\n'.join([' '.join(row) for row in peprec]))
            logger.debug('MS2Pip PEPREC file written to %s', f_pep.name)

            # Run MS2Pip
            f_pep.seek(0)
            f_config.seek(0)
            logger.debug('Write MS2Pip spectrum predictions to '
                         '%s_predictions.csv', f_out.name)
            with PrintSuppressor():
                ms2pipC.run(f_pep.name, config_file=f_config.name,
                            num_cpu=os.cpu_count(), output_filename=f_out.name)

            # Collect the MS2Pip predictions.
            predictions = pd.read_csv(f'{f_out.name}_predictions.csv',
                                      index_col='spec_id')
            for spec_id, peptide, charge, md in zip(
                    sorted(predictions.index.unique()), peptides_no_mod,
                    charges, peptides_mods_md):
                precursor_mz = fast_mass(peptide, charge=charge) + md
                intensity = np.asarray(
                    2 ** predictions.loc[spec_id, 'prediction'] - 0.001)
                intensity_mask = intensity > 0
                intensity = intensity[intensity_mask]
                mz = np.asarray(predictions.loc[spec_id, 'mz'])[intensity_mask]
                yield MsmsSpectrum(f'ms2pip_{peptide}', precursor_mz, charge,
                                   mz, intensity, peptide=peptide)
