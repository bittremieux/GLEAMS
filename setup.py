import setuptools


setuptools.setup(
    name='gleams',
    version='0.1.0',
    author='Damon May',
    author_email='damonmay@uw.edu',
    description='GLEAMS is a Learned Embedding for Mass Spectra',
    license='Apache 2.0',
    packages=['gleams'],
    install_requires=[
        'keras',
        'numpy',
        'pyteomics',
        'spectrum_utils',
        'tqdm']
)
