"""
neuroimage pipeline customize for low resolution of neonates
"""

from setuptools import setup, find_packages

setup(
    name='low resolution neonate MRI image deep learning',
    version="1.0",
    author='Qian Qin',
    description=__doc__,
    packages=find_packages(),
    install_requires=["nipype", "dipy", "h5py", "fire"],
    include_package_data=True,
    scripts=["neo_lowres"],
    classifiers=[
        'Environment::Console',
        'Operating System:: POSIX',
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: NeuroImage"
        ],
    keywords='neonatal low resolution preprocess pipeline',
    zip_safe=False
)

