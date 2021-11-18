from distutils.core import setup

setup(name='torchnde',
      version='0.0',
      packages=['torchnde'],
      license='Creative Commons Attribution-Noncommercial-Share Alike license',
      description='Neural Density Estimation with pyTorch',
      long_description='Neural Density Estimation with pyTorch',
      entry_points = {'console_scripts': ['torchnde=torchnde.command_line:main']},
      author='Pablo Lemos',
      author_email='p.lemos@sussex.ac.uk',
      url='https://github.com/Pablo-Lemos/torchNDE'
      )
