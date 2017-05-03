from setuptools import setup

setup(name='tf_spectral',
      version='0.1',
      description='Spectrogram tools using tensorflow',
      url='http://github.com/zekearneodo/tf_spectral',
      author='Zeke Arneodo',
      author_email='earneodo@ucsd.edu',
      license='GNU3',
      packages=['tf_spectral'],
      install_requires=['numpy', 'tensorflow', 'semver'],
      zip_safe=False)