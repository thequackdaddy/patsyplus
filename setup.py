from distutils.core import setup

setup(name='patsyplus',
      version='0.0.1',
      description='patsy addons',
      author='Peter Quackenbush',
      author_email='',
      url='https://www.python.org/sigs/distutils-sig/',
      packages=['patsyplus'],
      install_requires=['patsy>=0.5.1',
                        'statsmodels>=0.10.2']
      )
