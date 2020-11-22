from setuptools import setup, find_packages

setup(name='hoqunm',
      version='0.1',
      description='Hospital queueing network model',
      author='Lovis Schmidt',
      author_email='lovis.schmidt@web.de',
      license='proprietary',
      packages=find_packages(),
      entry_points={
            "console_scripts":
                  ["hoqunm-cli=hoqunm.cli:cli"]
      })
