#! /usr/bin/env python

##########################################################################################
# setup.py
#
# This setup script defines the package metadata
#
# NOTE:
#
# Created: April 04, 2017
#   - Mitchell Allain
#   - allain.mitch@gmail.com
#
# Modified:
#   *
#
##########################################################################################

from setuptools import setup

setup(name='exc-analysis',
      version='0.1',
      description='Local package for quick access to analysis functions',
      # url='http://github.com/storborg/funniest',
      author='Mitchell Allain',
      author_email='allain.mitch@gmail.com',
      license='MIT',
      packages=['exc_analysis'],
      # install_requires=['matplotlib', 'pandas'],
      zip_safe=False)
