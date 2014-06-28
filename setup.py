#!/usr/bin/env python3

from setuptools import setup, find_packages


install_requires = [
		'numpy',
		]

setup(
	name='pathintmatmult',
	version='0.1',
	author='Dmitri Iouchtchenko',
	author_email='diouchtc@uwaterloo.ca',
	description='Path integrals via numerical matrix multiplication.',
	license='MIT',
	url='https://github.com/0/pathintmatmult',
	classifiers=[
		'Development Status :: 4 - Beta',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 3',
	],
	install_requires=install_requires,
	packages=find_packages(),
)
