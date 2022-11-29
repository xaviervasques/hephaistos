#!/usr/bin/python3
# ml_pipeline_function.py
# Author: Xavier Vasques (Last update: 29/11/2022)

# Copyright 2022, Xavier Vasques. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from setuptools import setup

setup(
    name='hephaistos',
    version='0.1.0',
    description='hephAIstos is a machine learning package that can run with CPU, GPU and QPU',
    url='https://github.com/xaviervasques/hephaistos.git', 
    author='Xavier Vasques',
    author_email='xaviervasques@lrenc.fr',
    license='Apache License version 2',
    packages=['hephaistos'],
    install_requires=['pandas',
                      'numpy',
                      'scipy',
                      'sklearn',
                      'category_encoders',
                      'tensorflow',
                      'matplotlib',
                      'qiskit',
                      'qiskit_machine_learning',
                      'hashlib'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: Apache License version 2',
        'Operating System :: Linux, MacOS, Windows',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        
    ],
)
