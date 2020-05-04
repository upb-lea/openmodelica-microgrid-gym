#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup_requirements = ['pytest-runner']

test_requirements = ['pytest>=3', 'tables>=3.4.1', 'safeopt>=0.16', 'GPy>=1.9.9']

setup(
    author="LEA - Uni Paderborn",
    author_email='upblea@mail.upb.de',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="OpenModelica Microgrid Gym",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='openmodelica_microgrid_gym',
    name='openmodelica_microgrid_gym',
    packages=find_packages(include=['openmodelica_microgrid_gym', 'openmodelica_microgrid_gym.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require={'examples': ['safeopt>=0.16', 'GPy>=1.9.9']},
    url='https://github.com/upb-lea/openmodelica-microgrid-gym',
    project_urls={
        "Documentation": "https://upb-lea.github.io/openmodelica-microgrid-gym/",
        "Source Code": "https://github.com/upb-lea/openmodelica-microgrid-gym",
    },
    version='0.1.2',
    zip_safe=False,
)
