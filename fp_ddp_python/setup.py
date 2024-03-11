from setuptools import find_packages, setup

setup(
    name='fp_ddp_python',
    packages=find_packages(include=['fp_ddp_python']),
    version='0.1.0',
    description='Python Library for FP-DDP',
    author='David Kiessling',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)