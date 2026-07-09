# Auto-generated script for creating setup.py
file(WRITE "/home/crx/code/hccl_three_sequence_rs/hccl_0701_br_sequence/hccl_2039/build/src/es_hccl_build/python_package/setup.py" "from setuptools import setup, find_packages

setup(
    name='es_hccl',
    version='1.0.0',
    description='ES Generated API for es_hccl operators',
    author='Huawei Technologies Co., Ltd.',
    packages=find_packages(),
    python_requires='>=3.7',
    entry_points={
        'ge.es.plugins': [
            'hccl = es_hccl:get_module',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
")
message(STATUS "Created setup.py for package 'es_hccl'")
