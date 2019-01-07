from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()


print(find_packages())

setup(
        name='scanpy_utils',
        version='0.0.1',
        description='complementary functions for scanpy',
        long_description=readme(),
        packages=find_packages(),
        install_requires=['numpy', 'pandas', 'scanpy'],
        author='Chichau Miau',
        author_email='zmiao@ebi.ac.uk',
        license='MIT'
    )
