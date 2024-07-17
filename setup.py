import io
import os
from pathlib import Path

from setuptools import find_packages, setup

NAME= 'my_package'
DESCRIPTION = 'Loan Prediction Model'
URL = 'https//github.com/Amirgadami'
EMAIL = 'ah.ghadami75@gmail.com'
AUTHOR='Amirhossein Ghadami'
REQUIRES_PYTHON = '>=3.9.7'


pwd = os.path.abspath(os.path.dirname(__file__))
print(pwd)
def list_reqs(fname='requirements.txt'):
    with io.open(os.path.join(pwd,fname),encoding='utf-8')as f:
        print(f.read().splitlines())

try:
    with io.open(os.path.join(pwd,'README.md'),encoding='utf-8')as f:
        long_description= '\n' + f.read()

except FileNotFoundError:
    long_description = DESCRIPTION

ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = os.path.join(ROOT_DIR,NAME)
about = {}

with open(os.path.join(PACKAGE_DIR,'VERSION')) as f:
    _version = f.read().strip()
    about['__version__'] = _version



setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    package_data={'prediction_model': ['VERSION']},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)