# Algorithms
Algorithms for CREW

## Setup

Create a conda environment with python version 3.10.11.
```bash
mkdir ../UnityLogs
conda create -n crew python==3.10.11
conda install -c conda-forge pyaudio
```


This project uses [Poetry](https://python-poetry.org/) to manage dependencies. If you don't have poetry already installed:

'''
curl -sSL https://install.python-poetry.org | python3 -
vim ~/.bashrc
'''

add export PATH="/home/USERNAME/.local/bin:$PATH" to a new line
then 
'''
source ~/.bashrc
'''


Then run the command to install all of the dependencies:
```bash
poetry install
```
<!-- 
Then to activate the shell run the following command:
```bash
poetry shell
``` -->

Then you can run all of the commands in this repository.

If you want to contribute to this repository, you can install the pre-commit hooks for some nice features:
```bash
poetry run pre-commit install
```

export PATH="$/Users/michael/.local/bin.poetry/bin:$PATH"