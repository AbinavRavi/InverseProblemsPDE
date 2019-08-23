# InverseProblemsPDE

A Guided research project on Solving Inverse problems in PDE using Deep learning SS2019.

## Setup
Please follow the following procedure to prepare the environment.

### Prerequisites

1. Python Version 3 check with:
`
python3 --version
`
Python venv should be installed

2. virtualenv installed
`
virtualenv -h
`
If not installed install on linux with:
`
sudo apt update && sudo apt install virtualenv
`


### Getting started

1. Install Miniconda and get started by initiating a virtual environment for installing fenics
`
conda install -c conda-forge fenics 
`
(Instead of fenics you can give your own environment name)
2. Activate the environment now
`
conda activate fenics
`

3. Install Dependencies
`
pip install -r requirements.txt
