# Predict Customer Churn with Clean Code

The first project for the [ML DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) by Udacity.

Comprehensive Guide to Grouping and Aggregating with Pandas
https://pbpython.com/groupby-agg.html

https://www.datasciencemadesimple.com/group-by-mean-in-pandas-dataframe-python-2/

https://medium.com/illumination/introduction-to-mlops-f877ccf10db1

https://medium.com/@m_mcclarty/tech-book-talk-clean-code-in-python-aa2c92c6564f

https://towardsdatascience.com/python-clean-code-6-best-practices-to-make-your-python-functions-more-readable-7ea4c6171d60

## Description

This project is part of Unit 2: Clean Code Principles. The problem is to predict credit card customers that are most likely to churn using clean code best practices.

The move from a Data Scientist to a Machine Learning Engineer requires a move to coding best practices. In this project, we are tasked with moving code from a notebook that completes the data science process, but doesn't lend itself easy to reproduce, production-level code, to two scripts:

1. The first script `churn_library.py` is a python library containing functions needed to complete the same data science process.

2. The second script `churn_script_logging_and_tests.py` contains tests and logging that test the functions of your library and log any errors that occur.  

The original python notebook `churn_notebook.ipynb` contains the code to be refactored.

The new code was formatted using [autopep8](https://pypi.org/project/autopep8/), and both scripts provided [pylint](https://pypi.org/project/pylint/) scores exceeding **8.6**.

## Prerequisites

Python and Jupyter Notebook are required.
Also a Linux environment may be needed within windows through WSL.

## Dependencies
- joblib==0.11
- matplotlib==2.1.0
- numpy==1.12.1
- pandas==0.23.3
- scikit-learn==0.22
- seaborn==0.8.1
- shap==0.36.0

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies from the ```requirements.txt```

```bash
pip install -r requirements.txt
```

## Usage

The entire process can be checked in the jupyter notebook **churn_notebook.ipynb**

The data is stored in ```./data/```

The config file is stored in ```./config.yaml```

The main script to run using the following command.
```bash
python churn_library.py
```
which will generate
- EDA plots in the directory ```./images/eda/```
- Model metrics plots in the directory ```./images/results/```
- Saved model pickle files in the directory ```./models/```
- A log file ```./log/churn_library.log```

The tests script can be used with the following command which will generate a log file ```./log/tests_churn_library.log```
```bash
python churn_script_logging_and_tests.py
```

## File Structure
```
.
│   churn_library.py    a Python script to all functions
│   churn_script_logging_and_tests.py
│                       a Python script to test each of the functions and provide any errors to a file stored in the logs folder
│   churn_notebook.ipynb
│                       a Notebook to train and test model
│   Guide.ipynb         a Notebook to guide
│   config.json         a data file that contains names of files for configuring ML Python scripts     
│   LICENSE
│   README.md           *this file
│   requirements.txt    a text file containing current versions of all the modules the scripts use
│   
├───data                contains some data for practice
│       bank_data.csv
│       
├───models              contain ML models that are created as practice
│  
├───images              contain your images
│       eda             contain your eda images
│       results         contain your results images
│       
└───logs                contains a file to store errors

```

## Running Files
How do you run your files? What should happen when you run your files?

Install dependencies and libraries
`pip install -r requirements.txt`

This module holds all the tests of the functions
created in the `churn_library.py` file. To run it
use `python churn_script_logging_and_tests.py`

All python libraries used in this repository can be `pip` installed.  All files were created and tested using Python **3.x**.  


If you have the same folder structure as this repository, as well as the data available from [Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers?select=BankChurners.csv), then you can run the commands below to retrieve all results.

This following will test each of the functions and provide any errors to a file stored in the `logs` folder.
```
ipython churn_script_logging_and_tests_solution.py
```
This following will contain all the functions and refactored code associated with the original notebook.
```
ipython churn_library_solution.py
```

You can also check the pylint score, as well as perform the auto-formatting using the following commands:

```
pylint churn_library_solution.py
pylint churn_script_logging_and_tests_solution.py
```

The files here were formated using:
```
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests_solution.py
autopep8 --in-place --aggressive --aggressive churn_library_solution.py
```
