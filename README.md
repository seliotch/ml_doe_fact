# ml_doe_fact
Script for running a four factor, two level factorial on machine learning hyperparameters

## 1. Description
This script was designed to implement a factorial design of experiments on the hyperparameters of a machine learning model. It was also designed to be used in a jupyter notebook and not meant to be run all at once.

## 2. Necessary libraries

The following libraries are needed to run this script:

[pandas](https://github.com/pandas-dev/pandas)

[tensorflow](https://github.com/tensorflow/tensorflow)

## 3. How it works

### experiment()
1. When `experiment()` is implemented on a recipe list, values from factor levels are pulled and the model is run. The model should have its own self-stopping mechanism.
2. The output returns a list of three values which is flattened and then added to a master list. 

### csv_out()
1. Converts a list of lists to a dataframe by iterating over each item in the master list.

## 4. Usage

This is designed to be run in the same folder as your model defined as "model.py". The script is also intended to be used in an environment like jupyter notebook and not intended to be run all at once, although it can be. Verify that your model has similar or the same names defined for hyperparameters including: learning_rate_1, learning_rate_2, dropout_rate_1, dropout_rate_2. These names can be altered to test other hyperparameters as needed. 

### Variables

Select up to four variables to test with a full or partial factorial following these guidelines: [NIST engineering handbook](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3347.htm). If you choose to use less variables, then set high and low values to the same number. The four factors are labeled: a, b, c d, at two levels each: high, low. 

### Script Use

If you are implementing this in jupyter notebook. First import the defined libraries, define your factor levels, and run the portion that includes the functions. Leave the rest to run as needed. Within the function experiment you may need to change hyperparameter name defintions. Define your recipe as needed for full or partial factorials.


