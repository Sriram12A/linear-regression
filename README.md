# 📈 Linear Regression Machine Learning Models

Welcome to the **Linear Regression Machine Learning Models** repository! This project serves as a comprehensive introduction and practical implementation of both Simple and Multiple Linear Regression algorithms using Python and popular data science libraries.

## 📖 Overview

Linear regression is one of the most fundamental statistical and machine learning techniques. It is used to predict a continuous dependent variable based on one or more independent variables. 

This repository explores two main variations of Linear Regression:
1. **Simple Linear Regression:** Predicting a target variable using a single feature.
2. **Multiple Linear Regression:** Predicting a target variable using two or more features.

## 🗂️ Project Structure

The codebase is organized into Jupyter Notebooks for interactive exploration and CSV files for datasets.

### 📓 Jupyter Notebooks
* **`linear regression with single variable.ipynb`**: Contains the code and mathematical intuition behind predicting a target variable using one independent variable. 
* **`linear regression with multiple variables.ipynb`**: Demonstrates how to handle multiple independent variables to make predictions. Also covers handling missing data and data preprocessing.

### 📊 Datasets included
This project uses several datasets to train and evaluate the models:

* **`canada_per_capita_income.csv`**: A time-series dataset tracking the per capita income of Canada over the years. (Used for single variable regression: *Year ➡️ Income*)
* **`hiring.csv`**: A dataset containing candidate experience, written test scores, and personal interview scores to predict the salary they should be offered. (Used for multiple variable regression: *Experience, Test Score, Interview Score ➡️ Salary*)
* **`info.csv` / `Book1.csv` / `11.csv` / `multi.csv` / `test set.csv`**: Additional datasets and test sets used for model evaluation and experiments.

## ⚙️ Prerequisites & Setup

To run the notebooks locally, you will need Python installed along with the following libraries:
- `pandas` (for data manipulation)
- `numpy` (for numerical computations)
- `scikit-learn` (for building the machine learning models)
- `matplotlib` (for data visualization)
- `jupyter` (to run the notebooks)

You can install all the required dependencies using pip:
```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

## 🚀 How to Use

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/Sriram12A/linear-regression.git
   ```
2. Navigate to the directory:
   ```bash
   cd linear-regression
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open either `linear regression with single variable.ipynb` or `linear regression with multiple variables.ipynb` in your browser and run the cells sequentially to see the models in action!
