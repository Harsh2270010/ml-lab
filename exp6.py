import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# Load the dataset
heartDisease = pd.read_csv("heart.csv")

# Display sample instances and data types
print('Sample instances from the dataset:')
print(heartDisease.head())
print('\nAttributes and datatypes:')
print(heartDisease.dtypes)

# Define the Bayesian Network structure
model = BayesianNetwork([
    ('age', 'target'),
    ('sex', 'target'),
    ('cp', 'target'),
    ('restecg', 'target'),
    ('chol', 'target')
])

# Fit the model using Maximum Likelihood Estimators
print('\nLearning CPD using Maximum likelihood estimators...')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# Inference with the Bayesian Network
HeartDiseasetest_infer = VariableElimination(model)

# Query 1: Probability of Heart Disease given evidence for 'restecg'
print('\n1. Probability of Heart Disease given evidence= restecg:')
q1 = HeartDiseasetest_infer.query(variables=['target'], evidence={'restecg': 1})
print(q1)

# Query 2: Probability of Heart Disease given evidence for 'cp'
print('\n2. Probability of Heart Disease given evidence= cp:')
q2 = HeartDiseasetest_infer.query(variables=['target'], evidence={'cp': 2})
print(q2) 