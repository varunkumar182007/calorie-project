import numpy as np
import matplotlib.pyplot as plt
##from sklearn.linear_model import LinearRegression
##from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd

df = pd.read_excel("sample_superstore_excel_dataset.xlsx",sheet_name="Orders")
print(df.head())
print(df.info())
print(df.describe())
