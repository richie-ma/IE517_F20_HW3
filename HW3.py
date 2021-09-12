# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## this is code is based on the bond data

## load the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns

data = pd.read_csv("C:/Users/ruchuan2/Box/IE 517 Machine Learning in FIN Lab/HW3/HY_Universe_corporate bond.csv", header='infer')

## overall view of this data
data.head

## there are 2721 rows and 37 columns

## calculate statistics
## numeric variables
## examples

coupon_mean = np.mean(data['Coupon'])
print(coupon_mean)

iss_amount_mean = np.mean(data['Issued Amount'])
print(iss_amount_mean)

coupon_var = np.var(data['Coupon'])
print(coupon_var)

## covariance matrix bettwen coupon and iss_amount
cou_iss_corr = np.corrcoef(data['Coupon'],data['Issued Amount'])
print(cou_iss_corr)

## calculate the pearson correlation coefficient

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    
    corr_mat=np.corrcoef(x,y)

    # Return entry [0,1]
    return corr_mat[0,1]

pearson_r = pearson_r(data['Coupon'],data['Issued Amount'])
print(pearson_r)

#print summary of data frame
summary = data.describe()
print(summary)

## visualization
## histogram
## numeric variables
    
sns.set()
_ = plt.hist(data['Maturity At Issue months'])
_ = plt.xlabel('Maturity At Issue months')
_ = plt.ylabel('count')
plt.show()

sns.set()
_ = plt.hist(data['Issued Amount'])
_ = plt.xlabel('Issued Amount')
_ = plt.ylabel('count')
plt.show()

## bee swarm plot

_ = sns.swarmplot(x='bond_type', y='LiquidityScore', data=data)

_ = plt.xlabel('Bond Type')
_ = plt.ylabel('LiquidityScore')

plt.show()

## ECDFs
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y
score = data['LiquidityScore']
bond_type = data['bond_type']
bond1 = score[bond_type==1]
bond2 = score[bond_type==2]
bond3 = score[bond_type==3]
bond4 = score[bond_type==4]
bond5 = score[bond_type==5]

x_bond1, y_bond1 = ecdf(bond1)
x_bond2, y_bond2 = ecdf(bond2)
x_bond3, y_bond3 = ecdf(bond3)
x_bond4, y_bond4 = ecdf(bond4)
x_bond5, y_bond5 = ecdf(bond5)

# Plot all ECDFs on the same plot

_ = plt.plot(x_bond1,y_bond1,marker='.',linestyle='none')
_ = plt.plot(x_bond2,y_bond2,marker='.',linestyle='none')
_ = plt.plot(x_bond3,y_bond3,marker='.',linestyle='none')
_ = plt.plot(x_bond4,y_bond4,marker='.',linestyle='none')
_ = plt.plot(x_bond5,y_bond5,marker='.',linestyle='none')

# Annotate the plot
plt.legend(('bond1', 'bond2', 'bond3', 'bond4', 'bond5'), loc='lower right')
_ = plt.xlabel('Bond Type')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()

##scatter plot

_ = plt.scatter(data['n_trades'], data['LiquidityScore'])
_ = plt.xlabel("Number of Trades")
_ = plt.ylabel("Liquidity Score")
plt.show()

_ = plt.scatter(data['volume_trades'], data['LiquidityScore'])
_ = plt.xlabel("Trading Volume")
_ = plt.ylabel("Liquidity Score")
plt.show()

## corr heat map
corMat = data.corr()
_ = plt.pcolor(corMat)
plt.show()

## box plot

array = data.iloc[:,[9,13,15]].values
_ = plt.boxplot(array)
_ = plt.xlabel("Attribute Index")
_ = plt.ylabel("Quantile Range")
plt. show()

## ending
print("My name is Richie Ma")
print("My NetID is: ruchuan2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")