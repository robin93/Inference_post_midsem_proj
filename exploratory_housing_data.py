#coding: utf-8
#Data description
"""
1 CRIM     per capita crime rate by town 
2 ZN       proportion of residential land zoned for lots over 25,000 sq.ft. 
3 INDUS    proportion of non-retail business acres per town. 
4 CHAS     Charles River dummy variable (1 if tract bounds river; 0 otherwise) 
5 NOX      nitric oxides concentration (parts per 10 million) RM       
6 RM       average number of rooms per dwelling 
7 AGE      proportion of owner-occupied units built prior to 1940 
8 DIS      weighted distances to five Boston employment centres 
9 RAD      index of accessibility to radial highways 
10 TAX      full-value property-tax rate per $10,000 
11 PTRATIO  pupil-teacher ratio by town 
12 B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
13 LSTAT    percentage lower status of the population 
14 MEDV     Median value of owner-occupied homes in $1000
"""

#import the data
import pandas as pd
import numpy as np
from scipy.stats import norm
import os,sys,xlrd
import matplotlib.pyplot as plt
cwd = os.getcwd()
raw_data = pd.read_excel(os.path.join(cwd,'Boston_Housing Data.xls'),'Data')  #xlrd module should be installed in python
#print raw_data.head(10)

#print raw_data.describe()

#print 'Are there any null values in the data set :',raw_data.isnull().values.any()


def scatter_plots(input_list):
	for element in input_list:
		if element ==1:
			raw_data.plot(x='CRIM',y='MEDV',kind = 'scatter')
			plt.xlabel('per capita crime rate by town');plt.ylabel('Median value of owner occupied homes');plt.title('Scatter plot of CRIM and MEDV'),plt.grid(True)
		elif element == 2:
			raw_data.plot(x='ZN',y='MEDV',kind = 'scatter')
			plt.xlabel('Proportion of residential land zoned for >25K Sq Ft');plt.ylabel('Median value of owner occupied homes');plt.title('Scatter plot of ZN and MEDV'),plt.grid(True)
		elif element == 3:
			raw_data.plot(x='INDUS',y='MEDV',kind = 'scatter')
			plt.xlabel('Proportion of non-retail business acres');plt.ylabel('Median value of owner occupied homes');plt.title('Scatter plot of INDUS and MEDV'),plt.grid(True)
		elif element == 4:
			raw_data.plot(x='CHAS',y='MEDV',kind = 'scatter')
			plt.xlabel('charles river in bounds or not');plt.ylabel('Median value of owner occupied homes');plt.title('Scatter plot of CHAS and MEDV'),plt.grid(True)
		elif element == 5:
			raw_data.plot(x='NOX',y='MEDV',kind = 'scatter')
			plt.xlabel('Nitric Oxides concentration');plt.ylabel('Median value of owner occupied homes');plt.title('Scatter plot of NOX and MEDV'),plt.grid(True)
		elif element == 6:
			raw_data.plot(x='RM',y='MEDV',kind = 'scatter')
			plt.xlabel('Avg # of rooms per dwelling');plt.ylabel('Median value of owner occupied homes');plt.title('Scatter plot of RM and MEDV'),plt.grid(True)
		elif element == 7:
			raw_data.plot(x='AGE',y='MEDV',kind = 'scatter')
			plt.xlabel('Proportion of homes before 1940');plt.ylabel('Median value of owner occupied homes');plt.title('Scatter plot of AGE and MEDV'),plt.grid(True)
		elif element == 8:
			raw_data.plot(x='DIS',y='MEDV',kind = 'scatter')
			plt.xlabel('Weighted distance to the boston centres');plt.ylabel('Median value of owner occupied homes');plt.title('Scatter plot of DIS and MEDV'),plt.grid(True)
		elif element == 9:
			raw_data.plot(x='RAD',y='MEDV',kind = 'scatter')
			plt.xlabel('Index of availaibility to radial highways');plt.ylabel('Median value of owner occupied homes');plt.title('Scatter plot of RAD and MEDV'),plt.grid(True)
		elif element == 10:
			raw_data.plot(x='TAX',y='MEDV',kind = 'scatter')
			plt.xlabel('Property tax rate');plt.ylabel('Median value of owner occupied homes');plt.title('Scatter plot of TAX and MEDV'),plt.grid(True)
		elif element == 11:
			raw_data.plot(x='PTRATIO',y='MEDV',kind = 'scatter')
			plt.xlabel('Pupil teacher ratio');plt.ylabel('Median value of owner occupied homes');plt.title('Scatter plot of PTRATIO and MEDV'),plt.grid(True)
		elif element == 12:
			raw_data.plot(x='B',y='MEDV',kind = 'scatter')
			plt.xlabel('Proportions of blacks in the town');plt.ylabel('Median value of owner occupied homes');plt.title('Scatter plot of B and MEDV'),plt.grid(True)
		elif element == 13:
			raw_data.plot(x='LSTAT',y='MEDV',kind = 'scatter')
			plt.xlabel('Percentage lower status of the population');plt.ylabel('Median value of owner occupied homes');plt.title('Scatter plot of LSTAT and MEDV'),plt.grid(True)
	plt.show()

def distribution_of_medv():
	plt.hist(raw_data.MEDV);plt.xlabel('Median prices of houses');plt.ylabel('Frequency');plt.title('MEDV histogram of all observations');plt.show()
	print "Mean of the Median housing prices",float(raw_data[['MEDV']].mean(axis=0))
	print "Variance of the Median housing prices",float(raw_data[['MEDV']].var(axis=0))
	print "Standard deviation of the Median Housing prices",float(raw_data[['MEDV']].std(axis=0))

def mean_plots(input_list):
	for element in input_list:
		if element == 13:
			x_values = [5,15,30]
			mean1,mean2,mean3 = (raw_data[(raw_data.LSTAT>0)&(raw_data.LSTAT<10)])['MEDV'].mean(axis =0),(raw_data[(raw_data.LSTAT>10)&(raw_data.LSTAT<20)])['MEDV'].mean(axis =0),(raw_data[(raw_data.LSTAT>20)])['MEDV'].mean(axis =0)
			std1,std2,std3 = (raw_data[(raw_data.LSTAT>0)&(raw_data.LSTAT<10)])['MEDV'].std(axis =0),(raw_data[(raw_data.LSTAT>10)&(raw_data.LSTAT<20)])['MEDV'].std(axis =0),(raw_data[(raw_data.LSTAT>20)])['MEDV'].std(axis =0)
			plt.errorbar(x_values,[mean1,mean2,mean3],[std1,std2,std3]);plt.xlim(0,40);plt.grid(True)
			plt.xlabel('LSTAT values');plt.ylabel('MEDV mean and standard deviation');plt.title('Mean plot of MEDV vs LSTAT values')
			plt.show()

def hypothesis_test(input_list):
	for element in input_list:
		if element == 13:
			pop_std_of_mean = (float(raw_data[['MEDV']].std(axis=0)))/np.sqrt(len(raw_data))
			z_value = (19-39)/pop_std_of_mean
			print 'Z_value',z_value
			print float(norm.cdf(z_value))
			




#############Execution################
#scatter_plots([13]) 
#distribution_of_medv()
mean_plots([13])
hypothesis_test([13])




#df.to_excel('path_to_file.xlsx', sheet_name='Sheet1')   #write the file to xls format