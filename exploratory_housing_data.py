#coding: utf-8
#Data description
"""
CRIM     per capita crime rate by town 
ZN       proportion of residential land zoned for lots over 25,000 sq.ft. 
INDUS    proportion of non-retail business acres per town. 
CHAS     Charles River dummy variable (1 if tract bounds river; 0 otherwise) 
NOX      nitric oxides concentration (parts per 10 million) 
RM       average number of rooms per dwelling 
AGE      proportion of owner-occupied units built prior to 1940 
DIS      weighted distances to five Boston employment centres 
RAD      index of accessibility to radial highways 
TAX      full-value property-tax rate per $10,000 
PTRATIO  pupil-teacher ratio by town 
B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
LSTAT    percentage lower status of the population 
MEDV     Median value of owner-occupied homes in $1000
"""

#import the data
import pandas as pd
import os,sys,xlrd
import matplotlib.pyplot as plt
cwd = os.getcwd()
raw_data = pd.read_excel(os.path.join(cwd,'Boston_Housing Data.xls'),'Data')  #xlrd module should be installed in python
#print raw_data.head(10)

print raw_data.describe()

print 'Are there any null values in the data set :',raw_data.isnull().values.any()


def plots(input_list):
	for element in input_list:
		if element ==1:
			raw_data.plot(x='CRIM',y='MEDV',kind = 'scatter')
			plt.xlabel('per capita crime rate by town')
			plt.ylabel('Median value of owner occupied homes')
			plt.title('Scatter plot of CRIM and MEDV')
			plt.show()


plot_input = [1]

plots(plot_input) 




#df.to_excel('path_to_file.xlsx', sheet_name='Sheet1')   #write the file to xls format