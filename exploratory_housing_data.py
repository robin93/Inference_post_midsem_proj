#coding: utf-8
#import the data
import pandas as pd
import numpy as np
from scipy.stats import norm,t
import os,sys,xlrd
import matplotlib.pyplot as plt
cwd = os.getcwd()
raw_data = pd.read_excel(os.path.join(cwd,'Boston_Housing Data.xls'),'Data')  #xlrd module should be installed in python
#print raw_data.head(10)

#print raw_data.describe()

#print 'Are there any null values in the data set :',raw_data.isnull().values.any()

def confidence_interval(standard_deviation,observations,confidence):
	confidence_fraction = (1 - (100-float(confidence))/200)
	if observations > 30:
		total_length_of_confidence_interval = (standard_deviation*2*norm.ppf(confidence_fraction)/np.sqrt(observations))
	else:
		total_length_of_confidence_interval = (standard_deviation*2*t.ppf(confidence_fraction,observations)/np.sqrt(observations))
	return total_length_of_confidence_interval

def histograms(input_list):
	for choice in input_list:
		if choice == 9:
			raw_data['RAD'].plot(kind = 'hist')
		plt.show()

def scatter_plots(input_list):
	for element in input_list:
		if element ==1:
			raw_data.plot(x='CRIM',y='MEDV',kind = 'scatter')
			plt.xlabel('per capita crime rate by town');plt.ylabel('Median value of owner occupied homes');plt.title('Scatter plot of CRIM and MEDV'),plt.xlim(0,5),plt.grid(True)
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
	for choice in input_list:
		if choice == 13:
			x_values = [5,15,30]
			mean1,mean2,mean3 = (raw_data[(raw_data.LSTAT>0)&(raw_data.LSTAT<10)])['MEDV'].mean(axis =0),(raw_data[(raw_data.LSTAT>10)&(raw_data.LSTAT<20)])['MEDV'].mean(axis =0),(raw_data[(raw_data.LSTAT>20)])['MEDV'].mean(axis =0)
			std1,std2,std3 = (raw_data[(raw_data.LSTAT>0)&(raw_data.LSTAT<10)])['MEDV'].std(axis =0),(raw_data[(raw_data.LSTAT>10)&(raw_data.LSTAT<20)])['MEDV'].std(axis =0),(raw_data[(raw_data.LSTAT>20)])['MEDV'].std(axis =0)
			n1,n2,n3 = len(raw_data[(raw_data.LSTAT>0)&(raw_data.LSTAT<10)]),len(raw_data[(raw_data.LSTAT>10)&(raw_data.LSTAT<20)]),len(raw_data[(raw_data.LSTAT>20)])
			cint1,cint2,cint3 = confidence_interval(std1,n1,95),confidence_interval(std2,n2,95),confidence_interval(std3,n3,95)
			plt.errorbar(x_values,[mean1,mean2,mean3],[cint1,cint2,cint3]);plt.xlim(0,40);plt.grid(True)
			plt.xlabel('LSTAT values');plt.ylabel('Confidence interval of Mean House prices');plt.title('Confidence interval of mean values of house prices by LSTAT sub ranges')
		elif choice ==10:
			x_values = [300,700]
			sub_data1,sub_data2 = raw_data[(raw_data.TAX>100)&(raw_data.TAX<500)],raw_data[(raw_data.TAX>600)&(raw_data.TAX<800)]
			mean1,mean2 = sub_data1['MEDV'].mean(axis =0),sub_data2['MEDV'].mean(axis =0)
			std1,std2 = sub_data1['MEDV'].std(axis =0),sub_data2['MEDV'].std(axis =0)
			n1,n2 = len(sub_data1),len(sub_data2)
			cint1,cint2 = confidence_interval(std1,n1,95),confidence_interval(std2,n2,95)
			plt.errorbar(x_values,[mean1,mean2],[cint1,cint2]);plt.xlim(100,800);plt.grid(True)
			plt.xlabel('Full property tax rate');plt.ylabel('Confidence interval of Mean House prices');plt.title('Confidence interval of mean values of house prices by TAX sub ranges')
		elif choice ==9:
			x_values = [1,2,3,4,5,6,7,8,24]
			sub_data1 = raw_data[(raw_data.RAD==1)]['MEDV']
			sub_data2 = raw_data[(raw_data.RAD==2)]['MEDV']
			sub_data3 = raw_data[(raw_data.RAD==3)]['MEDV']
			sub_data4 = raw_data[(raw_data.RAD==4)]['MEDV']
			sub_data5 = raw_data[(raw_data.RAD==5)]['MEDV']
			sub_data6 = raw_data[(raw_data.RAD==6)]['MEDV']
			sub_data7 = raw_data[(raw_data.RAD==7)]['MEDV']
			sub_data8 = raw_data[(raw_data.RAD==8)]['MEDV']
			sub_data9 = raw_data[(raw_data.RAD==24)]['MEDV']
			means = [sub_data1.mean(axis =0),sub_data2.mean(axis =0),sub_data3.mean(axis =0),sub_data4.mean(axis =0),sub_data5.mean(axis =0),sub_data6.mean(axis =0),sub_data7.mean(axis =0),sub_data8.mean(axis =0),sub_data9.mean(axis =0)]
			std1,std2,std3,std4,std5,std6,std7,std8,std9 = sub_data1.std(axis =0),sub_data2.std(axis =0),sub_data3.std(axis =0),sub_data4.std(axis =0),sub_data5.std(axis =0),sub_data6.std(axis =0),sub_data7.std(axis =0),sub_data8.std(axis =0),sub_data9.std(axis =0)
			n1,n2,n3,n4,n5,n6,n7,n8,n9 = len(sub_data1),len(sub_data2),len(sub_data3),len(sub_data4),len(sub_data5),len(sub_data6),len(sub_data7),len(sub_data8),len(sub_data9)
			conf = 95
			cint1,cint2,cint3,cint4,cint5,cint6,cint7,cint8,cint9 = confidence_interval(std1,n1,conf),confidence_interval(std2,n2,conf),confidence_interval(std3,n3,conf),confidence_interval(std4,n4,conf),confidence_interval(std5,n5,conf),confidence_interval(std6,n6,conf),confidence_interval(std7,n7,conf),confidence_interval(std8,n8,conf),confidence_interval(std9,n9,conf)
			plt.errorbar(x_values,means,[cint1,cint2,cint3,cint4,cint5,cint6,cint7,cint8,cint9]);plt.xlim(0,30);plt.grid(True)
			plt.xlabel('Index of radial distance from highways');plt.ylabel('Confidence interval of Mean House prices');plt.title('Confidence interval of mean values of house prices by RAD')		
		elif choice ==7:
			x_values = [20,60,90]
			sub_data1,sub_data2,sub_data3 = (raw_data[(raw_data.AGE>0)&(raw_data.AGE<40)])['MEDV'],(raw_data[(raw_data.AGE>40)&(raw_data.AGE<80)])['MEDV'],(raw_data[(raw_data.AGE>80)&(raw_data.AGE<100)])['MEDV']
			mean1,mean2,mean3 = sub_data1.mean(axis =0),sub_data2.mean(axis =0),sub_data3.mean(axis =0)
			std1,std2,std3 = sub_data1.std(axis =0),sub_data2.std(axis =0),sub_data3.std(axis =0)
			n1,n2,n3 = len(sub_data1),len(sub_data2),len(sub_data3)
			conf = 95
			cint1,cint2,cint3 = confidence_interval(std1,n1,conf),confidence_interval(std2,n2,conf),confidence_interval(std3,n3,conf)
			plt.errorbar(x_values,[mean1,mean2,mean3],[cint1,cint2,cint3]);plt.xlim(0,100);plt.grid(True)
			plt.xlabel('Proportion of house before 1940 (directly prop to Age)');plt.ylabel('Confidence interval of Mean House prices');plt.title('Confidence interval of mean values of house prices by AGE sub ranges')
		elif choice ==5:
			x_values = [0.5,0.7,0.85]
			sub_data1,sub_data2,sub_data3 = (raw_data[(raw_data.NOX>0.4)&(raw_data.NOX<0.6)])['MEDV'],(raw_data[(raw_data.NOX>0.6)&(raw_data.NOX<0.8)])['MEDV'],(raw_data[(raw_data.NOX>0.8)&(raw_data.NOX<0.9)])['MEDV']
			mean1,mean2,mean3 = sub_data1.mean(axis =0),sub_data2.mean(axis =0),sub_data3.mean(axis =0)
			std1,std2,std3 = sub_data1.std(axis =0),sub_data2.std(axis =0),sub_data3.std(axis =0)
			n1,n2,n3 = len(sub_data1),len(sub_data2),len(sub_data3)
			conf = 95
			cint1,cint2,cint3 = confidence_interval(std1,n1,conf),confidence_interval(std2,n2,conf),confidence_interval(std3,n3,conf)
			plt.errorbar(x_values,[mean1,mean2,mean3],[cint1,cint2,cint3]);plt.xlim(0.3,0.9);plt.grid(True)
			plt.xlabel('NOX concentration');plt.ylabel('Confidence interval of Mean House prices');plt.title('Confidence interval of mean values of house prices by NOX sub ranges')
		elif choice ==3:
			x_values = [7.5,22.5]
			sub_data1,sub_data2 = raw_data[(raw_data.INDUS>0)&(raw_data.INDUS<16)],raw_data[(raw_data.INDUS>16)&(raw_data.INDUS<30)]
			mean1,mean2 = sub_data1['MEDV'].mean(axis =0),sub_data2['MEDV'].mean(axis =0)
			std1,std2 = sub_data1['MEDV'].std(axis =0),sub_data2['MEDV'].std(axis =0)
			n1,n2 = len(sub_data1),len(sub_data2)
			cint1,cint2 = confidence_interval(std1,n1,95),confidence_interval(std2,n2,95)
			plt.errorbar(x_values,[mean1,mean2],[cint1,cint2]);plt.xlim(0,30);plt.grid(True)
			plt.xlabel('Proportion of non-retail business acres');plt.ylabel('Confidence interval of Mean House prices');plt.title('Confidence interval of mean values of house prices by INDUS sub ranges')
		elif choice ==4:
			x_values = [0,1]
			sub_data1,sub_data2 = raw_data[(raw_data.CHAS==0)],raw_data[(raw_data.CHAS==1)]
			mean1,mean2 = sub_data1['MEDV'].mean(axis =0),sub_data2['MEDV'].mean(axis =0)
			std1,std2 = sub_data1['MEDV'].std(axis =0),sub_data2['MEDV'].std(axis =0)
			n1,n2 = len(sub_data1),len(sub_data2)
			cint1,cint2 = confidence_interval(std1,n1,95),confidence_interval(std2,n2,95)
			plt.errorbar(x_values,[mean1,mean2],[cint1,cint2]);plt.xlim(-0.5,1.5);plt.grid(True)
			plt.xlabel('Charles river tract bound');plt.ylabel('Confidence interval of Mean House prices');plt.title('Confidence interval of mean values of house prices by CHAS values')
		elif choice ==1:
			x_values = [0.5,1.5,11]
			sub_data1,sub_data2,sub_data3 = raw_data[(raw_data.CRIM>0)&(raw_data.CRIM<1)],raw_data[(raw_data.CRIM>1)&(raw_data.CRIM<2)],raw_data[(raw_data.CRIM>2)&(raw_data.CRIM<20)]
			mean1,mean2,mean3 = sub_data1['MEDV'].mean(axis =0),sub_data2['MEDV'].mean(axis =0),sub_data3['MEDV'].mean(axis =0)
			std1,std2,std3 = sub_data1['MEDV'].std(axis =0),sub_data2['MEDV'].std(axis =0),sub_data3['MEDV'].std(axis =0)
			n1,n2,n3 = len(sub_data1),len(sub_data2),len(sub_data3)
			cint1,cint2,cint3 = confidence_interval(std1,n1,95),confidence_interval(std2,n2,95),confidence_interval(std3,n3,95)
			plt.errorbar(x_values,[mean1,mean2,mean3],[cint1,cint2,cint3]);plt.xlim(0,30);plt.grid(True)
			plt.xlabel('Crime rate');plt.ylabel('Confidence interval of Mean House prices');plt.title('Confidence interval of mean values of house prices by CRIM sub ranges')

			# x_values = [2.5,12.5]
			# sub_data1,sub_data2 = raw_data[(raw_data.CRIM>0)&(raw_data.CRIM<5)],raw_data[(raw_data.CRIM>5)&(raw_data.CRIM<20)]
			# mean1,mean2 = sub_data1['MEDV'].mean(axis =0),sub_data2['MEDV'].mean(axis =0)
			# std1,std2 = sub_data1['MEDV'].std(axis =0),sub_data2['MEDV'].std(axis =0)
			# n1,n2 = len(sub_data1),len(sub_data2)
			# cint1,cint2 = confidence_interval(std1,n1,95),confidence_interval(std2,n2,95)
			# plt.errorbar(x_values,[mean1,mean2],[cint1,cint2]);plt.xlim(0,30);plt.grid(True)
			# plt.xlabel('Crime rate');plt.ylabel('Confidence interval of Mean House prices');plt.title('Confidence interval of mean values of house prices by CRIM sub ranges')

			# x_values = [5,15]
			# sub_data1,sub_data2 = raw_data[(raw_data.CRIM>0)&(raw_data.CRIM<10)],raw_data[(raw_data.CRIM>10)&(raw_data.CRIM<20)]
			# mean1,mean2 = sub_data1['MEDV'].mean(axis =0),sub_data2['MEDV'].mean(axis =0)
			# std1,std2 = sub_data1['MEDV'].std(axis =0),sub_data2['MEDV'].std(axis =0)
			# n1,n2 = len(sub_data1),len(sub_data2)
			# cint1,cint2 = confidence_interval(std1,n1,95),confidence_interval(std2,n2,95)
			# plt.errorbar(x_values,[mean1,mean2],[cint1,cint2]);plt.xlim(0,30);plt.grid(True)
			# plt.xlabel('Crime rate');plt.ylabel('Confidence interval of Mean House prices');plt.title('Confidence interval of mean values of house prices by CRIM sub ranges')
		elif choice ==2:
			x_values = [0,30,80]
			sub_data1,sub_data2,sub_data3 = raw_data[(raw_data.ZN == 0)],raw_data[(raw_data.ZN>0)&(raw_data.ZN<60)],raw_data[(raw_data.ZN>60)&(raw_data.CRIM<100)]
			mean1,mean2,mean3 = sub_data1['MEDV'].mean(axis =0),sub_data2['MEDV'].mean(axis =0),sub_data3['MEDV'].mean(axis =0)
			std1,std2,std3 = sub_data1['MEDV'].std(axis =0),sub_data2['MEDV'].std(axis =0),sub_data3['MEDV'].std(axis =0)
			n1,n2,n3 = len(sub_data1),len(sub_data2),len(sub_data3)
			cint1,cint2,cint3 = confidence_interval(std1,n1,95),confidence_interval(std2,n2,95),confidence_interval(std3,n3,95)
			plt.errorbar(x_values,[mean1,mean2,mean3],[cint1,cint2,cint3]);plt.xlim(-10,100);plt.grid(True)
			plt.xlabel('Proportion of land zoned for >25K');plt.ylabel('Confidence interval of Mean House prices');plt.title('Confidence interval of mean values of house prices by ZN sub ranges')
		plt.show()

def hypothesis_test(input_list):
	for element in input_list:
		if element == 13:
			pop_std_of_mean = (float(raw_data[['MEDV']].std(axis=0)))/np.sqrt(len(raw_data))
			z_value = (19-39)/pop_std_of_mean
			print 'Z_value',z_value
			print float(norm.cdf(z_value))
			

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

Plotting: http://pandas.pydata.org/pandas-docs/version/0.15.0/visualization.html#histograms

"""
#############Execution################
#histograms([9])
#scatter_plots([1]) 
#distribution_of_medv()
mean_plots([2])
#hypothesis_test([13])
#print pd.unique(raw_data['CHAS'])




#df.to_excel('path_to_file.xlsx', sheet_name='Sheet1')   #write the file to xls format