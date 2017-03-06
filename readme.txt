Requirements:
=============

1. Python 2.7
2. Python Packages:
	a. Numpy
	b. Pandas

Run the program:
================

If you can run a Makefile

	$ make data_set=$(path-to-the-data-set)

	OR,

	$ make

Otherwise,

	 $ python main.py -d $(path-to-the-data-set)

	OR,

	 $ python main.py

N.B: if the path to the dataset is not provided, the default dataset "kc_house_data.csv" available in this folder will be used.


Points to be noted:
===================

1. Number of iterations used : 
	a. model selection : 1000
	b. others : 10000

2. Stepsize used : 1e-14
3. Data set has not been normalized
4. Since the program uses very small step_size and the number of iteration is small in order to eliminate error caused by "nan" as the dataset is not normalized, the program might not be able to find the global optimum, hence giving large error and coefficients.	
5. Size of the coefficient and error increases as the degree of the regression model increases 