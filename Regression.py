import numpy as np
import pandas as panda
import pdb


class Regression(object):

    """
    Different Regressions
    @author Niraj Rajbhandari

    Private properties:

    data_file: name of the file that stores the training data
    iterations: number of iterations to run before selecting the regression model
    features: total features to consider in regression model
    actual_output: the column that gives the actual result

    """

    FEATURE_MATRIX_FIRST_COL_VAL = 1
    COLUMN_AXIS = 1
    ROW_AXIS = 0

    def __init__(self,data_file,actual_output, step_size=1.0e-10, features=[], iterations=10000):
        if len(features) == 0:
            raise RuntimeError("Please select the feature to build the model")

        self.step_size = step_size
        self.iterations = iterations
        self.actual_output = actual_output
        self.features = features
        self.features.append(actual_output)
        if len(features) > 0:
            self.data = panda.read_csv(data_file, usecols=self.features)
        else:
            self.data = panda.read_csv(data_file)

        self.features.remove("price")

    @staticmethod
    def initialize_weights(weight_count):
        """
        Initialize the weights to 0.0
        :param weight_count: number of coefficients to be used in the model
        :return: Initialized weights matrix
        """
        weight_vector = np.zeros(weight_count)
        return weight_vector.T

    def _get_output_matrix(self, data):
        """
        Gets actual output matrix
        :param data: data from which output is to be retrieved
        :return: actual output matrix
        """
        output_matrix = data[self.actual_output].as_matrix()
        return output_matrix.T

    def _get_feature_matrix(self, data, degree=1 ):
        """
        Gets the feature matrix
        :param degree: degree of the model to be selected
        :return: feature matrix
        """
        feature_matrix = data[self.features].as_matrix()
        temp_feature_matrix = feature_matrix
        for deg in range(2, degree+1):
            powered_feature = np.power(feature_matrix,deg)
            temp_feature_matrix = np.concatenate((temp_feature_matrix,powered_feature), axis=self.COLUMN_AXIS)

        feature_matrix = np.insert(temp_feature_matrix, 0, self.FEATURE_MATRIX_FIRST_COL_VAL, axis=self.COLUMN_AXIS) # adds 1 to the start of all the rows

        return feature_matrix

    def regression(self,is_ridge=False, degree=1, data=panda.DataFrame({'A': []}), lamda=0.1):
        """
        Performs Gradient Descent to select a model
        :param is_ridge: is the type of regression ridge
        :param degree: degree of the model to be selected
        :param data: training data for regression model generation
        :param lamda: penalty factor for high magnitude in ridge regression
        :return: list of new coefficients of selected model
        """
        training_data = self.data if data.empty else data

        weight_count = degree+1 if len(self.features) == 1 else len(self.features)+1
        weight_matrix = self.initialize_weights(weight_count)
        feature_matrix = self._get_feature_matrix(degree=degree, data=training_data)
        actual_output_matrix = self._get_output_matrix(training_data)

        for iteration in range(1,self.iterations):
            if iteration % 2 == 0:
                self.step_size /= pow(iteration,4)  # decrease stepsize by i^4 every second iteration
            rss_gradient_ii = np.dot(feature_matrix, weight_matrix)  # Hw
            rss_gradient_iii = np.subtract(actual_output_matrix, rss_gradient_ii.T)  # y-Hw
            rss_gradient_final = np.dot(feature_matrix.T, rss_gradient_iii)  # Ht(y-Hw) => Gradient of RSS
            new_weight_i = (2*self.step_size) / feature_matrix.shape[0]  # (2*step_size)/N
            new_weight_ii = np.multiply(new_weight_i,rss_gradient_final)  # (2*step_size*Ht(y-Hw)) / N
            weight_matrix = np.add(weight_matrix,new_weight_ii)  # w + (2*step_size*Ht(y-Hw)) / N

        return weight_matrix.tolist()

    def cross_validate(self, fold_count, lambda_list=[0.1], is_ridge=False, degree=1):
        """
        Nested cross validation for the regression
        :param fold_count: value of k in k-fold-cross-validation
        :param lambda_list: lists of lambda to be validated
        :param is_ridge: Checks if the regression done is ridge or not
        :param degree: degree of the regression model
        :return: list of weights and lambda that have been cross validated and tested
        """
        data_size = self.get_data_size()
        unit_test_fold_size = data_size / fold_count
        test_cv_limit = unit_test_fold_size
        test_cv_offset = 0
        test_error = {} if is_ridge else []

        for test_set_count in range(0, fold_count):
            remaining_data, test_data = self._partition_data(self.data, test_cv_limit, test_cv_offset)
            rem_data_size = len(remaining_data)
            unit_validation_fold_size = rem_data_size / fold_count
            val_cv_limit = unit_validation_fold_size
            val_cv_offset = 0

            if is_ridge:
                lamda_errors = {}
                for lamda in lambda_list:
                    for validation_set_count in range(0, fold_count):
                        training_data ,validation_data = self._partition_data(remaining_data, val_cv_limit, val_cv_offset)
                        """
                        TODO: train the regression model with training data and validate the generated model
                        """
                        trained_weight_list = self.regression(is_ridge=is_ridge, degree=degree,data=training_data,lamda=lamda)
                        val_cv_offset = val_cv_limit
                        val_cv_limit += unit_validation_fold_size
            else:
                trained_weight_list = self.regression(is_ridge=is_ridge, degree=degree, data=remaining_data)
                rmse = self.calculate_error(data=test_data, weight_list=trained_weight_list, degree=degree)
                test_error.append(rmse)

            """
            TODO: test the validated model with the test data
            """
            test_cv_offset = test_cv_limit
            test_cv_limit += unit_test_fold_size
        return test_error

    @staticmethod
    def _partition_data(data,test_set_end_index, test_set_start_index=0):
        """
        Partitions the data into training and test set
        :param data: data to be partitioned
        :param test_set_end_index: start index of test set
        :param test_set_start_index: end index of the test set
        :return: pandas.dataframe training_set, test_set
        """
        data_size = len(data)
        test_set = data.iloc[test_set_start_index:test_set_end_index]
        training_set = data.iloc[test_set_end_index: data_size]

        return training_set, test_set

    def calculate_error(self, data, weight_list, degree):
        """
        Calculate the error of the regression model
        :param data: test data
        :param weight_list: lists of weight for the model
        :param degree: degree of the regression model
        :return: Root mean square error (RMSE)
        """
        feature_matrix = self._get_feature_matrix(degree=degree, data=data)
        output_matrix = self._get_output_matrix(data)
        weights = np.array(weight_list).T
        rss_i = np.dot(feature_matrix,weights)
        rss_ii = np.subtract(output_matrix, rss_i)
        rss_iii = np.dot(rss_ii.T, rss_ii)
        rss_final = np.divide(rss_iii, len(data))
        return self.square_root(rss_final)

    def get_data_size(self):
        """
        Get size of the data in the object
        :return: size of the data
        """
        return len(self.data)

    @staticmethod
    def square_root(number):
        """
        Calculates the square root of a number
        :param number: number whose square root is to be calculated
        :return: square root of the number
        """
        return pow(number, 0.5)








