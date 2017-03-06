import sys
from Regression import Regression
import Constants


def main():
    """
    Entry point to the application
    :return: None
    """
    arguments = sys.argv

    data = Constants.DATA_FILE
    is_menu = False
    is_help = False

    for arg_index in range(1, len(arguments)):
        if arguments[arg_index] == "--data_set" or arguments[arg_index] == "-d":
            arg_index += 1
            if arg_index > len(arguments):
                raise RuntimeError("Please provide the path to dataset")
            data = arguments[arg_index]
        elif arguments[arg_index] == "--menu" or arguments[arg_index] == "-m":
            is_menu = True
            arg_index += 1
        elif arguments[arg_index] == "--help" or arguments[arg_index] == "-h":
            is_help = True
            arg_index += 1

    if is_menu:
        menu(data)
    elif is_help:
        help_menu(arguments[0])
    else:
        calculate_everything(data)
    return


def least_square_regression(regression, degree=Constants.DEFAULT_DEGREE, multi_feature=False):
    """
    Least Square regression
    :param regression: Regression class Object
    :param degree: degree of the regression model
    :param multi_feature: does it have multiple feature
    :return: trained coefficients of least square regression
    """
    if degree != 0:
        coefficients = regression.regression(is_ridge=False, degree=degree)
        return coefficients
    else:
        raise RuntimeError("Please provide the degree of the regression model")


def cross_validate(regression, is_ridge=False, degree=Constants.DEFAULT_DEGREE, multi_feature=False):
    """
    Cross validates the regression model using K-fold cross validation
    :param regression: Regression class Object
    :param is_ridge: is the regression ridge regression?
    :param degree: degree of the regression model
    :param multi_feature: does the regression model have multiple feature
    :return: rmse for k-fold cross validation
    """
    if degree != 0:
        error = regression.cross_validate(is_ridge=is_ridge, fold_count=Constants.FOLD_COUNT, degree=degree,
                                          lambda_list=Constants.LAMBDA_LIST)

        return error
    else:
        raise RuntimeError("Please provide the degree of the regression model")


def ridge_regression(regression, degree=Constants.DEFAULT_DEGREE, multi_feature=False):
    """
    Performs Ridge Regression
    :param regression: Regression class object
    :param degree: degree of the regression model
    :param multi_feature: does the model contain multiple features
    :return: list of trained coefficients
    """
    if degree != 0:
        coefficients = regression.regression(is_ridge=True, degree=degree)
        return coefficients
    else:
        raise RuntimeError("Please provide the degree of the regression model")


def calculate_everything(data):
    """
    Runs everything, this is the default method run
    :return: None
    """

    regression = Regression(data_file=data, actual_output=Constants.OUTPUT_FEATURE,
                            iterations=Constants.ITERATIONS, step_size=Constants.STEP_SIZE)
    regression.set_features(Constants.SINGLE_FEATURE)

    print "==========================="
    print "1. Least Square Regression"
    print "==========================="

    print "\n\ti. wo + w1 * (sqft_living) + w2 * (sqft_living)[2] :"
    lsr_square_coefficient = least_square_regression(regression,2)
    print "\n\tTrained Coefficients :"
    print "\t========================"
    print_weights(lsr_square_coefficient)

    print "\n\tii. wo + w1 * (sqft_living) + w2 * (sqft_living)[2] + w3 * (sqft_living)[3] + w4 * (sqft_living)[4] :"
    lsr_forth_coefficient = least_square_regression(regression, 4)
    print "\n\tTrained Coefficients :"
    print "\t========================"
    print_weights(lsr_forth_coefficient)

    print "\n\tiii. wo + w1 * (sqft_living) + w2 * (sqft_lot) + w3 * (bedrooms) + w4 * (bathrooms) :"
    regression.set_features(Constants.MULTIPLE_FEATURES)
    lsr_multi_feature_coefficient = least_square_regression(regression)
    print "\n\tTrained Coefficients :"
    print "\t========================"
    print_weights(lsr_multi_feature_coefficient)

    print "\n====================================================="
    print "2. 10-Fold Cross Validation (Least Square Regression)"
    print "====================================================="

    print "\n\ti. wo + w1 * (sqft_living) + w2 * (sqft_living)[2] :"
    regression.set_features(Constants.SINGLE_FEATURE)
    lsr_square_rmse = cross_validate(regression, degree=2)
    print "\n\tRoot Mean Square Error (RMSE) :"
    print "\t========================"
    print_rmse(lsr_square_rmse)

    print "\n\tii. wo + w1 * (sqft_living) + w2 * (sqft_living)[2] + w3 * (sqft_living)[3] + w4 * (sqft_living)[4] :"
    lsr_forth_rmse = cross_validate(regression, degree=4)
    print "\n\tRoot Mean Square Error (RMSE) :"
    print "\t========================"
    print_rmse(lsr_forth_rmse)

    print "\n\tiii. wo + w1 * (sqft_living) + w2 * (sqft_lot) + w3 * (bedrooms) + w4 * (bathrooms) :"
    regression.set_features(Constants.MULTIPLE_FEATURES)
    lsr_multi_feature_rmse = cross_validate(regression)
    print "\n\tRoot Mean Square Error (RMSE) :"
    print "\t========================"
    print_rmse(lsr_multi_feature_rmse)

    print "\n==========================="
    print "3. Ridge Regression"
    print "==========================="

    print "\n\ti. wo + w1 * (sqft_living) + w2 * (sqft_living)[2] :"
    regression.set_features(Constants.SINGLE_FEATURE)
    ridge_square_coefficient = ridge_regression(regression, degree=2)
    print "\n\tTrained Coefficients :"
    print "\t========================"
    print_weights(ridge_square_coefficient)

    print "\n\tii. wo + w1 * (sqft_living) + w2 * (sqft_living)[2] + w3 * (sqft_living)[3] + w4 * (sqft_living)[4] :"
    ridge_forth_coefficient = ridge_regression(regression, degree=4)
    print "\n\tTrained Coefficients :"
    print "\t========================"
    print_weights(ridge_forth_coefficient)

    print "\n\tiii. wo + w1 * (sqft_living) + w2 * (sqft_lot) + w3 * (bedrooms) + w4 * (bathrooms) :"
    regression.set_features(Constants.MULTIPLE_FEATURES)
    ridge_multi_feature_coefficient = ridge_regression(regression)
    print "\n\tTrained Coefficients :"
    print "\t========================"
    print_weights(ridge_multi_feature_coefficient)

    print "\n================================"
    print "4. Model Selection"
    print "================================"

    regression.iterations = Constants.MODEL_SELECTION_ITERATION
    print "\n\ti. wo + w1 * (sqft_living) + w2 * (sqft_living)[2] :"
    regression.set_features(Constants.SINGLE_FEATURE)
    lsr_sqr_model_selection = cross_validate(regression, is_ridge=True, degree=2)
    print "\n\tSelected Models :"
    print "\t========================\n"

    for error_index in range(0, len(lsr_sqr_model_selection)):
        for lamda, model_rmse in lsr_sqr_model_selection[error_index].iteritems():
            print "\tlamda: " + str(lamda)
            print "\tAverage RMSE : " + str(model_rmse)

    print "\n\tii. wo + w1 * (sqft_living) + w2 * (sqft_living)[2] + w3 * (sqft_living)[3] + w4 * (sqft_living)[4] :"
    lsr_forth_model_selection = cross_validate(regression, is_ridge=True, degree=4)
    print "\n\tSelected Models :"
    print "\t========================\n"

    for error_index in range(0,len(lsr_forth_model_selection)):
        for lamda, model_rmse in lsr_forth_model_selection[error_index].iteritems():
            print "\tlamda: " + str(lamda)
            print "\tAverage RMSE : " + str(model_rmse)

    print "\n\tiii. wo + w1 * (sqft_living) + w2 * (sqft_lot) + w3 * (bedrooms) + w4 * (bathrooms) :"
    regression.set_features(Constants.MULTIPLE_FEATURES)
    lsr_multi_feature_model_selection = cross_validate(regression, is_ridge=True)
    print "\n\tSelected Models :"
    print "\t========================\n"

    for error_index in range(0, len(lsr_multi_feature_model_selection)):
        for lamda, model_rmse in lsr_multi_feature_model_selection[error_index].iteritems():
            print "\tlamda: " + str(lamda)
            print "\tAverage RMSE : " + str(model_rmse)



def _regression_option():
    """
    Displays user interface to select model type for regression
    :return: degree, multi_feature
    """
    multi_feature = False
    user_choice = "c"
    degree = 1
    while True:
        print "\n\ta. Regression Using Single Feature"
        print "\tb. Regression Using Multiple Feature"
        print "\tc. Exit"
        user_choice = raw_input("\n\tPlease choose your option (a,b or c) : ")
        if user_choice == "a" or user_choice == "b" or user_choice == "c":
            break
        else:
            print "\n\n\tERROR: Please select a correct option."
    if user_choice == "a":
        while True:
            deg = raw_input("\n\tPlease enter the degree of the model : ")
            if deg.isdigit():
                break
        degree = int(deg)

    elif user_choice == "b":
        multi_feature = True

    elif user_choice == "c":
        degree = 0

    return degree, multi_feature


def menu(data):
    """
    User Interface for the application
    :return: None
    """
    regression = Regression(data_file=data, actual_output=Constants.OUTPUT_FEATURE,
                            iterations=Constants.ITERATIONS, step_size=Constants.STEP_SIZE)
    print "***************************************************************************************************"
    print "*                                            Regression                                           *"
    print "***************************************************************************************************"
    while True:
        print "\n\n\t1. Least Square Regression "
        print "\t2. Cross Validate Least Square Regression"
        print "\t3. Ridge Regression"
        print "\t4. Select Model from Ridge Regression"
        print "\t5. Exit\n"
        try:
            user_choice = int(raw_input("\tPlease select your option (1 - 5) : "))
            if user_choice == 1:
                degree, multiple_feature = _regression_option()
                features = Constants.MULTIPLE_FEATURES if multiple_feature else Constants.SINGLE_FEATURE
                regression.set_features(features)
                coefficients = least_square_regression(regression, degree)
                print coefficients
            elif user_choice == 2:
                degree, multiple_feature = _regression_option()
                features = Constants.MULTIPLE_FEATURES if multiple_feature else Constants.SINGLE_FEATURE
                regression.set_features(features)
                rmse = cross_validate(regression, is_ridge=False, degree=degree)
                print rmse
            elif user_choice == 3:
                degree, multiple_feature = _regression_option()
                features = Constants.MULTIPLE_FEATURES if multiple_feature else Constants.SINGLE_FEATURE
                regression.set_features(features)
                coefficients = ridge_regression(regression, degree=degree)
                print coefficients
            elif user_choice == 4:
                regression.iterations = Constants.MODEL_SELECTION_ITERATION
                degree, multiple_feature = _regression_option()
                features = Constants.MULTIPLE_FEATURES if multiple_feature else Constants.SINGLE_FEATURE
                regression.set_features(features)
                rmse = cross_validate(regression, is_ridge=True, degree=degree)
                print rmse
            elif user_choice == 5:
                break
            else:
                raise ValueError("Invalid option")
        except ValueError as err:
            print err
            print "\n\n\tERROR: Please select a correct option."


def help_menu(filename):
    """
    Displays help information for the application
    :param filename: name of the main file
    :return: None
    """
    print "\nHELP: "
    print "====="
    print "Command : python " + filename + " [options]"
    print "\n\t Options:"
    print "\t ========"
    print "\t -m | --menu : Displays a menu"
    print "\t -h | --help : Displays the help\n"


def print_weights(weights_list):
    """
    Prints weight list
    :param weights_list: list of weights to be printed
    :return: None
    """
    for index in range(0,len(weights_list)):
        print "\tw" + str(index) + " = " + str(weights_list[index])


def print_rmse(rmse_list):
    """
    Prints error per fold
    :param rmse_list: list of rmse to be printed
    :return: None
    """
    print "\tFOLD\tRMSE"
    print "\t====\t===="
    for index in range(0, len(rmse_list)):
        print "\t" + str(index+1) + "\t" + str(rmse_list[index])


if __name__ == "__main__":
    main()


