import sys
from Regression import Regression
import Constants


def main():
    arguments = sys.argv
    if len(arguments) > 2:
        print "Invalid command line arguments"
    elif len(arguments) == 2:
        if arguments[1] == "--menu" or arguments[1] == "-m":
            menu()
        elif arguments[1] == "--help" or arguments[1] == "-h":
            help_menu(arguments[0])
        else:
            print "Invalid command line arguments"
    else:
        calculate_everything()
    return


def least_square_regression(degree=Constants.DEFAULT_DEGREE, multi_feature=False):
    """
    Least Square regression
    :param degree: degree of the regression model
    :param multi_feature: does it have multiple feature
    :return:
    """
    if degree != 0:
        input_features = Constants.MULTIPLE_FEATURES if multi_feature else Constants.SINGLE_FEATURE

        regression = Regression(Constants.DATA_FILE, actual_output=Constants.OUTPUT_FEATURE,features=input_features,iterations=Constants.ITERATIONS, step_size=Constants.STEP_SIZE)
        coefficients = regression.regression(is_ridge=False, degree=degree)
        return coefficients
    else:
        raise RuntimeError("Please provide the degree of the regression model")


def cross_validate(is_ridge=False, degree=Constants.DEFAULT_DEGREE, multi_feature=False):

    if degree != 0:
        input_features = Constants.MULTIPLE_FEATURES if multi_feature else Constants.SINGLE_FEATURE

        regression = Regression(Constants.DATA_FILE, actual_output=Constants.OUTPUT_FEATURE, features=input_features,
                                iterations=Constants.ITERATIONS, step_size=Constants.STEP_SIZE)
        error = regression.cross_validate(fold_count=Constants.FOLD_COUNT, degree=degree)

        return error
    else:
        raise RuntimeError("Please provide the degree of the regression model")


def ridge_regression(degree=Constants.DEFAULT_DEGREE, multi_feature=False):
    degree, multi_feature = _regression_option()

    if degree != 0:
        input_features = Constants.MULTIPLE_FEATURES if multi_feature else Constants.SINGLE_FEATURE

        regression = Regression(Constants.DATA_FILE, actual_output=Constants.OUTPUT_FEATURE, features=input_features,
                                iterations=Constants.ITERATIONS, step_size=Constants.STEP_SIZE)
        coefficients = regression.regression(is_ridge=True, degree=degree)
        return coefficients
    else:
        raise RuntimeError("Please provide the degree of the regression model")



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


def menu():
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
                least_square_regression(degree, multiple_feature)
            elif user_choice == 2:
                degree, multiple_feature = _regression_option()
                cross_validate(is_ridge=False, degree=degree, multi_feature=multiple_feature)
            elif user_choice == 3:
                degree, multiple_feature = _regression_option()
                ridge_regression(degree=degree, multi_feature=multiple_feature)
            elif user_choice == 4:
                degree, multiple_feature = _regression_option()
                cross_validate(True, degree, multiple_feature)
            elif user_choice == 5:
                break
            else:
                raise ValueError("Invalid option")
        except ValueError as err:
            print err
            print "\n\n\tERROR: Please select a correct option."


def help_menu(filename):
    print "\nHELP: "
    print "====="
    print "Command : python " + filename + " [options]"
    print "\n\t Options:"
    print "\t ========"
    print "\t -m | --menu : Displays a menu"
    print "\t -h | --help : Displays the help\n"


def calculate_everything():
    print "1. Least Square Regression"
    print "==========================="

    print "\n ======================================================"
    print "i. wo + w1 * (sqft_living) + w2 * (sqft_living)[2] :"
    lsr_square_coefficient = least_square_regression(2)
    print "TrainedCoefficients :"
    print lsr_square_coefficient

    print "\n ======================================================"
    print "ii. wo + w1 * (sqft_living) + w2 * (sqft_living)[2] + w3 * (sqft_living)[3] + w4 * (sqft_living)[4] :"
    lsr_forth_coefficient = least_square_regression(4)
    print "TrainedCoefficients :"
    print lsr_forth_coefficient

    print "\n ======================================================"
    print "iii. wo + w1 * (sqft_living) + w2 * (sqft_living) + w3 * (sqft_lot) + w4 * (bedrooms) + w5 * (bathrooms) :"
    lsr_multi_feature_coefficient = least_square_regression(multi_feature=True)
    print "TrainedCoefficients :"
    print lsr_multi_feature_coefficient


if __name__ == "__main__":
    main()


