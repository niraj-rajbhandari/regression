
import sys
from Regression import Regression
import Constants


def main():
    print "***************************************************************************************************"
    print "*                                            Regression                                           *"
    print "***************************************************************************************************"
    while True:
        print "\n\n\t1. Least Square Regression "
        print "\t2. Cross Validate Least Square Regression"
        print "\t3. Ridge Regression"
        print "\t4. Select Model from Ridge Regression"
        print "\t5. Exit\n"
        # try:
        if True:
            user_choice = int(raw_input("\tPlease select your option (1 - 5) : "))
            if user_choice == 1:
                least_square_regression()
            elif user_choice == 2:
                cross_validate()
            elif user_choice == 3:
                ridge_regression()
            elif user_choice == 4:
                cross_validate(True)
            elif user_choice == 5:
                break
            else:
                raise ValueError("Invalid option")
        # except ValueError as err:
        #     print err
        #     print "\n\n\tERROR: Please select a correct option."
    return


def least_square_regression():

    degree, multi_feature = _regression_option()

    if degree != 0:
        input_features = Constants.MULTIPLE_FEATURES if multi_feature else Constants.SINGLE_FEATURE

        regression = Regression(Constants.DATA_FILE, actual_output=Constants.OUTPUT_FEATURE,features=input_features,iterations=Constants.ITERATIONS, step_size=Constants.STEP_SIZE)
        coefficients = regression.regression(is_ridge=False, degree=degree)
        print coefficients

    return


def cross_validate(is_ridge=False):
    degree, multi_feature = _regression_option()
    if degree != 0:
        input_features = Constants.MULTIPLE_FEATURES if multi_feature else Constants.SINGLE_FEATURE

        regression = Regression(Constants.DATA_FILE, actual_output=Constants.OUTPUT_FEATURE, features=input_features,
                                iterations=Constants.ITERATIONS, step_size=Constants.STEP_SIZE)
        error = regression.cross_validate(fold_count=Constants.FOLD_COUNT, degree=degree)
        print "RMSE: "
        print error

    return

def ridge_regression():
    print "Still working on it"
    pass


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


if __name__ == "__main__":
    main()


