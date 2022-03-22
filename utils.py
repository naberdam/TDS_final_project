import pandas as pd
import csv

HOUSE_PRICES_PATH = 'data/test_raw_datasets/house_prices.csv'
BANK_CHURNERS_PATH = 'data/test_raw_datasets/BankChurners.csv'
STUDENT_MAT_PATH = 'data/test_raw_datasets/student-mat.csv'
FOOD_CODED_PATH = 'data/test_raw_datasets/food_coded.csv'


def get_relevant_columns_from_test_sets(datasets):
    house_prices_columns = []
    bank_churners_columns = []
    student_mat_columns = []
    food_coded_columns = []
    with open('test_y.csv', 'r') as read_file:
        reader = csv.reader(read_file)
        for row in reader:
            if row[0] in datasets['house_prices']:
                house_prices_columns.append(row[0])
            elif row[0] in datasets['bank_churners']:
                bank_churners_columns.append(row[0])
            elif row[0] in datasets['student_mat']:
                student_mat_columns.append(row[0])
            elif row[0] in datasets['food_coded']:
                food_coded_columns.append(row[0])
    return {'house_prices': house_prices_columns, 'bank_churners': bank_churners_columns,
            'student_mat': student_mat_columns, 'food_coded': food_coded_columns}


def load_datasets():
    house_prices = pd.read_csv(HOUSE_PRICES_PATH)
    bank_churners = pd.read_csv(BANK_CHURNERS_PATH)
    student_mat = pd.read_csv(STUDENT_MAT_PATH)
    food_coded = pd.read_csv(FOOD_CODED_PATH)
    return {'house_prices': house_prices, 'bank_churners': bank_churners,
            'student_mat': student_mat, 'food_coded': food_coded}


def check_accuracy(results, test_y_path):
    with open(test_y_path, 'r') as readfile:
        accuracy = 0
        reader = csv.reader(readfile)
        row_count = 0
        for row in reader:
            row_count += 1
            if row[0] in results and results[row[0]] == int(row[1]):
                accuracy += 1
        accuracy /= row_count
    return accuracy

