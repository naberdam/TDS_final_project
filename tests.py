import numpy as np
import pandas as pd
from enum import Enum
import csv

HOUSE_PRICES_PATH = 'data/test_raw_datasets/house_prices.csv'
BANK_CHURNERS_PATH = 'data/test_raw_datasets/BankChurners.csv'
STUDENT_MAT_PATH = 'data/test_raw_datasets/student-mat.csv'
FOOD_CODED_PATH = 'data/test_raw_datasets/food_coded.csv'


class Result(Enum):
    numeric = 1
    categorical = 2


def test_with_threshold(columns, threshold):
    results = {}
    for col in columns:
        results[col] = Result.numeric.name if columns[col].nunique() > threshold else Result.categorical.name
    return results
#     numeric_columns = dtf.dtypes[(dtf.dtypes=="float64") | (dtf.dtypes=="int64")].index.tolist()
#     very_numerical = [nc for nc in numeric_columns if dtf[nc].nunique()>20]
#     categorical_columns = [c for c in dtf.columns if c not in numeric_columns]


def get_numeric_columns(dataset):
    integer_columns = dataset.dtypes[(dataset.dtypes == "float64") | (dataset.dtypes == "int64")].index.tolist()
    return dataset[integer_columns]


def create_draft_test_y():
    house_prices = pd.read_csv(HOUSE_PRICES_PATH)
    bank_churners = pd.read_csv(BANK_CHURNERS_PATH)
    student_mat = pd.read_csv(STUDENT_MAT_PATH)
    food_coded = pd.read_csv(FOOD_CODED_PATH)
    datasets = [house_prices, bank_churners, student_mat, food_coded]
    with open('draft_test_y.csv', 'w', newline='') as test_y_file:
        writer = csv.writer(test_y_file)
        for dataset in datasets:
            only_numeric_columns = get_numeric_columns(dataset)
            for col in only_numeric_columns:
                writer.writerow([col])


def get_relevant_columns_from_test_sets():
    house_prices = pd.read_csv(HOUSE_PRICES_PATH)
    bank_churners = pd.read_csv(BANK_CHURNERS_PATH)
    student_mat = pd.read_csv(STUDENT_MAT_PATH)
    food_coded = pd.read_csv(FOOD_CODED_PATH)
    house_prices_columns = []
    bank_churners_columns = []
    student_mat_columns = []
    food_coded_columns = []
    with open('test_y.csv', 'r') as read_file:
        reader = csv.reader(read_file)
        for row in reader:
            if row[0] in house_prices:
                house_prices_columns.append(row[0])
            elif row[0] in bank_churners:
                bank_churners_columns.append(row[0])
            elif row[0] in student_mat:
                student_mat_columns.append(row[0])
            elif row[0] in food_coded:
                food_coded_columns.append(row[0])
    return [house_prices_columns, bank_churners_columns, student_mat_columns, food_coded_columns]


def create_test_y():
    house_prices = pd.read_csv(HOUSE_PRICES_PATH)
    bank_churners = pd.read_csv(BANK_CHURNERS_PATH)
    student_mat = pd.read_csv(STUDENT_MAT_PATH)
    food_coded = pd.read_csv(FOOD_CODED_PATH)
    house_prices_columns = []
    bank_churners_columns = []
    student_mat_columns = []
    food_coded_columns = []
    results = {}
    with open('draft_test_y.csv', 'r') as read_file:
        reader = csv.reader(read_file)
        for row in reader:
            if row[2] == 'remove':
                continue
            results[row[0]] = Result.numeric.name if row[1] == 'n' else Result.categorical.name
    with open('test_y.csv', 'w', newline='') as write_file:
        writer = csv.writer(write_file)
        for feature in results.keys():
            writer.writerow([feature, results[feature]])
    # return results, [house_prices[house_prices_columns], bank_churners[bank_churners_columns],
    #                  student_mat[student_mat_columns], food_coded[food_coded_columns]]


if __name__ == '__main__':
    res = get_numeric_columns(pd.read_csv(HOUSE_PRICES_PATH))
    # res = test_with_threshold(res, 20)
    # print(res)
    # a, b = create_test_y()
    a = get_relevant_columns_from_test_sets()
    pass
