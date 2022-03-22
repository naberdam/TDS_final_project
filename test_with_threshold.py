import csv
from utils import load_datasets, get_relevant_columns_from_test_sets

INTEGER = 1
CATEGORICAL = 0
TEST_Y_PATH = 'test_y.csv'


def test_with_threshold_on_single_dataset(columns, threshold):
    results = {}
    for col in columns:
        results[col] = INTEGER if columns[col].nunique() > threshold else CATEGORICAL
    return results


def test_with_threshold(datasets, relevant_columns, threshold):
    results = {}
    for dataset_name in datasets.keys():
        results.update(test_with_threshold_on_single_dataset(datasets[dataset_name][relevant_columns[dataset_name]],
                                                             threshold))
    return results


def check_accuracy(results, test_y_path):
    with open(test_y_path, 'r') as readfile:
        accuracy = 0
        reader = csv.reader(readfile)
        row_count = 0
        for row in reader:
            row_count += 1
            if results[row[0]] == int(row[1]):
                accuracy += 1
        accuracy /= row_count
    return accuracy


if __name__ == '__main__':
    test_datasets = load_datasets()
    test_relevant_columns = get_relevant_columns_from_test_sets(test_datasets)
    test_accuracy = test_with_threshold(test_datasets, test_relevant_columns, 20)
    print(check_accuracy(test_accuracy, TEST_Y_PATH))
