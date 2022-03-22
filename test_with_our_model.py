import csv
from utils import load_datasets, get_relevant_columns_from_test_sets, check_accuracy
import math
import pandas as pd
import tensorflow as tf

INTEGER = 1
CATEGORICAL = 0
TEST_Y_PATH = 'test_y.csv'

KEY_TO_FILE_NAME = {'house_prices': 'house_prices',
                    'bank_churners': 'BankChurners',
                    'student_mat': 'student-mat',
                    'food_coded': 'food_coded'}


def create_test_list(data: dict, field_list: list):
    df = pd.DataFrame(columns=field_list)
    train_list_x_df = pd.DataFrame(columns=['unique_per_num_values', 'std_value', "unique_value_size", "min_value",
                                            "max_value"])
    unique_per_num_values_list = []
    unique_value_size_list = []
    std_value_list = []
    min_value_list = []
    max_value_list = []
    train_features_list = []
    for file_name in data:
        df_temp = pd.read_csv(f'data/test_raw_datasets/{KEY_TO_FILE_NAME[file_name]}.csv', keep_default_na=False,
                              low_memory=False)
        df_temp = df_temp.replace({'\'': ''}, regex=True)
        for field_name in df_temp:
            if field_name in field_list[file_name]:
                df_temp[field_name] = pd.to_numeric(df_temp[field_name], errors='coerce')
                if any(math.isnan(unique_value) for unique_value in df_temp[field_name].unique()):
                    df_temp[field_name] = df_temp[field_name].dropna().reset_index(drop=True)
                    if df_temp[field_name].empty:
                        df_temp = df_temp.drop(field_name, axis=1)
                        continue
                print(f'{file_name}_{field_name}    {df_temp[field_name].unique()}')
                unique_per_num_values = df_temp[field_name].unique().size / df_temp[field_name].count()
                unique_value_size = df_temp[field_name].unique().size
                min_value = df_temp[field_name].min()
                max_value = df_temp[field_name].max()
                df_temp[field_name] = (df_temp[field_name] - df_temp[field_name].min()) / \
                                      (df_temp[field_name].max() - df_temp[field_name].min())
                std_value = df_temp[field_name].std()
                if math.isnan(std_value):
                    df_temp = df_temp.drop(field_name, axis=1)
                    # del y_list[-1]
                    continue
                min_value_list.append(min_value)
                max_value_list.append(max_value)
                unique_per_num_values_list.append(unique_per_num_values)
                unique_value_size_list.append(unique_value_size)
                std_value_list.append(std_value)
                train_features_list.append(field_name)
    df.to_csv('./data/test_with_our_model.csv', index=False)
    train_list_x_df["unique_per_num_values"] = unique_per_num_values_list
    train_list_x_df["unique_value_size"] = unique_value_size_list
    train_list_x_df["std_value"] = std_value_list
    train_list_x_df["min_value"] = min_value_list
    train_list_x_df["max_value"] = max_value_list
    return train_list_x_df, train_features_list


def get_predictions(train_list_x, train_features_list):
    reconstructed_model = tf.keras.models.load_model("./data/data_type_identifier.h5")
    predictions = reconstructed_model.predict(train_list_x)
    results = {}
    for field_name, is_integer in zip(train_features_list, predictions):
        if is_integer > 0.5:
            results[field_name] = INTEGER
        else:
            results[field_name] = CATEGORICAL
    return results


if __name__ == '__main__':
    test_datasets = load_datasets()
    test_relevant_columns = get_relevant_columns_from_test_sets(test_datasets)
    test_list, features_list = create_test_list(test_datasets, test_relevant_columns)
    test_accuracy = get_predictions(test_list.to_numpy(), features_list)
    print(check_accuracy(test_accuracy, TEST_Y_PATH))
