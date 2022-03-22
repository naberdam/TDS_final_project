import csv
from test_with_our_model import create_test_list
from utils import load_datasets, get_relevant_columns_from_test_sets, check_accuracy
from main import read_json, convert_json_to_csv, shuffle
import pandas as pd
import tensorflow as tf

INTEGER = 1
CATEGORICAL = 0
TEST_Y_PATH = 'test_y.csv'
TEST_WITH_UNIQUE_VALUES_PATH = "./data/test_with_unique_values.h5"


def sigmoid_neuron(X, y, path, epoch, validation_split, batch_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=8, input_dim=1))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=epoch, validation_split=validation_split, batch_size=batch_size)
    model.save(path)
    print("model saved to " + str(path))
    return model


def train_model_with_unique_values():
    # new_data, field_names = read_json('./data/type_annotations.json')
    # convert_json_to_csv(new_data, field_names)
    train_list_x_df = pd.read_csv('./data/train_list_x.csv')
    train_list_x_df = train_list_x_df['unique_value_size']
    train_list_y_df = pd.read_csv('./data/train_list_y.csv')
    train_list_x, train_list_y = train_list_x_df.to_numpy(), train_list_y_df.to_numpy()
    train_list_x, train_list_y = shuffle(train_list_x, train_list_y)
    return sigmoid_neuron(X=train_list_x,
                          y=train_list_y,
                          path=TEST_WITH_UNIQUE_VALUES_PATH,
                          epoch=30,
                          validation_split=0.1,
                          batch_size=10)


def get_predictions(model, train_list_x, train_features_list):
    predictions = model.predict(train_list_x)
    results = {}
    for field_name, is_integer in zip(train_features_list, predictions):
        if is_integer > 0.5:
            results[field_name] = INTEGER
        else:
            results[field_name] = CATEGORICAL
    return results


if __name__ == '__main__':
    model = train_model_with_unique_values()
    test_datasets = load_datasets()
    test_relevant_columns = get_relevant_columns_from_test_sets(test_datasets)
    test_list, features_list = create_test_list(test_datasets, test_relevant_columns)
    test_accuracy = get_predictions(model, test_list["unique_value_size"].to_numpy(), features_list)
    print(check_accuracy(test_accuracy, TEST_Y_PATH))
