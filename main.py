import json
import pandas as pd
import numpy as np
import math
import tensorflow as tf

INTEGER = 1
CATEGORIAL = 0


def read_json(name):
    with open(name) as json_file:
        data = json.load(json_file)
        new_data = {}
        field_names = []
        for file_name in data:
            fields_new_data = dict((f'{file_name}_{key}', data[file_name][key]) for key in data[file_name] \
                                   if (data[file_name][key] == 'integer' or isinstance(data[file_name][key], list)))
            if fields_new_data == {}: continue
            new_data[file_name] = fields_new_data
            field_names.extend([field for field in new_data[file_name]])
            fields_new_data[file_name] = dict((f'{file_name}_{key}', data[file_name][key]) for key in data[file_name] \
                                              if (data[file_name][key] == 'integer' \
                                                  or (isinstance(data[file_name][key], list) and isinstance(
                        data[file_name][key][
                            0], int))))
            if fields_new_data == {}: continue
        return new_data, field_names


# function that shuffles the train_x with train_y
def shuffle(train_x, train_y):
    permutation = np.random.permutation(len(train_x))
    return train_x[permutation], train_y[permutation]


def convert_json_to_csv(data: dict, field_list: list, add_features_to_csv=False):
    df = pd.DataFrame(columns=field_list)
    train_list_x_df = pd.DataFrame(columns=['unique_per_num_values', 'std_value', "unique_value_size", "min_value",
                                            "max_value"])
    train_list_y_df = pd.DataFrame(columns=['result'])
    unique_per_num_values_list = []
    unique_value_size_list = []
    std_value_list = []
    y_list = []
    min_value_list = []
    max_value_list = []
    train_features_list = []
    for file_name in data:
        if file_name in ["Midwest_Survey_nominal", "usp05"]:
            continue
        df_temp = pd.read_csv(f'data/train_raw_datasets/{file_name}.csv', keep_default_na=False, low_memory=False)
        df_temp = df_temp.replace({'\'': ''}, regex=True)
        for field_name in df_temp:
            if f'{file_name}_{field_name}' in field_list:
                if f'{file_name}_{field_name}' in ["anneal_bc", "zoo_type"]:
                    continue
                if isinstance(data[file_name][f'{file_name}_{field_name}'], list):
                    df_temp[field_name] = df_temp[field_name].astype(str)
                    df_temp[field_name] = df_temp[field_name].str.lower()
                    df_temp[field_name] = df_temp[field_name].replace(
                        [x.lower() for x in data[file_name][f'{file_name}_{field_name}']],
                        list(range(1, len(data[file_name][f'{file_name}_{field_name}']) + 1)))
                    y_list.append(CATEGORIAL)
                else:
                    y_list.append(INTEGER)
                df_temp[field_name] = pd.to_numeric(df_temp[field_name], errors='coerce')
                if any(math.isnan(unique_value) for unique_value in df_temp[field_name].unique()):
                    df_temp[field_name] = df_temp[field_name].dropna().reset_index(drop=True)
                    if df_temp[field_name].empty:
                        df_temp = df_temp.drop(field_name, axis=1)
                        del y_list[-1]
                        continue
                df[f'{file_name}_{field_name}'] = df_temp[field_name]
                unique_per_num_values = df_temp[field_name].unique().size / df_temp[field_name].count()
                unique_value_size = df_temp[field_name].unique().size
                min_value = df_temp[field_name].min()
                max_value = df_temp[field_name].max()
                df_temp[field_name] = (df_temp[field_name] - df_temp[field_name].min()) / \
                                      (df_temp[field_name].max() - df_temp[field_name].min())
                std_value = df_temp[field_name].std()
                if math.isnan(std_value):
                    df_temp = df_temp.drop(field_name, axis=1)
                    del y_list[-1]
                    continue
                min_value_list.append(min_value)
                max_value_list.append(max_value)
                # train_list_x.append([unique_per_num_values, unique_value_size, std_value])
                unique_per_num_values_list.append(unique_per_num_values)
                unique_value_size_list.append(unique_value_size)
                std_value_list.append(std_value)
                train_features_list.append(field_name)
    df.to_csv('./data/data.csv', index=False)
    train_list_y_df["result"] = y_list
    train_list_x_df["unique_per_num_values"] = unique_per_num_values_list
    train_list_x_df["unique_value_size"] = unique_value_size_list
    train_list_x_df["std_value"] = std_value_list
    train_list_x_df["min_value"] = min_value_list
    train_list_x_df["max_value"] = max_value_list
    train_list_x_df.to_csv('./data/train_list_x.csv', index=False)
    train_list_y_df.to_csv('./data/train_list_y.csv', index=False)
    if add_features_to_csv:
        train_list_x_df['field_name'] = train_features_list
        train_list_x_df['is_integer'] = y_list
        train_list_x_df.to_csv('./data/train_list_with_features.csv', index=False)


def get_train():
    train_list_x_df = pd.read_csv('./data/train_list_x.csv')
    train_list_y_df = pd.read_csv('./data/train_list_y.csv')
    return train_list_x_df.to_numpy(), train_list_y_df.to_numpy()


def sigmoid_neuron(X, y, path, epoch, validation_split, batch_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=X.shape[1], input_dim=X.shape[1]))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=epoch, validation_split=validation_split, batch_size=batch_size)
    model.save(path)
    print("model saved to " + str(path))
    return model


def predict_on_model(model_path):
    pass


if __name__ == '__main__':
    # new_data, field_names = read_json('./data/type_annotations.json')
    # convert_json_to_csv(new_data, field_names, add_features_to_csv=True)
    train_list_x, train_list_y = get_train()
    train_list_x, train_list_y = shuffle(train_list_x, train_list_y)
    data_type_identifier_model = sigmoid_neuron(X=train_list_x,
                                                y=train_list_y,
                                                path="./data/data_type_identifier.h5",
                                                epoch=30,
                                                validation_split=0.1,
                                                batch_size=10)
