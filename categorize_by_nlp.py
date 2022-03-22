import numpy as np
import pandas as pd
import csv

GLOVE_PATH = "I:\האחסון שלי\יאיר\לימודים\שנה ג סמסטר א\מדעי נתונים טבלאיים\glove.6B\glove.6B.300d.txt"
TRAIN_PATH = 'data/train_list_with_features.csv'


def get_features_names_and_label():
    features_names_and_labels = {
        'c': [],
        'n': []
    }
    train_set = pd.read_csv(TRAIN_PATH)
    for i in range(train_set.shape[0]):
        if train_set['is_integer'][i] == 0:
            features_names_and_labels['c'].append(train_set['field_name'][i])
        else:
            features_names_and_labels['n'].append(train_set['field_name'][i])
    return features_names_and_labels


def main():
    features_names_and_labels = get_features_names_and_label()
    train_data = {
        'c': features_names_and_labels['c'],
        'n': features_names_and_labels['n']
    }
    # test_data = {
    #     'c': features_names_and_labels['c'][30:],
    #     'n': features_names_and_labels['n'][30:]
    # }
    categories = {word: key for key, words in train_data.items() for word in words}

    # Load the whole embedding matrix
    embeddings_index = {}
    with open(GLOVE_PATH, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embed = np.array(values[1:], dtype=np.float32)
            embeddings_index[word] = embed
    print('Loaded %s word vectors.' % len(embeddings_index))
    # Embeddings for available words
    data_embeddings = {key: value for key, value in embeddings_index.items() if key in categories.keys()}

    def process(query):
        try:
            query_embed = embeddings_index[query]
        except Exception:
            return 'unknown query'
        scores = {}
        for word, embed in data_embeddings.items():
            category = categories[word]
            dist = query_embed.dot(embed)
            dist /= len(train_data[category])
            scores[category] = scores.get(category, 0) + dist
        return scores

    # for c_feature in test_data['c']:
    #     print(c_feature + ',' + str(process(c_feature)))
    #
    # print('\n\n================================\n\n')
    #
    # for n_feature in test_data['n']:
    #     print(n_feature + ',' + str(process(n_feature)))


if __name__ == '__main__':
    main()
