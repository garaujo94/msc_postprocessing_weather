import numpy as np
import pandas as pd
import utm
import dgl
import networkx as nx
import torch


def to_utm(x):
    """
    Transform lat/long coordinates in UTM
    :param x: [lat, long]
    :return: utm
    """

    return utm.from_latlon(x[0], x[1])


def calculate_stations_distance(x: np.array, y: np.array) -> np.array:
    """
    Get arrays of coordinates and calculate the distance between each other
    :param x: np.array
    :param y: np.array
    :return:
    """
    distances = np.ndarray(shape=(x.shape[0] - 1, x.shape[0]))

    for i in range(x.shape[0] - 1):
        next_x = np.concatenate((x[i + 1:], x[:i + 1]), axis=None)
        next_y = np.concatenate((y[i + 1:], y[:i + 1]), axis=None)

        diff_x = x - next_x
        diff_y = y - next_y

        diff_x_square = diff_x ** 2
        diff_y_square = diff_y ** 2

        sum_square = diff_x_square + diff_y_square

        distances[i] = np.sqrt(sum_square)

    distances = distances.transpose()

    return distances


def calculate_graph_structure_dataframe(distances: np.array, n_nearest: int = 5) -> pd.DataFrame:
    """
    Get a distance array and the n_nearest param and return a DataFrame with the graph structure with edges
    linking the n_nearest stations from each station
    :param n_nearest: int
    :param distances: np.array
    :return: pd.DataFrame
    """
    i = 0
    src = []
    dst = []
    weight = []
    for r in distances.argsort()[:, :n_nearest]:
        for value in r:
            src.append(i)
            dst.append(value)
            weight.append(distances[i, value] / 1000)

        i += 1

    to_df = {
        'src': src,
        'dst': dst,
        'weight': weight
    }

    return pd.DataFrame(to_df)


def calculate_features_and_labels(df: pd.DataFrame,
                                  max_rows=1400,
                                  shape=None,
                                  features_columns=None,
                                  label_column=None):
    features = np.ndarray(shape=(shape, max_rows, len(features_columns)))
    labels = np.ndarray(shape=(shape, max_rows))

    for node in df.node.unique():

        node_label = df[df.node == node][label_column].to_numpy()
        node_feature = df[df.node == node][features_columns].to_numpy()

        to_complete = int(max_rows - node_feature.shape[0])

        if to_complete > node_feature.shape[0]:
            time_to_repeat = np.floor(to_complete / node_feature.shape[0])

            node_to_add = np.repeat(node_feature, time_to_repeat, axis=0)
            label_to_add = np.repeat(node_label, time_to_repeat, axis=0)

            left_to_add = to_complete - node_to_add.shape[0]

            node_to_add = np.concatenate((node_to_add, node_feature[:left_to_add]))
            label_to_add = np.concatenate((label_to_add, node_label[:left_to_add]))

            features[node] = np.concatenate((node_feature, node_to_add), axis=0)
            labels[node] = np.concatenate((node_label, label_to_add), axis=0).reshape(max_rows)
            continue

        features[node] = np.concatenate((node_feature, node_feature[:to_complete]), axis=0)
        labels[node] = np.concatenate((node_label, node_label[:to_complete]), axis=0).reshape(max_rows)

    return features, labels


def create_graph_structure(edges_data):
    src = edges_data['src'].to_numpy()
    dst = edges_data['dst'].to_numpy()

    g = dgl.graph((src, dst))
    return g


def print_graph(g):
    nx_g = g.to_networkx().to_undirected()
    # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
    pos = nx.kamada_kawai_layout(nx_g)
    nx.draw(nx_g, pos, with_labels=True, node_color=[[.7, .7, .7]])
