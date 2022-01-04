import numpy as np
import pandas as pd
import torch
import utm
import dgl
from sklearn.preprocessing import StandardScaler
import pickle


class Stations:
    def __init__(self):
        pass


class OriginalData:
    def __init__(self):
        pass


def to_utm(x):
    """
    Transform lat/long coordinates in UTM
    :param x: [lat, long]
    :return: utm
    """
    return utm.from_latlon(x[0], x[1])


def create_graph_structure(edges_data):
    src = edges_data['src'].to_numpy()
    dst = edges_data['dst'].to_numpy()

    g = dgl.graph((src, dst))
    return g


def calculate_stations_distances(x, y):
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


class WeatherDataset:
    def __init__(self, name, max_distance=35):
        self.graph = None
        self.features_columns = ['lat', 'long', 'month', 'day', 'hour', 'forecast', 'gridpp']
        self.label_column = ['observation']
        self.stations_dict = None
        self.edges = None
        self.distances = None
        self.stations = None
        self.n_nearest = None
        self.dataframe = None
        self.name = name
        self.max_distance = max_distance

    def create(self, path=None, n_nearest: int = 5):
        self.dataframe = pd.read_csv(path)
        self.n_nearest = n_nearest
        self.stations = self.dataframe.copy()
        self.stations.drop_duplicates(subset=['station_id'], inplace=True)
        self.__calculate_utm()
        self.distances = calculate_stations_distances(np.array(self.stations.utm_x), np.array(self.stations.utm_y))
        self.edges = self.__calculate_graph_structure_dataframe()
        self.stations_dict = self.__create_stations_dict()

        self.graph = self.__create_graph()

    def __calculate_utm(self):
        self.stations['utm'] = self.stations[['lat', 'long']].apply(lambda x: to_utm(x), axis=1)
        self.stations['utm_x'] = self.stations['utm'].apply(lambda x: x[0])
        self.stations['utm_y'] = self.stations['utm'].apply(lambda x: x[1])

    def __calculate_graph_structure_dataframe(self):
        """
        Get a distance array and the n_nearest param and return a DataFrame with the graph structure with edges
        linking the n_nearest stations from each station

        """
        distances = self.distances
        n_nearest = self.n_nearest

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

        to_df = pd.DataFrame(to_df)

        return to_df[to_df.weight < self.max_distance]

    def __create_stations_dict(self):
        stations_dict = self.stations[['station_id']].reset_index(drop=True).to_dict()['station_id']
        new_dict = dict([(value, key) for key, value in stations_dict.items()])
        self.dataframe['node'] = self.dataframe.station_id.apply(lambda x: new_dict[x])

        return new_dict

    def __scale_data(self, features, labels):
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        self.scaler_x.fit(features.reshape(-1, features.shape[-1]))
        self.scaler_y.fit(labels.reshape(-1, labels.shape[-1]))

        return self.scaler_x.transform(features.reshape(-1, features.shape[-1])).reshape(
            features.shape), self.scaler_y.transform(labels.reshape(-1, labels.shape[-1])).reshape(
            labels.shape)

    def __calculate_features_and_labels(self):
        shape = np.array(self.stations.utm_x).shape[0]
        max_rows = self.dataframe.groupby(['node']).count()['station_id'].max()

        features = np.ndarray(shape=(shape, max_rows, len(self.features_columns)))
        labels = np.ndarray(shape=(shape, max_rows))

        for node in self.dataframe.node.unique():

            node_label = self.dataframe[self.dataframe.node == node][self.label_column].to_numpy()
            node_feature = self.dataframe[self.dataframe.node == node][self.features_columns].to_numpy()

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

    def __create_graph(self):
        features, labels = self.__calculate_features_and_labels()
        features, labels = self.__scale_data(features, labels)
        graph = create_graph_structure(self.edges)
        graph.ndata['x'] = torch.from_numpy(features)
        graph.ndata['y'] = torch.from_numpy(labels)

        n_nodes = graph.ndata['x'].shape[0]
        n_train = int(n_nodes * 0.8)
        n_val = int(n_nodes * 0.1)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask

        return graph

    def save(self):
        pickle.dump(self, open(f'{self.name}.pkl', 'wb'))


def read_weather_dataset(path):
    return pickle.load(open(path, 'rb'))
