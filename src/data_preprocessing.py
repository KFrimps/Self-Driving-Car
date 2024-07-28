import pandas as pd
import ntpath

def load_data(datadir):
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    data = pd.read_csv(f'{datadir}/driving_log.csv', names=columns)
    pd.set_option('display.max_colwidth', 1)
    return data

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail

def preprocess_data(data):
    data['center'] = data['center'].apply(path_leaf)
    data['left'] = data['left'].apply(path_leaf)
    data['right'] = data['right'].apply(path_leaf)
    return data
