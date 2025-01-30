import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_test_dataLoader(test_dataset, batch_size):

  #Define dataloader
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return test_dataloader

def get_data_dir(dataset, type='Mnist'):
    if 'EMnist' in dataset or type=='EMnist':
        path_prefix=os.path.join('/', dataset)
        train_data_dir=os.path.join(path_prefix, 'train')
        test_data_dir=os.path.join(path_prefix, 'test')

    elif type == 'Mnist':
        path_prefix=os.path.join('/', dataset)
        train_data_dir=os.path.join(path_prefix, 'train')
        test_data_dir=os.path.join(path_prefix, 'test')

    elif 'celeb' in dataset.lower():
        dataset_ = dataset.lower().replace('user', '').replace('agg','').split('-')
        user, agg_user = dataset_[1], dataset_[2]
        path_prefix = os.path.join('..\\data', 'CelebA', 'user{}-agg{}'.format(user,agg_user))
        train_data_dir=os.path.join(path_prefix, 'train')
        test_data_dir=os.path.join(path_prefix, 'test')

    else:
        raise ValueError("Dataset not recognized.")
    return train_data_dir, test_data_dir

def read_user_data(index, data, dataset='', count_labels=False):
    #data contains: clients, groups, train_data, test_data, proxy_data(optional)
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    X_train, y_train = convert_data(train_data['x'], train_data['y'], dataset=dataset)
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    X_test, y_test = convert_data(test_data['x'], test_data['y'], dataset=dataset)
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    if count_labels:
        label_info = {}
        unique_y, counts=torch.unique(y_train, return_counts=True)
        unique_y=unique_y.detach().numpy()
        counts=counts.detach().numpy()
        label_info['labels']=unique_y
        label_info['counts']=counts
        return id, train_data, label_info
    return id, train_data, test_data

def read_data(dataset, dataType):
    train_data_dir, test_data_dir = get_data_dir(dataset, dataType)
    clients = []
    groups = []
    train_data = {}
    test_data = {}
    proxy_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json') or f.endswith(".pt")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        if file_path.endswith("json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        elif file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))

        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json') or f.endswith(".pt")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        if file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        elif file_path.endswith(".json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))
        test_data.update(cdata['user_data'])

    return clients, groups, train_data, test_data, proxy_data

def convert_data(X, y, dataset=''):
    if not isinstance(X, torch.Tensor):
        if 'celeb' in dataset.lower():
            X=torch.Tensor(X).type(torch.float32).permute(0, 3, 1, 2)
            y=torch.Tensor(y).type(torch.int64)

        else:
            X=torch.Tensor(X).type(torch.float32)
            y=torch.Tensor(y).type(torch.int64)
    return X, y

def get_dataLoader(config, total_users):
    data = read_data(config.datasetPath, dataType=config.DATASETTYPE)
    users = []

    for i in range(total_users):
        id, train_data, test_data  = read_user_data(i, data, config.datasetPath)
        user = USER(id, train_data, test_data)
        user.clean_data()
        users.append(user)
    return users