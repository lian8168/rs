import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split


def get_data(train_path, test_path):
    data = pd.read_csv(train_path)
    print("查看data的info")
    print(data.info())

    sparse_features = {"userId": 577600, "newsId": 231676, "netId": 4, "phoneName": 34,
                       "OS": 3, "province1": 215, "city1": 608, "age": 5, "gender": 3,
                       "category1": 39, "subCategory1": 186, "date": 7}
    dense_features = ["flushNum", "imageNum"]
    sparse_fea_names = list(sparse_features.keys())
    sparse_fea_nums = list(sparse_features.values())

    train, valid = train_test_split(data, test_size=0.2, random_state=2020)
    del data

    train_dataset = Data.TensorDataset(torch.LongTensor(train[sparse_fea_names].values),
                                       torch.FloatTensor(train[dense_features].values),
                                       torch.FloatTensor(train["label"].values), )
    del train

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
    del train_dataset

    valid_dataset = Data.TensorDataset(torch.LongTensor(valid[sparse_fea_names].values),
                                       torch.FloatTensor(valid[dense_features].values),
                                       torch.FloatTensor(valid["label"].values), )
    del valid
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=4096, shuffle=False)
    del valid_dataset
    test_data = pd.read_csv(test_path)
    test_dataset = Data.TensorDataset(torch.LongTensor(test_data[sparse_fea_names].values),
                                       torch.FloatTensor(test_data[dense_features].values),
                                       torch.FloatTensor(test_data["label"].values), )
    del test_data
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=4096, shuffle=False)
    del test_dataset

    return train_loader, valid_loader, test_loader, sparse_fea_nums, dense_features


def get_criteo(train_path, test_path):
    data = pd.read_csv(train_path)
    print(data.info())
    sparse_fea_names = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_fea_nums = []
    for s in sparse_fea_names:
        sparse_fea_nums.append(data[s].max()+1)
    train, valid = train_test_split(data, test_size=0.2, random_state=2020)
    del data

    train_dataset = Data.TensorDataset(torch.LongTensor(train[sparse_fea_names].values),
                                       torch.FloatTensor(train[dense_features].values),
                                       torch.FloatTensor(train["label"].values), )
    del train

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
    del train_dataset

    valid_dataset = Data.TensorDataset(torch.LongTensor(valid[sparse_fea_names].values),
                                       torch.FloatTensor(valid[dense_features].values),
                                       torch.FloatTensor(valid["label"].values), )
    del valid
    valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=4096, shuffle=False)
    del valid_dataset
    test_data = pd.read_csv(test_path)
    test_dataset = Data.TensorDataset(torch.LongTensor(test_data[sparse_fea_names].values),
                                      torch.FloatTensor(test_data[dense_features].values),
                                      torch.FloatTensor(test_data["label"].values), )
    del test_data
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=4096, shuffle=False)
    del test_dataset

    return train_loader, valid_loader, test_loader, sparse_fea_nums, dense_features


if __name__ == '__main__':
    get_criteo("dataset/一点资讯/sample_test.csv", "")
    # train_loader, valid_loader, test_loader, sparse_fea_nums, dense_features = get_data()