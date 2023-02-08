import torch
from data import get_data, get_criteo
import torch.nn as nn
from model import DeepFM, LogisticRegression, FM, FM2
import torch.optim as optim
from args import get_argparse


def init_for_train():
    args = get_argparse().parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    train_path = args.train_path
    test_path = args.test_path
    if args.dataset == 'criteo':
        train_loader, valid_loader, test_loader, sparse_features, dense_features = get_criteo(train_path, test_path)
    else:
        train_loader, valid_loader, test_loader, sparse_features, dense_features = get_data(train_path, test_path)
    if args.model_type == 'lr':
        model = LogisticRegression(sparse_features, nume_fea_size=len(dense_features))
        model.to(device)
    elif args.model_type == 'deepfm':
        model = DeepFM(sparse_features, nume_fea_size=len(dense_features))
        model.to(device)
    elif args.model_type == 'fm':
        model = FM2(sparse_features, nume_fea_size=len(dense_features))
        model.to(device)
    else:
        model = FM(sparse_features, nume_fea_size=len(dense_features))
        model.to(device)
    loss_fcn = nn.BCELoss()  # Loss函数
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    epochs = args.epochs
    name = args.model_type+'_'+args.dataset
    return model, train_loader, valid_loader, epochs, device, optimizer, loss_fcn, scheduler, test_loader, name
