import datetime
import os
import time

import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# 打印模型参数
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


# 定义日志（data文件夹下，同级目录新建一个data文件夹）
def write_log(w, save_name='best'):
    file_name = 'checkpoint/' + datetime.date.today().strftime('%m%d') + "_{}.log".format(save_name)
    t0 = datetime.datetime.now().strftime('%H:%M:%S')
    info = "{} : {}".format(t0, w)
    print(info)
    with open(file_name, 'a') as f:
        f.write(info + '\n')


def train_and_eval(model, train_loader, valid_loader, epochs, device, optimizer, loss_fcn, scheduler, test_loader,
                   save_name='best'):
    best_auc = 0.0
    for _ in range(epochs):
        """训练部分"""
        model.train()
        print("Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        write_log('Epoch: {}'.format(_ + 1), save_name)
        train_loss_sum = 0.0
        start_time = time.time()
        for idx, x in enumerate(train_loader):
            cate_fea, nume_fea, label = x[0], x[1], x[2]
            cate_fea, nume_fea, label = cate_fea.to(device), nume_fea.to(device), label.float().to(device)
            pred = model(cate_fea, nume_fea).view(-1)
            train_label = label.cpu().numpy().tolist()
            train_pred = pred.data.cpu().numpy().tolist()
            train_auc = roc_auc_score(train_label, train_pred)
            loss = loss_fcn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.cpu().item()
            if (idx + 1) % 50 == 0 or (idx + 1) == len(train_loader):
                write_log("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Auc {:.4f} | Time {:.4f}".format(
                    _ + 1, idx + 1, len(train_loader), train_loss_sum / (idx + 1), train_auc, time.time() - start_time)
                    , save_name)
        scheduler.step()
        """推断部分"""
        test(model, device, valid_loader, best_auc, save_name, False)
        test(model, device, test_loader, best_auc, save_name, True)


def test(model, device, data_loader, best_auc, save_name, is_test=True):
    model.eval()
    with torch.no_grad():
        valid_labels, valid_preds = [], []
        for idx, x in tqdm(enumerate(data_loader)):
            cate_fea, nume_fea, label = x[0], x[1], x[2]
            cate_fea, nume_fea = cate_fea.to(device), nume_fea.to(device)
            pred = model(cate_fea, nume_fea).reshape(-1).data.cpu().numpy().tolist()
            valid_preds.extend(pred)
            valid_labels.extend(label.cpu().numpy().tolist())
    cur_auc = roc_auc_score(valid_labels, valid_preds)
    if is_test:
        write_log('Test AUC: %.6f\n' % cur_auc, save_name)
    else:
        if cur_auc > best_auc:
            best_auc = cur_auc
            torch.save(model.state_dict(), "checkpoint/" + save_name + "_best.pth")
        write_log('Current val AUC: %.6f, Best val AUC: %.6f\n' % (cur_auc, best_auc), save_name)
