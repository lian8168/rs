import torch
import numpy as np
import math
from matplotlib import pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"


def read_log(log_name):
    train_losses = []
    train_aucs = []
    val_aucs = []
    test_aucs = []
    with open(log_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.find('Loss') != -1:
                l_s = line.split(' ')
                train_losses.append(float(l_s[11]))
                train_aucs.append(float(l_s[14]))
            if line.find('val') != -1:
                l_s = line.split(' ')
                val_aucs.append(float(l_s[5].split(',')[0]))
            if line.find('Test') != -1:
                l_s = line.split(' ')
                test_aucs.append(float(l_s[4].split('\n')[0]))
    f.close()
    return train_losses, train_aucs, val_aucs, test_aucs


def draw_curve(train_log, name, data_type, model_type):
    # train_loss曲线
    x = np.linspace(0, len(train_log), len(train_log))
    plt.plot(x, train_log, label=name, linewidth=1.5)
    plt.xlabel("step of each batch*50")
    plt.ylabel(name.split('_')[-1])
    plt.title("train loss of " + model_type + " on " + data_type)  # 标题
    plt.savefig("output/train loss of " + model_type + " on " + data_type + '.jpg')
    plt.legend()
    plt.show()

    # plt.clf()


def draw_test_loss(train_aucs, val_aucs, test_aucs, data_type, model_type):
    x = np.linspace(0, len(train_aucs), len(train_aucs))
    plt.plot(x, train_aucs, marker='o', color='r', label=u'train_auc')
    plt.plot(x, val_aucs, marker='*', color='g', label=u'val_auc')
    plt.plot(x, test_aucs, marker='+', color='b', label=u'test_auc')
    plt.legend()  # 让图例生效
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"num of batches")  # X轴标签
    plt.ylabel("auc")  # Y轴标签
    plt.title("train, val and test auc of " + model_type + " on " + data_type)  # 标题
    plt.savefig("output/train, val and test auc of " + model_type + " on " + data_type + '.jpg')
    plt.show()



def draw(data_type, model_type):
    path = "checkpoint/1207_" + model_type + "_" + data_type + ".log"
    if data_type == 'yidianzixun':
        train_losses, train_aucs, val_aucs, test_aucs = read_log(path)
        print("一点资讯数据集 " + model_type + " best val auc:" + str(max(val_aucs)))
        print("一点资讯数据集 " + model_type + " best test auc:" + str(max(test_aucs)))
        batch = [0] * 255
        batch_auc = []
        for i in range(0, 2550):
            if i % 255 == 0 and i != 0:
                batch_auc.append(sum(batch) / 255)
            batch[i % 255] = train_aucs[i]
        batch_auc.append(sum(batch) / 255)
        draw_curve(train_losses, "train_loss", data_type, model_type)
        draw_test_loss(batch_auc, val_aucs, test_aucs, data_type, model_type)
    else:
        train_losses, train_aucs, val_aucs, test_aucs = read_log(path)
        print("criteo数据集 " + model_type + " best val auc:" + str(max(val_aucs)))
        print("criteo数据集  " + model_type + " best test auc:" + str(max(test_aucs)))
        batch = [0] * 7
        batch_auc = []
        for i in range(0, 350):
            if i % 7 == 0 and i != 0:
                batch_auc.append(sum(batch) / 7)
            batch[i % 7] = train_aucs[i]
        batch_auc.append(sum(batch) / 7)
        draw_curve(train_losses, "train_loss", data_type, model_type)
        draw_test_loss(batch_auc, val_aucs, test_aucs, data_type, model_type)


if __name__ == '__main__':
    draw("yidianzixun", "lr")
    draw("yidianzixun", "fm")
    draw("yidianzixun", "deepfm")
    draw("criteo", "lr")
    draw("criteo", "fm")
    draw("criteo", "deepfm")
    # train_losses, train_aucs, val_aucs, test_aucs = read_log("checkpoint/1207_lr_yidianzixun.log")
    # batch = [0]*255
    # batch_auc = []
    # for i in range(0, 2550):
    #     if i % 255 == 0 and i != 0:
    #         batch_auc.append(sum(batch)/255)
    #     batch[i % 255] = train_aucs[i]
    # batch_auc.append(sum(batch) / 255)
    # draw_curve(train_losses, "train_loss")
    # draw_curve(batch_auc, "train_aucs")
    # draw_curve(val_aucs, "val_aucs")
    # draw_curve(test_aucs, "test_aucs")
