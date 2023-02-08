import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='deepfm', type=str,
                        help="模型类型")
    parser.add_argument("--dataset", default='criteo', type=str,
                        help="数据集类型")
    parser.add_argument("--train_path", default='dataset/dac/train_sample.csv', type=str,
                        help="训练集路径")
    parser.add_argument("--test_path", default='dataset/dac/test_sample.csv', type=str,
                        help="测试集路径")
    parser.add_argument("--epochs", default=50, type=int,
                        help="训练轮次")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="批量大小")
    parser.add_argument("--hidden_size", default=32, type=int,
                        help="隐层单元数")
    parser.add_argument("--lr", default=0.0001, type=float,
                        help="学习率")
    return parser