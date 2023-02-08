from init import init_for_train
from train import train_and_eval

if __name__ == '__main__':
    model, train_loader, valid_loader, epochs, device, optimizer, loss_fcn, scheduler, test_loader, name\
        = init_for_train()
    train_and_eval(model, train_loader, valid_loader, epochs, device, optimizer, loss_fcn, scheduler, test_loader, name)
