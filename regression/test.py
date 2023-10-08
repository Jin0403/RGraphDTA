import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from metrics import get_cindex, get_rm2
from dataset import *
from model import RGraphDTA
from utils import *
import csv

def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            pred_list.append(pred.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    epoch_cindex = get_cindex(label, pred)
    epoch_r2 = get_rm2(label, pred)
    epoch_loss = running_loss.get_average()
    running_loss.reset()

    return epoch_loss, epoch_cindex, epoch_r2

def main():

    data_root = "data"
    DATASET = 'davis'
    model_path = './save/20230818_220315_davis/model/epoch-1176, loss-0.0578, cindex-0.9591, val_loss-0.2001.pt'

    fpath = os.path.join(data_root, DATASET)

    test_set = GNNDataset(fpath, types='test')
    print("Number of test: ", len(test_set))
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=8)

    device = torch.device('cuda:0')
    model = RGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)

    criterion = nn.MSELoss()
    load_model_dict(model, model_path)
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []
    result_csv = open('result.csv', 'w', newline='')
    writer = csv.writer(result_csv)

    for data in test_loader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            pred_list.append(pred.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

        pred = np.concatenate(pred_list, axis=0)
        label = np.concatenate(label_list, axis=0)

        epoch_cindex = get_cindex(label, pred)
        epoch_r2 = get_rm2(label, pred)
        epoch_loss = running_loss.get_average()
        running_loss.reset()
        msg = "test_loss:%.4f, test_cindex:%.4f, test_r2:%.4f" % (epoch_loss, epoch_cindex, epoch_r2)
        print(msg)
        writer.writerow([epoch_loss, epoch_cindex, epoch_r2])
    result_csv.close()

if __name__ == "__main__":
    main()
