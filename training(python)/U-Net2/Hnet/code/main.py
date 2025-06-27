"""
训练器模块
"""
import os
import numpy
import scipy
from scipy import io
import unet
import torch
import dataset
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
import h5py
# 图像尺寸
img_height = 480
img_width = 640
# 网络输入通道数
input_channel = 1
output_channel = 1
# batchsize
batch_size = 2


# 网络参数初始化
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


# batch数据读取
def readdata(index, path):
    input = numpy.empty((batch_size, input_channel, img_height, img_width),
                        dtype="float32")
    output = numpy.empty((batch_size, output_channel, img_height, img_width),
                         dtype="float32")
    index = index.numpy()
    # 读取输入
    img_add = path + 'input/I(' + str(index[0]) + ').mat'
    mat_train = h5py.File(img_add,'r')
    img = mat_train['a'][()].T
    input[0, 0, :, :] = img[:]

    img_add = path + 'input/I(' + str(index[1]) + ').mat'
    mat_train = h5py.File(img_add,'r')
    img = mat_train['a'][()].T
    input[1, 0, :, :] = img[:]
    # 读取输出
    img_add = path + 'gt/H(' + str(index[0]) + ').mat'
    mat_train = h5py.File(img_add,'r')
    img = mat_train['H'][()].T
    output[0, 0, :, :] = img[:]

    img_add = path + 'gt/H(' + str(index[1]) + ').mat'
    mat_train = h5py.File(img_add,'r')
    img = mat_train['H'][()].T
    output[1, 0, :, :] = img[:]
    # 转tensor
    input = torch.from_numpy(input)
    output = torch.from_numpy(output)
    return input, output


# 训练器
class Trainer:

    def __init__(self, model):
        self.model = model
        # 使用的设备
        # self.device =torch.device("cpu")
        self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 网络

        self.net = unet.UNet()
        initialize_weights(self.net)  # 初始化权重he_normal
        self.net = self.net.to(self.device)

        self.opt = torch.optim.Adam(self.net.parameters())

        self.MSE = nn.MSELoss(reduction='mean')  # mse
        self.MAE = nn.L1Loss(reduction='mean')  # mae
        num_workers = 10
        self.train_loader = DataLoader(dataset.Datasets_train(), batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers)
        self.valid_loader = DataLoader(dataset.Datasets_valid(), batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers)
        # 判断是否存在模型
        if os.path.exists(self.model):
            self.net.load_state_dict(torch.load(model))
            print(f"Loaded{model}!")
        else:
            print("No Param!")

    def train(self, stop_value, path_train, path_valid):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'min', factor=0.5, patience=20, verbose=True,
                                                               threshold=0.001)
        epoch = 1
        mae_train = []
        mae_valid = []
        mse_train = []
        mse_valid = []
        lrs = []
        best_maes = []
        epochs = []
        best_mae = 10000000
        while True:
            # 训练
            self.net.train()
            for index in tqdm(self.train_loader, desc=f"Epoch {epoch}/{stop_value}",
                              ascii=True, total=len(self.train_loader)):
                input, output = readdata(index, path_train)
                input, label = input.to(self.device), output.to(self.device)
                # 正常梯度下降
                out = self.net(input)
                loss = self.MSE(out, label)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.net.parameters(), max_norm=10, norm_type=2)
                self.opt.step()
            # 验证+测试
            self.net.eval()
            valid_mae = 0
            train_mae = 0
            valid_mse = 0
            train_mse = 0
            with torch.no_grad():
                for index in tqdm(self.valid_loader, desc=f"Epoch {epoch}/{stop_value}",
                                  ascii=True, total=len(self.valid_loader)):
                    input, output = readdata(index, path_valid)
                    input, label1 = input.to(self.device), output.to(self.device)
                    out1 = self.net(input)
                    valid_mse += self.MSE(out1, label1)
                    valid_mae += self.MAE(out1, label1)

                for index in tqdm(self.train_loader, desc=f"Epoch {epoch}/{stop_value}",
                                  ascii=True, total=len(self.train_loader)):
                    input, output = readdata(index, path_train)
                    input, label2 = input.to(self.device), output.to(self.device)
                    out2 = self.net(input)
                    train_mse += self.MSE(out2, label2)
                    train_mae += self.MAE(out2, label2)

            # mse
            train_mse /= len(self.train_loader)
            valid_mse /= len(self.valid_loader)
            mse = train_mse.cpu()
            mse_train.append(mse.detach().numpy())
            mse = valid_mse.cpu()
            mse_valid.append(mse.detach().numpy())
            print(f"\nEpoch: {epoch}/{stop_value}, train_mse: {train_mse}")
            print(f"\nEpoch: {epoch}/{stop_value}, valid_mse: {valid_mse}")
            # mae
            train_mae /= len(self.train_loader)
            valid_mae /= len(self.valid_loader)
            mae = train_mae.cpu()
            mae_train.append(mae.detach().numpy())
            mae = valid_mae.cpu()
            mae_valid.append(mae.detach().numpy())
            print(f"\nEpoch: {epoch}/{stop_value}, train_mae: {train_mae}")
            print(f"\nEpoch: {epoch}/{stop_value}, valid_mae: {valid_mae}")
            # lr
            lrs.append(self.opt.state_dict()['param_groups'][0]['lr'])
            # 退火算法步进
            scheduler.step(train_mse)
            # 保存最优mae
            if mae < best_mae:
                best_mae = mae
                best_maes.append(best_mae)  # 最优mae
                epochs.append(epoch)  # 最优epoch
                torch.save(self.net.state_dict(), self.model)

            io.savemat('../record/mae_train.mat', {'mae_train': mae_train})
            io.savemat('../record/mae_valid.mat', {'mae_valid': mae_valid})
            io.savemat('../record/mse_train.mat', {'mse_train': mse_train})
            io.savemat('../record/mse_valid.mat', {'mse_valid': mse_valid})
            io.savemat('../record/best_maes.mat', {'best_maes': best_maes})
            io.savemat('../record/epochs.mat', {'epochs': epochs})
            io.savemat('../record/lrs.mat', {'lrs': lrs})
            epoch += 1
            if epoch > stop_value:
                break


if __name__ == '__main__':
    start_time = time.time()  # 开始计时

    t = Trainer(r'../model/model.plt')
    t.train(300, r"../dataset-H/train/", r"../dataset-H/valid/")

    end_time = time.time()#停止计时
    execution_time = (end_time - start_time)/3600#单位转换
    io.savemat('../record/execution_time.mat', {'execution_time': execution_time})
    print("程序执行时间：", execution_time, "h")
    os.system('shutdown')
