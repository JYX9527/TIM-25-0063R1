import scipy
import numpy
import random
from torch.utils.data import Dataset

# 数据集划分
train_number =300
valid_number = 60
# test_number = 47


class Datasets_train(Dataset):
    def __init__(self):
        self.index = numpy.empty((1, train_number), dtype="float32")
        self.readdata()

    def readdata(self):
        ind_list = [i + 1 for i in range(train_number)]
        random.shuffle(ind_list)
        index_train = ind_list
        self.index = index_train
        return "training data has been loaded"

    def __len__(self):
        return train_number

    def __getitem__(self, index):
        return self.index[index]


class Datasets_valid(Dataset):
    def __init__(self):
        self.index = numpy.empty((1, valid_number), dtype="float32")
        self.readdata()

    def readdata(self):
        ind_list = [i + 1 for i in range(valid_number)]
        random.shuffle(ind_list)
        index_valid = ind_list
        self.index = index_valid
        return "validing data has been loaded"

    def __len__(self):
        return valid_number

    def __getitem__(self, index):
        return self.index[index]


# class Datasets_test(Dataset):
#     def __init__(self):
#         self.index = numpy.empty((1, test_number), dtype="float32")
#         self.readdata()
#
#     def readdata(self):
#         ind_list = [i + 1 for i in range(test_number)]
#         random.shuffle(ind_list)
#         index_test = ind_list[0: test_number]
#         self.index = index_test
#         return "testing data has been loaded"
#
#     def __len__(self):
#         return test_number
#
#     def __getitem__(self, index):
#         return self.index[index]


if __name__ == '__main__':
    train = Datasets_train()
    valid = Datasets_valid()
    print(1)
