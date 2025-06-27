import numpy
import torch
import unet
from scipy import io
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = unet.UNet()
net.load_state_dict(torch.load(r"../model/model.plt"))
net = net.to(device)
n=1
net.eval()

while n<301:
    img = numpy.empty((1, 1, 480, 640), dtype="float32")
    img_url = f"../dataset-H/train/input/I({n}).mat"
    mat_train = h5py.File(img_url,'r')
    img1 = mat_train['a'][()].T
    img[0, 0, :, :] = img1[:]
    img = torch.tensor(img)
    img = img.to(device)
    out1 = net(img)
    H = out1[0, 0, :, :]
    H= ((H.cpu()).detach()).numpy()
    io.savemat(f"../dataset-H/train/output/H({n}).mat",{r'H': H})
    print(n)
    n=n+1

n=1
while n<61:
    img = numpy.empty((1, 1, 480, 640), dtype="float32")
    img_url = f"../dataset-H/valid/input/I({n}).mat"
    mat_train = h5py.File(img_url,'r')
    img1 = mat_train['a'][()].T
    img[0, 0, :, :] = img1[:]
    img = torch.tensor(img)
    img = img.to(device)
    out1 = net(img)
    H = out1[0, 0, :, :]
    H= ((H.cpu()).detach()).numpy()
    io.savemat(f"../dataset-H/valid/output/H({n}).mat",{r'H': H})
    print(n)
    n=n+1

n = 1
while n<41:
    img = numpy.empty((1, 1, 480, 640), dtype="float32")
    img_url = f"../dataset-H/test/input/I({n}).mat"
    mat_train = h5py.File(img_url,'r')
    img1 = mat_train['a'][()].T
    img[0, 0, :, :] = img1[:]
    img = torch.tensor(img)
    img = img.to(device)
    out1 = net(img)
    H = out1[0, 0, :, :]
    H= ((H.cpu()).detach()).numpy()
    io.savemat(f"../dataset-H/test/output/H({n}).mat",{r'H': H})
    print(n)
    n=n+1