import numpy
import torch
import unet
from scipy import io
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = unet.UNet()
net.load_state_dict(torch.load(r"../model/model.plt"))
net = net.to(device)
net.eval()
n=1
while n<41:
    img = numpy.empty((1, 1, 480, 640), dtype="float32")
    img_url = f"../dataset-MD/test/input/I({n}).mat"
    mat_train = h5py.File(img_url,'r')
    img1 = mat_train['a'][()].T
    img[0, 0, :, :] = img1[:]
    img = torch.tensor(img)
    img = img.to(device)
    out1 = net(img)
    M = out1[0, 0, :, :]
    D = out1[0, 1, :, :]
    M = ((M.cpu()).detach()).numpy()
    D = ((D.cpu()).detach()).numpy()
    io.savemat(f"../dataset-MD/test/output/M({n}).mat",{r'M': M})
    io.savemat(f"../dataset-MD/test/output/D({n}).mat",{r'D': D})
    print(n)
    n=n+1