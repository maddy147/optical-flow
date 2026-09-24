# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import cv2
from torchvision.utils import flow_to_image
from torchvision.io import decode_image
from torchvision.io import write_png
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
plt.rcParams["savefig.bbox"] = "tight"
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from FlowNetS import FlowNetS

model = FlowNetS()
# Checkpoint URL:- https://github.com/ClementPinard/FlowNetPytorch/blob/master/weights/flownets_bn_EPE2.459.pth"
model_dict = torch.load("./flownets_bn_EPE2.459.pth", map_location = device )
model.load_state_dict(model_dict["state_dict"])
model = model.eval()
model = model.to(device)

def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = TF.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()

tform = T.Compose([
        T.Normalize(0,255),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

inv_tform = T.Compose([
    T.Normalize(mean = (-1,-1,-1), std = (2,2,2)),
])

def pre_process(img):
    img = TF.resize(img, size=[520, 960], antialias=False)
    img = img.float()
    return tform(img)

from torchvision.models.optical_flow import raft_large

raft_model = raft_large(pretrained=True, progress=False).to(device)
raft_model = raft_model.eval()

class CustomDset(Dataset):
    def __init__(self,path,frame_nos,train = False,transform = None, max_offset = 5):
        if(train):
            self.path = path + "train/"
        else:
            self.path = path + "test/"

        self.frame_fldrs = os.listdir(self.path)
        self.frame_fldrs = sorted(self.frame_fldrs, key = lambda x: int(x[4:]))
        self.max_offset = max_offset
        self.transform = transform
        self.frame_nos = frame_nos

    def __len__(self):
        return len(self.frame_fldrs)

    def __getitem__(self,idx):
        frame_no = self.frame_nos[idx]
        img1 = decode_image(self.path + self.frame_fldrs[idx] + "/" + f"Frame_{frame_no}.png")
        img2 = decode_image(self.path + self.frame_fldrs[idx] + "/" + f"Frame_{frame_no + self.max_offset}.png")

        flow = torch.load(f"{self.path}" + f"{self.frame_fldrs[idx]}" + "/flow_" + f"{self.frame_nos[idx]}_{self.frame_nos[idx] + self.max_offset}.pth",weights_only=True).squeeze(0)
        # To match the flow shape of flownet
        flow = F.avg_pool2d(flow,(4,4))

        if(self.transform):
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1,img2),flow

chkpt_path = "./checkpoints/"
frames_path = "./VideoDst/frames/"
frame_nos = [45, 1, 4, 4, 40, 10, 20, 22, 37, 24, 7, 25, 25, 13, 2, 39, 40, 24, 25, 18, 38, 26, 14, 9]

batch_size = 4
trainset = CustomDset(frames_path,frame_nos,train = True,transform = pre_process)
trainloader = DataLoader(trainset,batch_size = batch_size,shuffle = False)

best_epe = 1e9
opt = torch.optim.SGD(model.parameters(),lr = 1e-3,momentum = 0.9)
start_epoch = 0
end_epoch = 50

model.train()
for ep in range(start_epoch,end_epoch):
    print(f"Epoch = {ep}")
    running_epe = 0.0
    for (f1,f2),gt_flows in trainloader:
        opt.zero_grad()
        fs = torch.cat((f1,f2),dim = 1)
        flows = model(fs.to(device))[0]
        epe = torch.norm(gt_flows.to(device) - flows, p = 2 ,dim = 1).mean()
        epe.backward()
        opt.step()
        running_epe = running_epe + epe.item()

    running_epe = running_epe / 6
    if(running_epe < best_epe):
        best_epe = running_epe

        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'best_epe': best_epe
        }, chkpt_path + "best_model.pth")

    torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'best_epe': best_epe
        }, chkpt_path + "checkpoint.pth")

    print(f"Loss at epoch={ep}:{running_epe}")

best_model_chkpt = torch.load(chkpt_path + "best_model.pth",map_location = device)
model.load_state_dict(best_model_chkpt['model_state_dict'])

# best_model_chkpt = torch.load(chkpt_path + "checkpoint.pth",map_location = device)
# model.load_state_dict(best_model_chkpt['model_state_dict'])

# frames_nos_test = [35, 10, 5, 45, 35, 35]
frames_nos_test = [35,10,5,45,35,35]
# frames_nos_test = [20,35,5,45,35,35]
testset = CustomDset(frames_path,frames_nos_test,train = False,transform = pre_process)
test_epe = 0.0
model.eval()
for (f1,f2),gt_flow in testset:
    fs = torch.cat((f1.unsqueeze(0),f2.unsqueeze(0)),dim = 1)
    flows = model(fs.to(device))[0]
    plot([flow_to_image(flows),flow_to_image(gt_flow)])
    epe = torch.norm(gt_flow.to(device) - flows, p = 2 ,dim = 1).mean()
    print(epe.item())
    test_epe = test_epe + epe.item()

print("Avg EPE :",test_epe/len(testset))

