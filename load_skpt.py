import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision import transforms, datasets
from networks.DDAM import DDAMNet
from DDAMFN1.sam import SAM


def load_ckpt(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

    model = DDAMNet()
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型可学习参数

    optimizer = SAM(model.parameters(), torch.optim.Adam, lr=0.00001, rho=0.05, adaptive=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    return model

# Load pre-trained model
model = load_ckpt('DDAMFN1/checkpoints_ver2.0/affecnet7_epoch19_acc0.671.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify the data transformation detail 
data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

# Prepare the Dataset for validation
val_dataset = datasets.ImageFolder('archive', transform=data_transforms_val)

print('Validation set size:', val_dataset.__len__())

# Define the dataloader
val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = 32,
                                               num_workers = 8,
                                               shuffle = False,  
                                               pin_memory = True)

with torch.no_grad():  # 禁用梯度计算
    bingo_cnt = 0
    sample_cnt = 0
    best_acc = 0
    for imgs, targets in val_loader:
        
        imgs = imgs.to(device)
        targets = targets.to(device)
        out,feat,heads = model(imgs)
        _, predicts = torch.max(out, 1) 
        correct_num  = torch.eq(predicts,targets)
        bingo_cnt += correct_num.sum().cpu()
        sample_cnt += out.size(0)
    acc = bingo_cnt.float()/float(sample_cnt)
    acc = np.around(acc.numpy(),4)
    best_acc = max(acc,best_acc)
    tqdm.write("best_acc:" + str(best_acc))