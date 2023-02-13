from deeprobust.image.attack.pgd import PGD
from deeprobust.image.config import attack_params
from deeprobust.image.utils import download_model
import torch
import deeprobust.image.netmodels.resnet as resnet
from torchvision import transforms,datasets

URL = "https://github.com/I-am-Bot/deeprobust_model/raw/master/CIFAR10_ResNet18_epoch_20.pt"
download_model(URL, "$MODEL_PATH$")
print("Load model")
model = resnet.ResNet18().to('cuda')
model.load_state_dict(torch.load("$MODEL_PATH$"))
model.eval()
print("DataLoader")
transform_val = transforms.Compose([transforms.ToTensor()])
test_loader  = torch.utils.data.DataLoader(
                datasets.CIFAR10('deeprobust/image/data', train = False, download=True,
                transform = transform_val),
                batch_size = 10, shuffle=True)

x, y = next(iter(test_loader))
x = x.to('cuda').float()


print("Adversary")
adversary = PGD(model, 'cuda')
Adv_img = adversary.generate(x, y, **attack_params['PGD_CIFAR10'])

import matplotlib.pyplot as plt

adv = Adv_img[0,:,:,:].cpu().detach().numpy().swapaxes(0,1).swapaxes(1,2)

base =  x[0,:,:,:].cpu().detach().numpy().swapaxes(0,1).swapaxes(1,2)


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(base)
ax2.imshow(adv)