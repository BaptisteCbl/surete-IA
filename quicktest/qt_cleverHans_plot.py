#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:11:08 2023

@author: guillaume
"""
import matplotlib.pyplot as plt
import torch

x = torch.load("data/x.pt").cpu().detach().numpy()
Adv_img_pgd = torch.load("data/x_pgd.pt").cpu().detach().numpy()
Adv_img_fgm = torch.load("data/x_fgm.pt").cpu().detach().numpy()

y_pred = torch.load("data/y_pred.pt").cpu().detach().numpy()
y_pred_pgd = torch.load("data/y_pred_pgd.pt").cpu().detach().numpy()
y_pred_fgm = torch.load("data/y_pred_fgm.pt").cpu().detach().numpy()

adv_pgd = Adv_img_pgd[0,:,:,:].swapaxes(0,1).swapaxes(1,2)
adv_fgm = Adv_img_fgm[0,:,:,:].swapaxes(0,1).swapaxes(1,2)

base =  x[0,:,:,:].swapaxes(0,1).swapaxes(1,2)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(base)
ax1.set_title(y_pred[0])
ax2.imshow(adv_pgd)
ax2.set_title(y_pred_pgd[0])
ax3.imshow(adv_fgm)
ax3.set_title(y_pred_fgm[0])

plt.show()
