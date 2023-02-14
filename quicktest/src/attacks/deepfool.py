"""***************************************************************************************
* Code adapted from the DeepRobust library:
* 
*    Title: deeprobust
*    Date: 14/02/2023
*    Code version: 0.2.6
*    Availability: https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/image/attack/deepfool.py
*
***************************************************************************************"""


import numpy as np
import torch as torch
import copy
import collections


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
        elif isinstance(x, collections.abc.Iterable):
            for elem in x:
                zero_gradients(elem)


def deepfool(model, image, num_classes, overshoot, max_iteration, device):
    """
    Call this function to generate adversarial examples.

    Parameters
    ----------
    image : 1*3*H*W
        original image
    kwargs :
        user defined paremeters

    Returns
    -------
    adv_img :
        adversarial examples
    """
    adv_img, r, ite = _deepfool(
        model, image, num_classes, overshoot, max_iteration, device
    )
    return adv_img


def _deepfool(model, imageO, num_classes, overshoot, max_iter, device):
    image = torch.zeros((1, 3, 32, 32)).to(device)
    image[0, :, :, :] = imageO

    f_image = model.forward(image).data.cpu().numpy().flatten()
    output = (np.array(f_image)).flatten().argsort()[::-1]

    output = output[0:num_classes]
    label = output[0]

    input_shape = image.cpu().numpy().shape
    x = copy.deepcopy(image).requires_grad_(True)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    fs = model.forward(x)

    for i in range(max_iter):
        pert = np.inf
        fs[0, output[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, output[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, output[k]] - fs[0, output[0]]).data.cpu().numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).to(device)

        x = pert_image.detach().requires_grad_(True)
        fs = model.forward(x)

        if not np.argmax(fs.data.cpu().numpy().flatten()) == label:
            break

    r_tot = (1 + overshoot) * r_tot
    return pert_image.to(device), r_tot, i
