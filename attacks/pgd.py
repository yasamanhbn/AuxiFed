from utils import *
import torch
import torch.nn as nn

def pgd_attack(model, criterion, images, labels, eps=0.3, alpha=0.00784313725490196, iters=10) :
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters) :
        images = images.clone().detach().requires_grad_(True)
        outputs = model(images, type="valid")

        model.zero_grad()
        cost = loss(outputs, labels).to(DEVICE)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=-1, max=1).detach_()

    return images