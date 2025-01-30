def fgsm_attack(model, loss, images, labels, eps=0.3):

    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    images = images.clone().detach().requires_grad_(True)

    outputs = model(images)

    model.zero_grad()
    cost = loss(outputs, labels).to(DEVICE)
    cost.backward()

    attack_images = images + eps * images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)

    return attack_images