import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
def trainACGAN(loader_train, generator, discriminator, device, epochs, batch_size, n_classes, id):
  adversarial_loss = torch.nn.BCELoss()
  auxiliary_loss = torch.nn.CrossEntropyLoss()

  optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
  optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

  FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
  LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

  Gen_los = []
  Disc_los = []
  acc_cls = []

  generator, optimizer_G = load_model(
        ckpt_path= config.ACGEN + str(id) +'.ckpt', model=generator, optimizer=optimizer_G
  )

  discriminator, optimizer_D = load_model(
        ckpt_path= config.ACDIS + str(id) +'.ckpt', model=discriminator, optimizer=optimizer_D
  )


  for epoch  in range(1, 1 + epochs):
    print("-"*10 + str(epoch) + "-"*10)
    # loop_train = tqdm(enumerate(loader_train, 1), total=len(loader_train), desc="train", position=0, leave=True)
    mean_lossD = 0
    mean_lossG = 0
    mean_d_acc = 0
    for idx, (imgs, labels) in enumerate(loader_train, 1):
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, config.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)
        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

        g_loss.backward()
        optimizer_G.step()

        mean_lossG += g_loss.item()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)
        mean_d_acc += d_acc

        d_loss.backward()
        optimizer_D.step()
        mean_lossD += d_loss.item()


        loop_train.set_description(f"Train - epoch : {epoch}")
        loop_train.set_postfix(
                    lossD="{:.4f}".format(mean_lossD/idx),
                    lossG="{:.4f}".format(mean_lossG/idx),
                    ACCD="{:.4f}".format(100 * mean_d_acc / idx),
                    refresh=True)

    Gen_los.append(mean_lossG / len(loader_train))
    Disc_los.append(mean_lossD / len(loader_train))
    acc_cls.append(mean_d_acc / len(loader_train))

  return generator