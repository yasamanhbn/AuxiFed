import csv

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def plot_loss(disc_loss, gen_loss, id, ganType):
  _, axes = plt.subplots(1, 1, figsize=(12, 3))
  plt.suptitle("Loss plot for Generator and Discriminator of " + ganType +" for client"+ str(id + 1), size=14)
  axes.plot(list(range(1, len(disc_loss) + 1)), disc_loss, 'b',label='discriminator loss')
  axes.plot(list(range(1, len(gen_loss) + 1)), gen_loss, 'r', label='generator loss')
  axes.set_ylabel('Loss Average', size=12, labelpad=10)
  axes.set_xlabel('Epoch', size=12, labelpad=10)
  axes.legend(loc='lower right', fontsize=10)
  axes.grid()
  plt.show()


def save_model_report(all_train_loss, all_acc_loss, all_val_loss, all_val_acc, file_path):

    rows = zip(all_train_loss, all_acc_loss, all_val_loss, all_val_acc)
    headers = ['train_loss', 'train_acc', 'val_loss', 'val_acc']

    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow([g for g in headers])
        for row in rows:
            writer.writerow(row)
    return

def plot_clients_stats(losses, acc):
  _, axes = plt.subplots(1, 2, figsize=(12, 3))
  colors = ['#ff4d4d', '#2eb8b8', '#cc66ff', '#ff9933', '#0066ff', '#cc0099', '#00ff80', '#000099']
  plt.suptitle("Loss and Accuracy plot for Clients", size=14)
  for idx, c_loss in enumerate(losses):
    axes[0].plot(list(range(1, len(c_loss) + 1)), c_loss, colors[idx], label='Client'+str(idx))
    axes[0].set_ylabel('Loss Average', size=12, labelpad=10)
    axes[0].set_xlabel('Epoch', size=12, labelpad=10)
    axes[0].legend(loc='upper right', fontsize=8)

    axes[1].plot(list(range(1, len(c_loss) + 1)), acc[idx], colors[idx], label='Client'+str(idx + 1))
    axes[1].set_ylabel('Accuracy Average', size=12, labelpad=10)
    axes[1].set_xlabel('Epoch', size=12, labelpad=10)
    axes[1].legend(loc='lower right', fontsize=8)

  plt.show()

def plot_acc_loss(train_loss, train_acc, test_loss, test_acc):
    _, axes = plt.subplots(1, 2, figsize=(16, 4))
    plt.suptitle("Accuracy and Loss for train and test data", size=14)
    train_acc = [x*100 for x in train_acc]
    test_acc = [x*100 for x in test_acc]
    axes[0].plot(list(range(1, len(train_acc)+1)), train_acc, 'b',label='train accuracy')
    axes[0].plot(list(range(1, len(train_acc)+1)), test_acc, 'r', label='test accuracy')
    axes[0].set_ylabel('Accuracy Average', size=12, labelpad=7)
    axes[0].set_xlabel('Epoch', size=12, labelpad=10)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid()

    axes[1].plot(list(range(1, len(train_acc)+1)), train_loss, 'b', label='train loss')
    axes[1].plot(list(range(1, len(train_acc)+1)), test_loss, 'r',label='test loss')
    axes[1].set_ylabel('Loss Average', size=12, labelpad=5)
    axes[1].set_xlabel('Epoch', size=12, labelpad=10)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid()
    plt.show()

import os
def save_model(file_path, file_name, model, optimizer=None):
    state_dict = dict()
    state_dict["model"] = model.state_dict()


    if not os.path.exists(file_path):
      os.makedirs(file_path)

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    checkpoint = torch.load(ckpt_path, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer

# custom weights initialization called on netG and netD
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def show_tensor_images(generator):
    n_row = 10
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    grid = make_grid(gen_imgs.cpu(), nrow=10, normalize=True)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(grid.permute(1, 2, 0).data, cmap='binary')
    ax.axis('off')
    plt.show()

def get_noise(samples_num, dimention, device):
    return torch.randn(samples_num, dimention, device=device)

def save_GAN_report(loss_D , loss_G, acc , file_path):

    rows = zip(loss_D , loss_G, acc)
    headers = ['discriminator_loss', 'generator_loss', 'accuracy']

    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow([g for g in headers])
        for row in rows:
            writer.writerow(row)
    return