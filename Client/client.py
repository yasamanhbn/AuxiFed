import torch
import numpy as np
from tqdm import tqdm

class Client():
  def __init__(self, batch_size, device, train_loader, test_loader, class_dict, gan_epoch, class_num, id):
    self.BATCH_SIZE = batch_size
    self.device = device
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.id = id
    self.class_dict = class_dict
    self.class_lens = []

    for i in range(class_num):
      if i not in class_dict.keys():
        self.class_dict[i] = []
      self.class_lens.append(len(self.class_dict[i]))

    self.generator = ACGenerator().to(device)
    self.generator.apply(weights_init_normal)
    self.discriminator = ACDiscriminator().to(device)
    self.discriminator.apply(weights_init_normal)
    self.generator = trainACGAN(train_loader, self.generator, self.discriminator, device, gan_epoch, batch_size, class_num, self.id)

  def train_client(self, local_model, iters, lr, mode = 'train'):
    torch.autograd.set_detect_anomaly(True)
    #optimzer for training the local models
    local_model.to(self.device)
    criterion = torch.nn.CrossEntropyLoss().to(self.device)
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
    train_loss = 0.0
    accuracy = 0
    local_model.train()

    #Iterate for the given number of Client Iterations
    for i in range(iters):
        batch_loss = 0.0
        correct = 0
        loop_train = tqdm(enumerate(self.train_loader, 1), total=len(self.train_loader), desc="train", position=0, leave=True)
        for batch_idx, (data, target) in loop_train:
            data, target = data.to(self.device),  target.type(torch.LongTensor).to(self.device)
            original_data = copy.deepcopy(data)
            data = data.detach()
            z = Variable(FloatTensor(np.random.normal(0, 1, (int(self.BATCH_SIZE / config.replace_probs), config.latent_dim))))
            # Get labels ranging from 0 to n_classes for n rows
            labels = Variable(LongTensor(target))


            if data.shape[0] == self.BATCH_SIZE and mode == 'train':
                  data1 = self.generator(z, labels[:int(self.BATCH_SIZE / config.replace_probs)])
                  data[:int(self.BATCH_SIZE / config.replace_probs)] = data1


            if config.adverseial_training:
                if config.attack == 'fgsm':
                    data2 = fgsm_attack(local_model, criterion, data[:int(self.BATCH_SIZE / config.adverseial_training_probs)], labels[:int(self.BATCH_SIZE / config.adverseial_training_probs)])
                elif config.attack == 'pgd':
                    data2 = pgd_attack(local_model, criterion, data[:int(self.BATCH_SIZE / config.adverseial_training_probs)], labels[:int(self.BATCH_SIZE / config.adverseial_training_probs)])
                data[:int(self.BATCH_SIZE / config.adverseial_training_probs)] = data2

            data = data.clone().detach().requires_grad_(True)

            #set gradients to zero
            optimizer.zero_grad()
            #Get output prediction from the Client model

            output = local_model(data, original_data, labels, self.class_lens, type=mode)
            #Computer loss
            c_loss = criterion(output, target)
            batch_loss = batch_loss + c_loss.item()
            #Collect new set of gradients
            c_loss.backward()
            #Update local model
            optimizer.step()

            correct += calculate_acc(output, target).item()


            loop_train.set_description(f"Train")
            loop_train.set_postfix(
                    loss="{:.4f}".format(batch_loss / batch_idx),
                    accuracy="{:.4f}".format(correct * 100 / batch_idx),
                    refresh=True,
            )
        #add loss for each iteration
        train_loss+=batch_loss/len(self.train_loader)
        accuracy += correct / len(self.train_loader)

    return local_model, local_model.state_dict(), optimizer, train_loss/iters, accuracy/iters


  def test_client(self, local_model):
    #optimzer for training the local models
    test_loss = 0.0
    correct = 0
    local_model.eval()
    local_model = local_model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    #Iterate for the given number of Client Iterations
    with torch.no_grad():
      for batch_idx, (data, target) in enumerate(self.test_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = local_model(data, type="valid").to(DEVICE)

        #Computer loss
        c_loss = criterion(output, target).to(DEVICE)
        test_loss = test_loss + c_loss.item()

        correct += calculate_acc(output, target).item()


    print("Validation Accuracy for Client " +str(self.id + 1) + ": " + str(correct * 100/len(self.test_loader)))
    print("Validation Loss for Client " +str(self.id + 1) + ": " + str(test_loss/len(self.test_loader)))

    return test_loss/len(self.test_loader), correct/len(self.test_loader)