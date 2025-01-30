from data import *
from Client import *
from Model import *
from utils import *

def train(config):
    test_data = load_test_dataset(config, batch_size=config.BATCH_SIZE)
    test_data  = get_test_dataLoader(test_data, config.BATCH_SIZE)
    users = get_dataLoader(config)

    clients = []

    global_model = CNN()
    global_params = global_model.state_dict()

    for c in range(config.NUM_CLIENTS):
      print("Client " + str(c + 1) + " Initialization has started")
      
      client_dataloader = torch.utils.data.DataLoader(users[c].train_data, batch_size=config.BATCH_SIZE, shuffle=True)
      client_test_dataloader = torch.utils.data.DataLoader(users[c].test_data, batch_size=config.BATCH_SIZE, shuffle=True)
      clients.append(Client(batch_size=config.BATCH_SIZE, device=DEVICE, train_loader=client_dataloader, test_loader=client_test_dataloader,
                            class_dict=users[c].class_dict, gan_epoch=config.gan_epoch, class_num=config.CLASS_NUM, id=c))


      print("-"*100)

    #get model and define criterion for loss
    global_model.train()
    all_train_loss = list()
    all_train_acc = list()
    all_val_loss = list()
    all_val_acc = list()

    client_losses = [list() for i in range(config.NUM_CLIENTS)]
    client_acces = [list() for i in range(config.NUM_CLIENTS)]

    #Train the model for given number of epochs
    for epoch in range(1, config.NUM_EPOCHS+1):
        print("-"*40 + "Epoch " + str(epoch) + "-"*40)
        local_params, local_losses, local_acc = [], [], []
        val_loss = 0
        val_acc = 0
        #Send a copy of global model to each client
        for idx, clnt in enumerate(clients):
            print("Clinet " + str(idx + 1) + ":")
            #Perform training on client side and get the parameters
            local_mdl, param, optimizer, c_loss, c_accuracy = clnt.train_client(copy.deepcopy(global_model), config.LOCAL_ITERS, lr=config.lr, mode=mode)
            loss_, acc_ = clnt.test_client(copy.deepcopy(local_mdl))

            local_params.append(copy.deepcopy(param))
            local_losses.append(copy.deepcopy(c_loss))
            local_acc.append(copy.deepcopy(c_accuracy))

            client_losses[idx].append(copy.deepcopy(c_loss))
            client_acces[idx].append(copy.deepcopy(c_accuracy))

        #Federated Average for the paramters from each client
        global_params = FedAvg(local_params)
        #Update the global model
        global_model.load_state_dict(global_params)
        all_train_loss.append(sum(local_losses)/len(local_losses))
        all_train_acc.append(sum(local_acc)/len(local_acc))

        print("Test phase for each client on final model")
        for idx, clnt in enumerate(clients):
            loss_, acc_ = clnt.test_client(copy.deepcopy(global_model))
            val_acc += (config.test_len[idx] * acc_)
            val_loss += (config.test_len[idx] * loss_)
        print("-"*100)
        print("Average Accuracy: " + str(val_acc * 100 / sum(config.test_len)))
        print("Average Loss: " + str(val_loss / sum(config.test_len)))
        print("-"*100)
        print("Final Test in global moedel with test dataset")

        test_loss, accuracy, all_preds = test(copy.deepcopy(global_model), test_data, optimizer)
        all_val_loss.append(test_loss)
        all_val_acc.append(accuracy)

        save_model(
          file_path=config.save_checkpoint,
          file_name=f"cgan-ckpt_epoch{epoch}.ckpt",
          model=global_model,
        )
    test_loss, accuracy, all_preds = test(copy.deepcopy(global_model), test_data, optimizer)

    plot_clients_stats(client_losses, client_acces)
    save_model_report(all_train_loss, all_train_acc, all_val_loss, all_val_acc, config.Results)
