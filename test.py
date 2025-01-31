import torch
from utils import *
from tqdm import tqdm

def test(model, dataloader, optimizer):
    """
    Tests the Federated global model for the given dataset
    :param model: Trained CNN model for testing
    :param dataloader: data iterator used to test the model
    :return test_loss: test loss for the given dataset
    :return preds: predictions for the given dataset
    :return accuracy: accuracy for the prediction values from the model
    """
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    model.eval()
    model = model.to(DEVICE)
    all_preds = []

    with torch.no_grad():
        loop_val = tqdm(enumerate(dataloader, 1), total=len(dataloader), desc="Test", position=0, leave=True)
        time1 = time.time()
        for batch_idx, (data, target) in loop_val:
              data, target = data.to(DEVICE), target.to(DEVICE)

              output = model(data, type="valid").to(DEVICE)

              loss = criterion(output, target)
              test_loss += loss.item()
              correct += calculate_acc(output, target).item()

              all_preds = all_preds + (output.detach().numpy().tolist())

              loop_val.set_description(f"Test")
              loop_val.set_postfix(
                loss="{:.4f}".format(test_loss / batch_idx),
                accuracy="{:.4f}".format(correct * 100/ batch_idx),
                refresh=True,
              )
    print("Final Test Total Acuracy:" + str(correct * 100/len(dataloader)))
    print("Final Test  Total Loss:" + str(test_loss/len(dataloader)))
    return test_loss/len(dataloader), correct/len(dataloader), all_preds