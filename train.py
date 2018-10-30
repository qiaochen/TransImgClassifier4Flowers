# Imports here
import torch
from torch import optim, nn
import json
from model_utils import get_model
from data_utils import prepare_data_loaders
import os
import argparse

torch.manual_seed(999)

def train(model, 
          dataloaders,
          lr=5e-4, 
          lr_decay=.995, 
          feedback_interval=10, 
          n_epoches=1000,
          tolerance_thred=3,
          tmp_path="./best_model.pth"):
    """
    Train the model using transfer learning
    """
    tolerance = tolerance_thred
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    model.to(device)
    dev_losses, lowest_dev_loss = [], float('inf')
    dev_acces = []
    losses = []
    for i_epoch in range(1, n_epoches+1):
        model.train()
        for ii, (inputs, labels) in enumerate(dataloaders['train']):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print("Epoch: {}, iter: {}, train_loss {}".format(i_epoch, ii+1, loss))
            if (ii+1) % feedback_interval == 0:
                dev_loss, dev_acc = validate(model, dataloaders['valid'], nn.NLLLoss(size_average=False))
                dev_losses.append(dev_loss)
                dev_acces.append(dev_acc)
                if dev_loss < lowest_dev_loss:
                    lowest_dev_loss = dev_loss
                    tolerance = tolerance_thred
                    torch.save(model.classifier.state_dict(), tmp_path)
                    print("Best model saved..")
                else:
                    if tolerance == 0:
                        break
                    model.classifier.load_state_dict(torch.load(tmp_path))
                    optimizer = optim.Adam(model.classifier.parameters(), lr=lr*lr_decay)
                    model.train()
                    tolerance -= 1
                    print("Recoverred from last best model.")
                print("Epoch: {}, iter: {}, dev_acc: {}, dev_loss {}".format(
                    i_epoch, ii+1, dev_acc, dev_loss))
                    
                    
        if tolerance == 0: 
            print("Performance halts, early-stop training.")
            break
    os.remove(tmp_path)
    return model, dev_losses, dev_acces, lowest_dev_loss, losses


def validate(model, testloader, criterion):
    """
    Validate model on development set or test set.
    """
    test_loss = 0
    accuracy = 0
    count = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model.forward(images)
        test_loss += criterion(outputs, labels)
        count += images.size()[0]
        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).sum()
        model.train()
    return test_loss/count, accuracy/count


if __name__ == "__main__":
    description = """Transfer learning flower imgage classification"""
    
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', default='./', type=str, help="Specify the path for saving the trained model.")
    parser.add_argument("--arch", default="densenet201", type=str, help="Specify the featrue extractor model for transfer learning. Ensure the model names follow those in torchvision.models (e.g. 'densenet201')")
    parser.add_argument("--gpu", default=False, action='store_true', help="Set the flag if training using GPU")
    parser.add_argument("--learning_rate",default=5e-4, type=float, help="Setting learning rate")
    parser.add_argument("--hidden_units",default=500, type=int, help="Setting number of hidden units in the feedforward classifiers")
    parser.add_argument("--epochs",default=100, type=int, help="Setting number of epoches for model training")
    args = parser.parse_args()
    
    device_name = "cuda" if args.gpu else "cpu"
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    final_model_path = os.path.join(args.save_dir, "final_model.pth")
    pre_model_name = args.arch
    lr = args.learning_rate
    hidden_dim = args.hidden_units
    output_dim = 102
    n_epoches = args.epochs
    
    device = torch.device(device_name)
    dataloaders, class2idx = prepare_data_loaders(train_dir, valid_dir, test_dir,  64)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
   
    model = get_model(pre_model_name, hidden_dim = 500, output_dim = 102)
    model, dev_losses, dev_acces, lowest_dev_loss, train_losses = \
        train(model, 
              dataloaders, 
              lr=lr, 
              lr_decay=.995, 
              feedback_interval=10, 
              n_epoches=n_epoches,
              tolerance_thred=3)
    print("-"*25)
    test_loss, accuracy = validate(model, dataloaders['test'], nn.NLLLoss(size_average=False))
    print("Test accuracy: {}, test loss: {}".format(accuracy, test_loss))
    
    torch.save({"class2idx":class2idx,
                "output_dim": output_dim,
                "hidden_dim": hidden_dim,
                "pre_model_name":pre_model_name,
                "cat_to_name":cat_to_name,
                "model_state_dict":model.classifier.state_dict()}, final_model_path)
    
    print("Model saved!")
    
    