import torch
from torch import nn
import numpy as np

from typing import Tuple, Union, List, Callable
from torch.optim import SGD
import torchvision
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

train_dataset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())

batch_size = 128

train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int( 0.1 * len(train_dataset))])

# Create separate dataloaders for the train, test, and validation set
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']
  
def train(
    model: nn.Module, optimizer: SGD,
    train_loader: DataLoader, val_loader: DataLoader,
    epochs: int = 20
)-> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Trains a model for the specified number of epochs using the loaders.

    Returns: 
    Lists of training loss, training accuracy, validation loss, validation accuracy for each epoch.
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)
    loss = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for e in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        # Main training loop; iterate over train_loader. The loop
        # terminates when the train loader finishes iterating, which is one epoch.
        for (x_batch, labels) in train_loader:
            x_batch, labels = x_batch.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            labels_pred = model(x_batch)
            batch_loss = loss(labels_pred, labels)
            train_loss = train_loss + batch_loss.item()

            labels_pred_max = torch.argmax(labels_pred, 1)
            batch_acc = torch.sum(labels_pred_max == labels)
            train_acc = train_acc + batch_acc.item()

            batch_loss.backward()
            optimizer.step()
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc / (batch_size * len(train_loader)))

        # Validation loop; use .no_grad() context manager to save memory.
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        

        with torch.no_grad():
            for (v_batch, labels) in val_loader:
                v_batch, labels = v_batch.to(DEVICE), labels.to(DEVICE)
                labels_pred = model(v_batch)
                v_batch_loss = loss(labels_pred, labels)
                val_loss = val_loss + v_batch_loss.item()

                v_pred_max = torch.argmax(labels_pred, 1)
                batch_acc = torch.sum(v_pred_max == labels)
                val_acc = val_acc + batch_acc.item()
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_acc / (batch_size * len(val_loader)))
            
        
        print(val_acc / (batch_size * len(val_loader)))
        print(get_lr(optimizer))
        scheduler.step(val_accuracies[-1])

    return train_losses, train_accuracies, val_losses, val_accuracies

def rand_search(
    model_fn: nn.Module
) -> float:
    """
    Parameter search for our linear model using SGD.

    Args:
    train_loader: the train dataloader.
    val_loader: the validation dataloader.
    model_fn: a function that, when called, returns a torch.nn.Module.

    Returns:
    The learning rate with the least validation loss.
    NOTE: you may need to modify this function to search over and return
     other parameters beyond learning rate.
    """
    num_iter = 10  # This will likely not be enough for the rest of the problem.
    best_loss = torch.tensor(np.inf)
    best_lr = 0.0
    lr_dict = {}

    # lrs = torch.linspace(10 ** (-6), 10 ** (-1), 100)
    lrs = torch.logspace(-1, -3, steps=3)
    # Ms = torch.linspace(100, 400, 100)
    momentums = torch.linspace(0.5, 0.8, 100)

    for i in range(num_iter):
        dataset_size = len(train_dataset)
        train_subset_size = int(0.1 * len(train_dataset))
        val_subset_size = int(0.1 * len(val_dataset))

        train_small, _ = random_split(train_dataset, [train_subset_size, len(train_dataset) - train_subset_size])
        val_small, _ = random_split(val_dataset, [val_subset_size, len(val_dataset) - val_subset_size])

        rand_lr = lrs[torch.randperm(len(lrs))[0]]
        rand_M = 400
        # rand_lr = torch.FloatTensor(1).uniform_(0.01, 0.011).item()
        # rand_momentum = torch.FloatTensor(1).uniform_(0.5, 0.8).item()
        print(f"trying learning rate {rand_lr}")
        
        rand_momentum = momentums[torch.randperm(len(momentums))[0]]
        print(f"trying momentum {rand_momentum}")
        
        '''
        tensor_size = lrs.size(0)
        random_index = torch.randint(low=0, high=tensor_size, size=(1,)).item()
        random_lr = lrs[random_index].item()
        print(f"trying learning rate {random_lr}")
        # print(f"trying momentum {momentum}")
        '''
        model = model_fn()
        optim = SGD(model.parameters(), lr=rand_lr, momentum=rand_momentum)

        
        train_subset_loader = DataLoader(
                                train_small,
                                batch_size=batch_size,
                                shuffle=True
                              )

        test_subset_loader = DataLoader(
                                val_small,
                                batch_size=batch_size,
                                shuffle=True
                              )
        
        train_loss, train_accuracy, val_loss, val_accuracy = train(
            model, optim, train_subset_loader, test_subset_loader, 15
        )

        lr_dict[tuple((rand_lr, rand_momentum, rand_M))] = min(val_loss)
        print(min(val_loss))

    return lr_dict

def evaluate(
    model: nn.Module, loader: DataLoader
) -> Tuple[float, float]:
    """Computes test loss and accuracy of model on loader."""
    loss = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for (batch, labels) in loader:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            y_batch_pred = model(batch)
            batch_loss = loss(y_batch_pred, labels)
            test_loss = test_loss + batch_loss.item()

            pred_max = torch.argmax(y_batch_pred, 1)
            batch_acc = torch.sum(pred_max == labels)
            test_acc = test_acc + batch_acc.item()
        test_loss = test_loss / len(loader)
        test_acc = test_acc / (batch_size * len(loader))
        return test_loss, test_acc

def relu_hidden_model() -> nn.Module:
    """Instantiate a linear model and send it to device."""
    model =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 400),
            nn.ReLU(),
            nn.Linear(400, 10)
         )
    return model.to(DEVICE)


def convolutional_model():
  M = 256
  N = 14
  k = 5
  model = nn.Sequential(
      nn.Conv2d(3, M, kernel_size=k),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=N),
      nn.Flatten(),
      nn.Linear(((33-k)//N)**2*M, 10)
  )
  return model.to(DEVICE)

def plot_model(model, threshold):
  print(threshold)
  lr_dict = rand_search(model)
  top_3 = sorted(lr_dict.items(), key=lambda x:x[1])[:3]

  m1 = model()
  m3 = model()
  m2 = model()
  
  optimizer1 = SGD(m1.parameters(), lr=top_3[0][0][0].item(), momentum=top_3[0][0][1].item())
  optimizer2 = SGD(m2.parameters(), lr=top_3[1][0][0].item(), momentum=top_3[1][0][1].item())
  optimizer3 = SGD(m3.parameters(), lr=top_3[2][0][0].item(), momentum=top_3[2][0][1].item())

  train_loss3, train_accuracy3, val_loss3, val_accuracy3 = train(
    m3, optimizer3, train_loader, val_loader, 35
  )
  print("m3")
  print(train_accuracy3)
  print(val_accuracy3)
  
  
  train_loss2, train_accuracy2, val_loss2, val_accuracy2 = train(
    m2, optimizer2, train_loader, val_loader, 35
  )
  print("m2")
  print(train_accuracy2)
  print(val_accuracy2)
  

  train_loss1, train_accuracy1, val_loss1, val_accuracy1 = train(
    m1, optimizer1, train_loader, val_loader, 35
  )
  print("m1")
  print(train_accuracy1)
  print(val_accuracy1)

  epochs = range(1, 36)
  plt.axhline(y=threshold, color='r')
  plt.plot(epochs, train_accuracy1, label="Train Accuracy for 1st lr")
  plt.plot(epochs, val_accuracy1, label="Validation Accuracy for 1st lr", linestyle="dashed")
  plt.plot(epochs, train_accuracy2, label="Train Accuracy for 2nd lr")
  plt.plot(epochs, val_accuracy2, label="Validation Accuracy for 2nd lr", linestyle="dashed")
  plt.plot(epochs, train_accuracy3, label="Train Accuracy for 3rd lr")
  plt.plot(epochs, val_accuracy3, label="Validation Accuracy for 3rd lr", linestyle="dashed")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.title("neural network Accuracy for CIFAR-10 vs Epoch")
  plt.show()

plot_model(relu_hidden_model, 0.5)
plot_model(convolutional_model, 0.65)