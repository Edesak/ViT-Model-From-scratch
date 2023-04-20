import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm.auto import tqdm
import wandb




def train_step(model:torch.nn.Module,
               dataloader:DataLoader,
               loss_fn: torch.nn.Module,
               acc_fn:MulticlassAccuracy,
               device:torch.device,
               optim:torch.optim.Optimizer):
    """
    You should use train() function
    :param model:
    :param dataloader:
    :param loss_fn:
    :param acc_fn:
    :param device:
    :param optim:
    :return:
    """
    model.train()

    batch_loss, batch_acc = 0,0

    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(device),y.to(device)
        y_logits = model(X)
        loss = loss_fn(y_logits,y)

        acc_fn.update(y_logits,y)
        acc = acc_fn.compute()

        batch_loss += loss
        batch_acc += acc

        optim.zero_grad()
        loss.backward()
        optim.step()

    train_loss = batch_loss/len(dataloader)
    train_acc = batch_acc*100/len(dataloader)

    return train_loss,train_acc



def val_step(model:torch.nn.Module,
               dataloader:DataLoader,
               loss_fn: torch.nn.Module,
               acc_fn:MulticlassAccuracy,
               device:torch.device):
    """
    You should use train() function
    :param model:
    :param dataloader:
    :param loss_fn:
    :param acc_fn:
    :param device:
    :return:
    """
    model.eval()
    with torch.inference_mode():
        batch_loss, batch_acc = 0, 0

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            loss = loss_fn(y_logits, y)

            acc_fn.update(y_logits, y)
            acc = acc_fn.compute()

            batch_loss += loss
            batch_acc += acc

        val_loss = batch_loss/len(dataloader)
        val_acc = batch_acc*100/len(dataloader)

    return val_loss, val_acc

def train(model:torch.nn.Module,
          device:str,
          train_dataloader:DataLoader,
          test_dataloader:DataLoader,
          loss_fn:torch.nn.Module,
          acc_fn:MulticlassAccuracy,
          optim:torch.optim.Optimizer,
          epochs:int,
          log_to_wandb:bool,
          lr_scheduler:torch.optim.lr_scheduler =None):
    """
    Training loop with progress bar that help you not write so much code like a monkey.

    :param model: Model you want to train on
    :param device: Select device you want to train on
    :param train_dataloader: Train dataloader for you dataset
    :param test_dataloader: Test dataloader for you dataset
    :param loss_fn: Loss function
    :param acc_fn: Accuracy function from torchmetric
    :param optim: Optimizer from pytorch
    :param epochs: Number of epochs to train
    :param log_to_wandb: Do you want to use wandb to upload your training data
    :param lr_scheduler: If you have scheduler past it here if None skip (Optional)
    :return results: (dict of results with name of the model and optimizer, dict loss with acc every epoch) if log_to_wnadb= True return None
    """
    acc_fn.to(device)
    model.to(device)
    pbar = tqdm(total=epochs, bar_format='{l_bar}{bar:50}{r_bar}', colour='green', leave=True)

    if log_to_wandb == False:
        loss_curve = {"train_loss": [],
                        "train_accuracy":[],
                        "validation_loss":[],
                        "validation_accuracy":[]}

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model, device=device, loss_fn=loss_fn, acc_fn=acc_fn,
                                                    dataloader=train_dataloader, optim=optim)
        val_loss, val_acc = val_step(model=model, device=device, loss_fn=loss_fn, acc_fn=acc_fn,
                                              dataloader=test_dataloader)


        pbar.update(1)
        tqdm.write(
            "\033[34m" + f" | Train_loss: {train_loss:.4f} Train_acc: {train_acc:.2f}% | Val_loss: {val_loss:.4f} Val_acc: {val_acc:.2f}%" + "\033[0m")

        if log_to_wandb:
            wandb.log({
                "train_loss":train_loss,
                "train_acc":train_acc,
                "val_loss":val_loss,
                "val_acc":val_acc
            })
        else:
            loss_curve["train_loss"].append(train_loss.item())
            loss_curve["train_accuracy"].append(train_acc.item())
            loss_curve["validation_loss"].append(val_loss.item())
            loss_curve["validation_accuracy"].append(val_acc.item())

        if lr_scheduler:
            lr_scheduler.step()





    if log_to_wandb ==False:
        result = {"model_name": model.__class__.__name__,
                  "train_loss": train_loss.item(),
                  "train_accuracy": train_acc.item(),
                  "validation_loss": val_loss.item(),
                  "validation_accuracy": val_acc.item(),
                  "optimizer_name": optim.__class__.__name__}
        return result,loss_curve
    else:
        return None
