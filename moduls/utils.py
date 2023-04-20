import math

import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import List, Tuple
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import pandas as pd


def setup_seed(seed:int = 42):
    """
    Simple fuction to set seed for reproduction of the code.
    :param seed: int default 42
    :return:
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def setup_device():
    """
    Check if GPU available if not returns cpu

    :return: cuda or cpu
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'




def save_model_dict(model: torch.nn.Module,
                    target_dir: str,
                    model_name: str):
    """Saves a PyTorch model to a target directory as dict, does not save whole model.

    Args:
      model: A target PyTorch model to save.
      target_dir: A directory for saving the model to.
      model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
      save_model(model=model_0,
                 target_dir="models",
                 model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def pred_plot_image(model: torch.nn.Module,
                    image_path: str,
                    class_names: List[str],
                    device: str,
                    image_size: Tuple[int, int] = (224, 224),
                    transform: transforms.Compose = None):
    """
    Predict and plot custom image with label name.

    :param model: Select model that will predict
    :param image_path: Select path of the image
    :param class_names: Class names in list of strings
    :param device: device you want to run at
    :param image_size: image size it should transform if not transform defined
    :param transform: Select your transform for data, if None take it creates transform used for Effitient_net_B0
    :return:
    """
    img = Image.open(image_path)

    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(size=image_size),
            transforms.ToTensor(),
            transforms.Normalize(std=[0.485, 0.456, 0.406],
                                 mean=[0.229, 0.224, 0.255])
        ])

    model.to(device)

    model.eval()
    with torch.inference_mode():
        # Creating [batch_size,color,height,width]
        img_trans = image_transform(img).unsqueeze(dim=0)
        img_logits = model(img_trans.to(device))

        img_probs = torch.softmax(img_logits, dim=1)
        img_pred = torch.argmax(img_probs, dim=1)
        probs = img_probs.tolist()[0]
        plt.figure(figsize=(10, 7))
        plt.imshow(img)
        plt.title(f"Pred: {class_names[img_pred]} | Prob: {probs[img_pred.item()] * 100:.2f} %")
        plt.axis(False)
        plt.show()


def preds_probs(model: torch.nn.Module,
                dataloader: DataLoader,
                device: str) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Small function for browsing the dataloader. Return predicted and true values. Using your model.

    Do you want to use it? Use it, but there is function plot_confmat_classification for easier use.

    :param model: Chosen model you want to predict on
    :param dataloader: Chosen dataloader you want to walk
    :param device: What device should you use
    :return: Tuple(Predicted List[Tensors], Target List[Tensors], Probabilities[Tensor])
    """
    preds = []
    y_true = []
    y_probs_m = []
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            y_probs = torch.softmax(y_logits, dim=1)
            y_pred = torch.argmax(y_probs, dim=1)

            y_probs = y_probs.to('cpu')

            for probability in y_probs:
                y_probs_m.append(probability[torch.argmax(probability)].item())

            preds.append(y_pred.to('cpu'))
            y_true.append(y.to('cpu'))

    return torch.cat(preds, dim=0), torch.cat(y_true, dim=0), y_probs_m


def plot_confmat_classification(model: torch.nn.Module,
                                dataloader: DataLoader,
                                device: str,
                                class_names: List[str]):
    """
    This is wholesome function to predict and plot confusion matrix for classification.

    :param model: Chosen model you want to predict on
    :param dataloader: Chosen dataloader you want to walk
    :param device: What device should you use
    :param class_names: What are the class names
    :return:
    """
    conf_matrix = MulticlassConfusionMatrix(num_classes=len(class_names))

    preds, y, _ = preds_probs(model, dataloader, device)

    print(f"Predicted: {preds}")
    print(f"Tatgets: {y}")

    conf_matrix = conf_matrix(preds, y)
    print(conf_matrix)
    plot_confusion_matrix(conf_mat=conf_matrix.numpy(), figsize=(10, 7), class_names=class_names)
    plt.show()


def load_model(path_pth, model, device):
    """
    simple function to load dict to model
    :param path_pth: Path to *.pth you want to load
    :param model: model you want to load the dict in
    :return:
    """
    model.load_state_dict(torch.load(f=path_pth))
    model.to(device)
    return model


def plot_most_wrong(model: torch.nn.Module,
                    dataloader: DataLoader,
                    device: str,
                    class_names: List[str],
                    num_images: int,
                    transform: transforms.Compose = None,
                    applied_mean: bool = True,
                    applied_std: bool = True,
                    fig_size: Tuple[int, int] = (10, 7)):
    """
    Function that plot the worst predition from your dataset of images.
    Recomendation: try to keep it at maximum of 10 images.
    This function will also print Dataframe of the worst images with indexes
    :param model:
    :param dataloader:
    :param device:
    :param class_names:
    :param num_images:
    :param transform:
    :param applied_mean:
    :param applied_std:
    :param fig_size:
    :return:
    """
    pd.options.display.max_columns = 6

    if transform is None:
        applied_std = False
        applied_mean = False

    #get prediction, targets and probabilities
    y_preds, y_true, y_probs = preds_probs(model, dataloader, device)

    df = pd.DataFrame({
        "Target": y_true,
        "Predicted": y_preds,
        "Probability": y_probs
    })
    #Creating ne column if match = True
    df['labels_match'] = df['Target'] == df['Predicted']
    #Creating separate df from all rows that have False -> ~True = False
    filtered_df = df[~df['labels_match']]
    #Sorting by probability
    filtered_df = filtered_df.sort_values(by='Probability', ascending=False)
    #Getting idexes
    indexes = filtered_df.index
    #Adding idexes to separate column
    filtered_df = filtered_df.reset_index()

    if len(indexes) < num_images:
        print(f"There is {len(indexes)} wrong images, but you trying to plot {num_images}. Please adjust your range.")
    else:
        index_in_batches = []
        batch_nums = []
        batch_and_index = []

        #Calculating in witch batch it is located and what index inside the batch
        for i, num in enumerate(indexes):
            batch_and_index.append(divmod(num, dataloader.batch_size))

            batch_num, index_in_batch = batch_and_index[i]

            batch_nums.append(batch_num)
            index_in_batches.append(index_in_batch)

        filtered_df["Batch number"] = batch_nums
        filtered_df["Index inside batch"] = index_in_batches

        print(filtered_df)

        #Finding the images
        worst_images = []
        for i, index in batch_and_index:
            for batch_num, (X, y) in enumerate(dataloader):
                if batch_num == i:
                    worst_images.append(X[index])
                    break

        #Revert std and mean for plotting
        if applied_std:
            std = torch.Tensor(transform.std)[:, None, None]
            worst_images_std = []
            for worst_image in worst_images:
                worst_image = worst_image * std
                worst_images_std.append(worst_image)
            worst_images = worst_images_std

        if applied_mean:
            mean = torch.Tensor(transform.mean)[:, None, None]
            worst_images_std_mean = []
            for worst_image in worst_images_std:
                worst_image = worst_image + mean
                worst_images_std_mean.append(worst_image)
            worst_images = worst_images_std_mean

        #Setting up number of rows and col
        num_rows, _ = divmod(num_images, 3)

        num_col = math.ceil(num_images / num_rows)

        fig = plt.figure(figsize=fig_size)

        #Plotting
        for i, index in enumerate(indexes):

            fig.add_subplot(num_rows, num_col, i + 1)
            plt.imshow(worst_images[i].permute(1, 2, 0))
            plt.title(
                f"Og: {class_names[y_true[index]]}| Pred: {class_names[y_preds[index]]} | Prob: {y_probs[index] * 100:.2f}%")
            plt.axis(False)

            if num_images <= i + 1:
                break

        plt.show()


def plot_images_clas(num_rows: int = 3,
                     num_col: int = 3,
                     fig_size: float = 2,
                     dataset=None,
                     class_names=None,
                     cmap: str = 'gray',
                     mode: str = 'CHW'):
    """

    This function helps with multiple images to show without fixed seed.
    Total number of images to show is equal to num_rows * num_col.
    This only shows Images not boxes. Expected use on classification.

    IMPORTANT Keep it resonable num_row=10 num_col=10 is bare maximum higher numbers i don't want to imagine.

    :param num_rows: Number of rows you want
    :param num_col: Number of collumns you want
    :param fig_size: Multiplication of num_rows, num_col as fig size
    :param dataset: ImageFolder with transformation min. (ToTensor)
    :param class_names: String array of names
    :param cmap: Color map you want to default is gray
    :param mode: In what format you have your image C-Color H-Height W-Width
    :return: Plot images
    """
    OPTIONS = {
        'CHW': (1, 2, 0),
        'HCW': (0, 2, 1),
        'HWC': (0, 1, 2),
        'CWH': (2, 1, 0),
        'WCH': (2, 0, 1),
        'WHC': (1, 0, 2)
    }

    if num_rows * num_col > len(dataset):
        print(f"You have selected {num_col * num_rows} images to show but you have only {len(dataset)}.")
        return None

    fig = plt.figure(figsize=(fig_size * num_rows, fig_size * num_col))
    for i in range(1, num_rows * num_col + 1):
        random_idx = torch.randint(0, len(dataset), size=[1]).item()

        img, label = dataset[random_idx][0], dataset[random_idx][1]
        fig.add_subplot(num_rows, num_col, i)

        if img.ndim == 3:
            plt.imshow(img.permute(OPTIONS[mode]), cmap=cmap)
        elif img.ndim == 2:
            plt.imshow(img, cmap='gray')
        else:
            print("Incorrect shape of image")
            return None

        plt.title(class_names[label])
        plt.axis(False)

    plt.show()

