import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from moduls import utils
import zipfile
from pathlib import Path
import os
from typing import Tuple, List

NUM_WORKERS = os.cpu_count()


def data_prep_imgfolder(path: str, batch_size: int = 32, transforms: Tuple[transforms.Compose, transforms.Compose] = None,
                        show_imgs: bool = False, num_plots: int = 2) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Creating datalaoader from datset with ImageFolder function (expected structure to have data main folder -> train/test -> labels -> images)


    :param path: string path of your main folder
    :param batch_size: set batchsize you want in dataloader
    :param transforms: This expects tuple of Compose in order (train_compose,test_compose)
    :param show_imgs: Select if you want to show images from dataset
    :param num_plots: if you want to show images how many times you want retake ploting random images (one plot is 9 images)
    :return: tuple(train_dataloader,test_dataloader,class_names) where class_names is List

    Example:

    from multiprocessing import freeze_support

    from Data_prep import Dataprep

    if __name__ == '__main__':
        freeze_support()

        train_dataloader,test_dataloader,class_names = Dataprep(path=dir_path,transform=(train_transform,test_transform))
    """

    image_path = Path(path)
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    train_transform, test_transform = transforms

    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=train_transform,  # data transform
                                      target_transform=None)  # Label transform
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=test_transform,
                                     target_transform=None)

    class_names = train_data.classes
    if show_imgs:
        for i in range(num_plots):
            utils.plot_images_clas(dataset=train_data, class_names=class_names)

    train_data_Dataloader = DataLoader(dataset=train_data,
                                       batch_size=batch_size,
                                       num_workers=0,
                                       shuffle=True,
                                       pin_memory=True)
    test_data_Dataloader = DataLoader(dataset=test_data,
                                      batch_size=batch_size,
                                      num_workers=0,
                                      shuffle=False,
                                      pin_memory=True)

    return train_data_Dataloader, test_data_Dataloader, class_names


def data_prep(train_path: str, test_path: str, batch_size: int = 32, transforms: Tuple = None, show_imgs: bool = False,
              num_plots: int = 2):
    """
    ***WIP***
    Creating datalaoader from datset with ImageFolder function (expected structure to have data main folder -> train/test -> labels -> images)
    Usefull if you change training data but want to be test data the same for experiment reasons

    If transforms is None it create default transform. Resize to (64,64) then ToTensor()

    :param train_path: Takes path in str to your train path
    :param test_path: Takes path in str to your test path
    :param batch_size: set batchsize you want in dataloader
    :param transform: This expects tuple of Compose in order (train_compose,test_compose)
    :param show_imgs: Select if you want to show images from dataset
    :param num_plots: if you want to show images how many times you want retake ploting random images (one plot is 9 images)
    :return: tuple(train_dataloader,test_dataloader,class_names) where class_names is List

    Example:

    from multiprocessing import freeze_support

    from Data_prep import Dataprep

    if __name__ == '__main__':
        freeze_support()

        train_dataloader,test_dataloader,class_names = Dataprep(path=dir_path,transform=(train_transform,test_transform))
    """

    # image_path = Path(path)
    train_dir = train_path
    test_dir = test_path

    if transforms is not None:
        train_transform, test_transform = transforms
    else:
        default_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(64,64)),
            torchvision.transforms.ToTensor()
        ])
        train_transform,test_transform = default_transform,default_transform


    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=train_transform,  # data transform
                                      target_transform=None)  # Label transform
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=test_transform,
                                     target_transform=None)

    class_names = train_data.classes
    if show_imgs:
        for i in range(num_plots):
            utils.plot_images_clas(dataset=train_data, class_names=class_names)

    train_data_Dataloader = DataLoader(dataset=train_data,
                                       batch_size=batch_size,
                                       num_workers=0,
                                       shuffle=True,
                                       pin_memory=True)
    test_data_Dataloader = DataLoader(dataset=test_data,
                                      batch_size=batch_size,
                                      num_workers=0,
                                      shuffle=False,
                                      pin_memory=True)

    return train_data_Dataloader, test_data_Dataloader, class_names


def unzip(zip_path: str, dir_path: str):
    """
    Simple function to unzip files
    :param zip_path: path to zip file
    :param dir_path: path to directory where you want to extract
    :return:
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zip_ref:
        zip_ref.extractall(dir_path)
