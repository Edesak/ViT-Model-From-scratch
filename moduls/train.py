import torch.cuda
import data_prep,models_builder,engine,utils
from torchvision import transforms
from torchmetrics.classification import MulticlassAccuracy
import argparse
import wandb



def main():
    NUM_EPOCHS = args.num_epochs if args.num_epochs is not None else 5
    BATCH_SIZE = args.batch_size if args.batch_size is not None else 32
    HIDDEN_UNITS = args.hidden_units if args.hidden_units is not None else 10
    LEARNING_RATE = args.learning_rate if args.learning_rate is not None else 0.001
    WANDB = args.wandb if args.wandb is not None else False
    print(WANDB)
    if WANDB:
        wandb.init(
            # set the wandb project where this run will be logged
            project="Base_line_tinyVGG_04",
            name=f"{NUM_EPOCHS}_epochs_without_aug",
            # track hyperparameters and run metadata
            config={

                "learning_rate": LEARNING_RATE,
                "architecture": "tinyVGG",
                "dataset": "CustomDataset_Food101_10%_3classes",
                "epochs": NUM_EPOCHS,
            }
        )

    dir = "D:\PycharmProjects\LearningPytorch\data\pizza_steak_sushi"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transforms = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])

    train_dataloader,test_dataloader,class_names = Data_prep.data_prep_imgfolder(path=dir,
                                                                                 batch_size=BATCH_SIZE,
                                                                                 transforms=(train_transforms,test_transforms))

    model = models.Tiny_VGG(in_features=3,hidden_units=HIDDEN_UNITS,out_features=len(class_names))

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=LEARNING_RATE)

    acc_fn = MulticlassAccuracy(num_classes=len(class_names))

    engine.train(model,device,train_dataloader,test_dataloader,loss_fn,acc_fn,optimizer,NUM_EPOCHS,WANDB)

    utils.save_model_dict(model=model,
                          target_dir="models",
                          model_name="D:\PycharmProjects\LearningPytorch\models\going_modular_lol.pth")

def str2bool(v):
    return v.lower() in ('true', '1')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start train with args")
    parser.add_argument("--num_epochs",help="set number of epochs",type=int)
    parser.add_argument("--batch_size", help="set batch size",type=int)
    parser.add_argument("--hidden_units", help="set number of hidden units",type=int)
    parser.add_argument("--learning_rate", help="set learning rate",type=float)
    parser.add_argument("--wandb", help="set bool of wandb", type=str2bool, nargs='?', const=True, default=False)
    args = parser.parse_args()
    print(args.num_epochs)
    print(args.wandb)
    main()