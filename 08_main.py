import matplotlib.pyplot as plt
import torch, torchvision
from moduls import utils, engine, data_prep, Helper_functions, models_builder
from torchvision import transforms
from torch import nn
from einops.layers.torch import Rearrange
from torchinfo import summary
from torchmetrics.classification import MulticlassAccuracy
from moduls.engine import train
import wandb
from moduls import utils
from moduls.models_builder import ViTModel
from torchvision.models import vit_b_16, ViT_B_16_Weights

train_dir, test_dir = "data/pizza_steak_sushi/train", "data/pizza_steak_sushi/test"

device = utils.setup_device()
utils.setup_seed()

IMG_SIZE = 224
BATCH_SIZE = 32

PATCH_SIZE = 16
COLOR = 3
EMBEDDING = PATCH_SIZE ** 2 * COLOR

EPOCHS = 10
LOG_WANDB = False
if LOG_WANDB:

    wandb.init(
        # set the wandb project where this run will be logged
        project="Vit_Replication",
        name=f"{EPOCHS}_epochs_ViT_TransferLearning",
        # track hyperparameters and run metadata
        config={

            "learning_rate": 0.001,
            "architecture": "ViT",
            "dataset": "CustomDataset_Food101_20%_3classes",
            "epochs": EPOCHS,
        }
    )
utils.setup_seed()

manual_transforms = transforms.Compose([
    transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_prep.data_prep(train_dir, test_dir,
                                                                     transforms=(manual_transforms, manual_transforms),
                                                                     batch_size=BATCH_SIZE, show_imgs=False)

def train_custom_Vit_model():
    vit_model = ViTModel()
    acc_fn = MulticlassAccuracy(num_classes=3)
    optim = torch.optim.Adam(params=vit_model.parameters(),lr=0.001,weight_decay=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    results = train(vit_model,device,train_dataloader,test_dataloader,loss_fn,acc_fn,optim,epochs=EPOCHS,log_to_wandb=LOG_WANDB)
    print(results)

torch.hub.set_dir("models")
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)
transformer = weights.transforms()

train_dataloader, test_dataloader, class_names = data_prep.data_prep(train_dir, test_dir,
                                                                     transforms=(transformer, transformer),
                                                                     batch_size=BATCH_SIZE, show_imgs=False)


summary(model,input_size=(32,3,224,224),col_names=["input_size","output_size","trainable"])
for layer in model.parameters():
    layer.requires_grad = False
    model.heads = nn.Sequential(
        nn.Linear(in_features=768, out_features=3)
    )

summary(model,input_size=(32,3,224,224),col_names=["input_size","output_size","trainable"])



acc_fn = MulticlassAccuracy(num_classes=3)
optim = torch.optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
results = train(model,device,train_dataloader,test_dataloader,loss_fn,acc_fn,optim,epochs=EPOCHS,log_to_wandb=LOG_WANDB)

utils.pred_plot_image(model,"data/pizza_steak_sushi/testing/Pizza_test.jpg",class_names,device,transform=transformer)