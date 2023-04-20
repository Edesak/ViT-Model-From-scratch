from torch import nn


def hello():
    print("Hello this is models_builder.py file.")


class FashionModelV1(nn.Module):
    """
    NonLinear model without any Convolutional layers
    Input shape: is number after Flatten layer
    Example: IMG size 64x64 -> input shape 64*64*Color
    """

    def __init__(self,
                 in_features: int,
                 hidden_units: int,
                 out_features: int):
        super().__init__()
        self.stacked_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=out_features),
            nn.ReLU()
        )

    def forward(self, x):
        return self.stacked_layers(x)


class FashionCNNV2(nn.Module):
    """
    Classic CNN from TinyVGG with little to none modifications.
    Hard coded Flatten->Linear layer shape
    Expected size: 64x64
    """

    def __init__(self,
                 in_features: int,
                 hidden_units: int,
                 out_features: int):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7,
                      out_features=out_features)
        )

    def forward(self, x):
        y = self.conv_block1(x)
        # print(y.shape)
        y = self.conv_block2(y)
        # print(y.shape)
        # print(self.conv_block2.parameters())
        y = self.classifier(y)
        return y


class BaselineModel(nn.Module):
    """
    Linear model without Convolutional layers
    Input shape: is number after Flatten layer
    Example: IMG size 64x64 -> input shape 64*64*Color
    """

    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)


class TinyVGG(nn.Module):
    """
    Replicated Tiny VGG model witout modifications
    Hard coded Flatten->Linear layer shape
    Expected size: 64x64
    """

    def __init__(self,
                 in_features,
                 hidden_units,
                 out_features):
        super().__init__()

        self.block_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=hidden_units, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )
        self.block_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 13 * 13, out_features=out_features)

        )

    def forward(self, x):
        return self.block_classifier(self.block_conv2(self.block_conv1(x)))


class PatchEmbedding(nn.Module):
    def __init__(self,
                 image_size: int = 224,
                 embedding: int = 768,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 batch_size: int = 32,
                 dropout=0.1):
        super().__init__()
        num_patches = (image_size * image_size) // patch_size ** 2
        self.patch_size = patch_size
        self.patch_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embedding, kernel_size=patch_size, stride=patch_size,
                      padding=0),
            Rearrange('b f h w -> b (h w) f')
        )
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding), requires_grad=True)
        self.position_token = nn.Parameter(torch.randn(1, num_patches + 1, embedding), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        image_res = x.shape[-1]
        assert image_res % self.patch_size == 0, f"Bad image shape {image_res, image_res} or patch size {self.patch_size} resulting in residue {image_res % self.patch_size} should be 0"

        y = self.patch_layer(x)
        class_token = self.class_token.expand(x.shape[0],-1,-1)
        y = torch.cat((class_token, y), dim=1)

        y = y + self.position_token
        y = self.dropout(y)
        return y


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 dropout: float = 0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.msa = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout,
                                         batch_first=True)

    def forward(self, x):
        y = self.layer_norm(x)
        y, _ = self.msa(query=y, key=y, value=y)
        return y


class MLPBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 hidden_units: int = 3072,
                 dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=hidden_units),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_units, out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        y = self.mlp(x)
        return y


class TransformerEncoder(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 hidden_units: int = 3072,
                 mpl_dropout: float = 0.1,
                 msa_dropout: float = 0):
        super().__init__()

        self.msa = MultiHeadAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads, dropout=msa_dropout)
        self.mlp = MLPBlock(embedding_dim=embedding_dim, hidden_units=hidden_units, dropout=mpl_dropout)

    def forward(self, x):
        y_msa = self.msa(x)
        y_msa = y_msa + x
        y_mlp = self.mlp(y_msa)
        y_mlp = y_mlp + y_msa
        return y_mlp


class ViTModel(nn.Module):
    """
    Replicated ViT model from paper. It is not optimized with Pytorch transfer that is custom.
    """
    def __init__(self,
                 image_size: int = 224,
                 batch_size: int = 32,
                 embedding: int = 768,
                 patch_size: int = 16,
                 num_transform_layers: int = 12,
                 in_channels: int = 3,
                 num_heads: int = 12,
                 dropout_msa: float = 0.0,
                 hidden_units: int = 3072,
                 dropout_mlp: float = 0.1,
                 embedding_dropout=0.1,
                 num_classes: int = 3
                 ):
        super().__init__()

        self.embed = PatchEmbedding(image_size=image_size, batch_size=batch_size, embedding=embedding,
                                    patch_size=patch_size, in_channels=in_channels, dropout=embedding_dropout)
        self.encoder = nn.Sequential(
            *[TransformerEncoder(embedding_dim=embedding, num_heads=num_heads, hidden_units=hidden_units,
                                 mpl_dropout=dropout_mlp, msa_dropout=dropout_msa) for _ in range(num_transform_layers)]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding),
            nn.Linear(in_features=embedding, out_features=num_classes)
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        x = self.classifier(x[:,0])
        return x

