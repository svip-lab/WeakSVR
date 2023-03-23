import torch.nn as nn
from models.vit.vit_pytorch import ViT


class Builder:
    def __init__(self,
                 num_clip=16,
                 pretrain=None,
                 use_TE=False,
                 use_gumbel=False,
                 dim_embedding=128):
        self.num_clip = num_clip
        self.pretrain = pretrain
        self.use_TE = use_TE
        self.dim_embedding = dim_embedding
        self.use_gumbel = use_gumbel
        self.input_dim = 512 if self.use_gumbel else 1024

    def build_transformer_encoder(self):
        return ViT(
            image_size=(6, 10 * self.num_clip),
            patch_size=(6, 10),
            dim=1024,
            depth=2,
            heads=8,
            mlp_dim=2048,
            pool='all',
            channels=128
        )

    def build_seq_features_extractor(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            Reshape(-1, self.num_clip, 2048)
        )

    def build_embed_head(self):
        if self.use_TE:
            # Backbone output dim of Transformer Encoder: 1024
            return nn.Sequential(
                Reshape(-1, self.num_clip * self.input_dim),
                nn.Linear(self.num_clip * self.input_dim, self.dim_embedding)
            )
        else:
            # Backbone output dim of ResNet50: 2048
            return nn.Sequential(
                Reshape(-1, self.num_clip * 2048),
                nn.Linear(self.num_clip * 2048, self.dim_embedding)
            )


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.reshape(self.shape)
