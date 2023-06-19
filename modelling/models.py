from torch import nn
import torch
import timm
import pytorch_lightning as pl
import clip
from .necks import base_neck


class BaseModelConstruct(pl.LightningModule):
    """ Class that takes any backbone and adds metric learning output layer
    Converts list of images in the format of numpy arrays  -> to a torch normalized tensors
    """

    def __init__(self,
                 backbone,
                 neck,
                 head=None,
                 normalize=True,
                 freeze_backbone=False,
                 dropout=0.5,
                 num_features=128):

        super(BaseModelConstruct, self).__init__()
        self.normalize = normalize
        self.num_features = num_features
        self.dropout = dropout
        self.backbone = backbone
        self.neck = neck

        # pml loss constructs head itself
        self.head = head

        # freeze layers
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

    def forward(self, x):

        x = self.backbone(x)
        x = self.neck(x.type(self.neck[0].weight.dtype))  # clip workaround
        if self.head is not None:
            x = self.head(x)
        if self.normalize:
            x = nn.functional.normalize(x, p=2, dim=1)

        return x


def baseline_backbone():
    return timm.create_model('hrnet_w18', pretrained=True, num_classes=0)


def baseline_backbone_neck(model_n_features=2048, emb_n_features=128):
    return BaseModelConstruct(backbone=baseline_backbone(),
                              neck=base_neck(model_n_features, emb_n_features),
                              num_features=emb_n_features,
                              normalize=True)


### VIT RELATED
def clip_vit16b():
    model, _ = clip.load("ViT-B/16", jit=False)
    return model


class ImageEncoder(torch.nn.Module):
    def __init__(self, keep_lang=False):
        super().__init__()

        self.model = clip_vit16b()
        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        return self.model.encode_image(images)


def clip_vitb16_with_neck(model_n_features=512, emb_n_features=128):
    return BaseModelConstruct(backbone=ImageEncoder(),
                              neck=base_neck(model_n_features, emb_n_features),
                              num_features=emb_n_features,
                              freeze_backbone=True,
                              normalize=True)
