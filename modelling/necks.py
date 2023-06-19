import torch.nn as nn


def base_neck(model_num_features, emb_num_features, dropout=0.5):
    return nn.Sequential(
                    nn.BatchNorm1d(model_num_features),  # momentum=0.9, eps=1e-05),
                    nn.Dropout(dropout // 2),
                    nn.Linear(model_num_features, model_num_features // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(model_num_features // 2),
                    # check it out if it will not slow NET too much
                    nn.Dropout(dropout // 2),
                    nn.Linear(model_num_features // 2, emb_num_features),
                    nn.BatchNorm1d(emb_num_features)  # momentum=0.9, eps=1e-05)  # Paddle Paddle variant
    )
