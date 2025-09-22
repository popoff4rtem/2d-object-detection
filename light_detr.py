"""
Lightweight DETR for a single GPU (e.g. RTX 4070).
- ResNet-18 backbone (optionally pretrained)
- hidden_dim=128, nheads=4, encoder_layers=2, decoder_layers=2
- num_queries default 50
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter

# -----------------------
# Positional embedding (sine)
# -----------------------
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale or 2 * math.pi

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        mask = torch.zeros(B, H, W, dtype=torch.bool, device=x.device)
        y_embed = torch.cumsum(~mask, dim=1).float()
        x_embed = torch.cumsum(~mask, dim=2).float()
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0,3,1,2)
        return pos


# -----------------------
# MLP for box regression
# -----------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------
# Lightweight DETR
# -----------------------
class LightweightDETR(nn.Module):
    def __init__(self,
                 num_classes,
                 num_queries=50,
                 hidden_dim=128,
                 nheads=4,
                 num_encoder_layers=2,
                 num_decoder_layers=2,
                 dim_feedforward=None,
                 backbone_pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Backbone: ResNet-18, возвращаем слой layer4
        # backbone = resnet18(pretrained=backbone_pretrained)  # ResNet-18
        backbone = resnet50(pretrained=backbone_pretrained) # ResNet-50
        return_layers = {'layer4': '0'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # self.input_proj = nn.Conv2d(512, hidden_dim, kernel_size=1) # ResNet-50
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1) # ResNet-50

        # Positional embedding
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2)

        # Transformer: use PyTorch nn.Transformer (encoder-decoder)
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4
        self.transformer = nn.Transformer(d_model=hidden_dim,
                                          nhead=nheads,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=0.1,
                                          batch_first=False)  # we'll feed (S,B,C) etc

        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # init
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.query_embed.weight, std=0.02)
        nn.init.constant_(self.class_embed.bias, 0)
        # bbox head small init
        for p in self.bbox_embed.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        x: tensor [B,3,H,W]
        returns:
            dict{
              'pred_logits': [B, Q, num_classes+1],
              'pred_boxes' : [B, Q, 4]
            }
        """
        B = x.shape[0]
        feat = self.backbone(x)['0']  # [B, 512, H/32, W/32] for resnet18
        src = self.input_proj(feat)   # [B, hidden, Hs, Ws]
        pos = self.pos_enc(src)       # [B, hidden, Hs, Ws]

        # flatten (S, B, C) for transformer
        B, C, Hs, Ws = src.shape
        src_flat = src.flatten(2).permute(2, 0, 1)   # [S, B, C], S = Hs*Ws
        pos_flat = pos.flatten(2).permute(2, 0, 1)   # [S, B, C]

        # memory: encoder
        memory = self.transformer.encoder(src_flat + pos_flat)  # [S, B, C]

        # prepare queries: [Q, B, C]
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [Q, B, C]

        # decoder: queries (tgt) shape [T, B, C], memory [S,B,C]
        hs = self.transformer.decoder(queries, memory)  # [Q, B, C]
        hs = hs.permute(1, 0, 2)  # [B, Q, C]

        outputs_class = self.class_embed(hs)            # [B, Q, num_classes+1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # [B, Q, 4], normalized cx,cy,w,h

        return {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
