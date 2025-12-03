# model.py
import torch
import torch.nn as nn
import timm

class SequentialPoseTransformer(nn.Module):
    def __init__(self, num_pose_outputs=7, sequence_length=5):
        super().__init__()
        
        # Aquí lo adapta a 128 px (Debido a las capacidades computacionales)
        self.feature_extractor = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=True, 
            num_classes=0, 
            img_size=128
        )
        embed_dim = self.feature_extractor.embed_dim
        
        # Transformer temporal
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # El CABEZAL FINAL que predice la pose
        self.pose_head = nn.Linear(embed_dim, num_pose_outputs)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)
        features = features.view(batch_size, seq_len, -1)
        temporal_output = self.temporal_transformer(features)
        last_item_output = temporal_output[:, -1, :] 
        pose = self.pose_head(last_item_output)
        
        return pose
