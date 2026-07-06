# simple_camera_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from vggt.heads.head_act import activate_pose

class SimpleCameraMLP(nn.Module):
    def __init__(self, dim_in=2048, pose_dim=9, hidden_dim=512,
                 trans_act="linear", quat_act="linear", fl_act="relu"):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim_in)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pose_dim),
        )
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act

    def forward(self, aggregated_tokens_list):
        # 和 CameraHead 一樣取最後一層的 camera token
        tokens = aggregated_tokens_list[-1]      # [B, S, T, C]
        pose_tokens = tokens[:, :, 0]            # [B, S, C]
        pose_tokens = self.token_norm(pose_tokens)
        pred_pose = self.mlp(pose_tokens)
        return activate_pose(
            pred_pose,
            trans_act=self.trans_act,
            quat_act=self.quat_act,
            fl_act=self.fl_act,
        )