import torch
import torch.nn as nn


class STR(nn.Module):
    """通用多尺度时序修正模块，可配置序列长度"""
    def __init__(self, num_nodes, seq_len, refiner_dim=256):
        super(STR, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.refiner_dim = refiner_dim

        self.node_encoder = nn.Linear(seq_len, refiner_dim)
        self.channel_adjust = nn.Conv1d(refiner_dim, refiner_dim, kernel_size=1)

        self.dilation_rates = [1, 2, 4, 8]
        self.dilated_convs = nn.ModuleList()
        for i, dilation in enumerate(self.dilation_rates):
            conv_block = nn.Sequential(
                nn.Conv1d(refiner_dim, refiner_dim, kernel_size=3,
                          padding=dilation, dilation=dilation,
                          padding_mode='replicate',
                          groups=min(refiner_dim, 8)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm1d(refiner_dim),
                nn.Dropout(0.03 + 0.02 * i)
            )
            self.dilated_convs.append(conv_block)

        self.adaptive_weights = nn.Parameter(torch.ones(len(self.dilation_rates)))

        self.feature_recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(refiner_dim, refiner_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(refiner_dim // 4, refiner_dim, 1),
            nn.Sigmoid()
        )

        self.output_conv = nn.Conv1d(refiner_dim, seq_len, kernel_size=1)

    def forward(self, x):
        # x: (batch, seq_len, num_nodes)
        batch_size, seq_len, num_nodes = x.shape
        original_input = x

        x = x.permute(0, 2, 1)          # (B, N, L)
        x = self.node_encoder(x)        # (B, N, H)
        x = x.permute(0, 2, 1)          # (B, H, N)
        x = self.channel_adjust(x)

        temporal_features = []
        current_x = x
        for i, dilated_conv in enumerate(self.dilated_convs):
            conv_output = dilated_conv(current_x)
            weighted_output = self.adaptive_weights[i] * conv_output
            temporal_features.append(weighted_output)
            current_x = conv_output

        if len(temporal_features) > 1:
            normalized_weights = torch.softmax(self.adaptive_weights, dim=0)
            fused = sum(normalized_weights[i] * temporal_features[i]
                        for i in range(len(temporal_features)))
        else:
            fused = temporal_features[0]

        attention_weights = self.feature_recalibrate(fused)
        recalibrated = fused * attention_weights
        output = self.output_conv(recalibrated)   # (B, seq_len, N)

        return original_input + output


class Model(nn.Module):
    """
    将 MultiScaleTemporalRefiner 作为独立预测模型。
    预测头支持 linear 或 mlp，与 CycleNet 配置方式一致。
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        self.refiner_dim = getattr(configs, 'refiner_dim', 256)
        
        

        # 核心 Refiner 模块
        self.refiner = STR(
            num_nodes=self.enc_in,
            seq_len=self.seq_len,
            refiner_dim=self.refiner_dim
        )

        # 预测头：将 seq_len 映射到 pred_len，对每个变量独立操作（作用于 permute 后的 (B, C, seq_len)）
        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.predictor = nn.Linear(self.seq_len, self.pred_len)
        else:
            self.predictor = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model, self.pred_len)
            )

    def forward(self, x, cycle_index=None):
        # x: (batch, seq_len, enc_in)
        batch_size = x.shape[0]

        # RevIN 归一化（可选）
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)          # (B, 1, C)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # Refiner 精炼输入
        refined_x = self.refiner(x)                    # (B, seq_len, C)

        # 预测：转置为 (B, C, seq_len) 通过预测头，再转置回 (B, pred_len, C)
        refined_x = refined_x.permute(0, 2, 1)         # (B, C, seq_len)
        y = self.predictor(refined_x)                  # (B, C, pred_len)
        y = y.permute(0, 2, 1)                         # (B, pred_len, C)

        # 逆归一化
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y


