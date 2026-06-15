import torch
import torch.nn as nn


class RecurrentCycle(torch.nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        index = index % self.cycle_len
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device)) % self.cycle_len
        return self.data[gather_index]


class STR(nn.Module):
    """STR模块"""
    def __init__(self, num_nodes, seq_len, refiner_dim=256):
        super(STR, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.refiner_dim = refiner_dim

        self.node_encoder = nn.Linear(seq_len, refiner_dim)
        self.channel_adjust = nn.Conv1d(refiner_dim, refiner_dim, kernel_size=1)

        self.dilation_rates = [1,2,4,8]
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
        # 原有 forward 逻辑：返回 x + correction
        correction = self.compute_correction(x)
        return x + correction

    def compute_correction(self, x):
        # 输入 x: (batch, seq_len, num_nodes)
        batch_size, seq_len, num_nodes = x.shape
        x_enc = x.permute(0, 2, 1)          # (B, N, L)
        x_enc = self.node_encoder(x_enc)    # (B, N, H)
        x_enc = x_enc.permute(0, 2, 1)      # (B, H, N)
        x_enc = self.channel_adjust(x_enc)

        temporal_features = []
        current_x = x_enc
        for i, dilated_conv in enumerate(self.dilated_convs):
            conv_output = dilated_conv(current_x)
            weighted_output = self.adaptive_weights[i] * conv_output
            temporal_features.append(weighted_output)
            current_x = conv_output

        if len(temporal_features) > 1:
            normalized_weights = torch.softmax(self.adaptive_weights, dim=0)
            fused = sum(normalized_weights[i] * temporal_features[i] for i in range(len(temporal_features)))
        else:
            fused = temporal_features[0]

        attention_weights = self.feature_recalibrate(fused)
        recalibrated = fused * attention_weights
        correction = self.output_conv(recalibrated)   # (B, seq_len, N)
        return correction

        return original_input + output


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        # ----- STR模块（作用于预测结果）-----
        self.output_refiner = STR(
            num_nodes=self.enc_in,
            seq_len=self.pred_len,                       
            refiner_dim=getattr(configs, 'refiner_dim', 256)
        )

        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.trend_model = nn.Linear(self.seq_len, self.pred_len)
        else:
            self.trend_model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model, self.pred_len)
            )

    def forward(self, x, cycle_index, return_components=False):
        cycle_index = cycle_index % self.cycle_len

        # 实例归一化
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        
        current_cycle = self.cycleQueue(cycle_index, self.seq_len)
        residual_x = x - current_cycle

        residual_permuted = residual_x.permute(0, 2, 1)          # (B, D, L)
        future_residual_pred = self.trend_model(residual_permuted)  # (B, D, H)
        future_residual_pred = future_residual_pred.permute(0, 2, 1)  # (B, H, D)

        future_cycle_index = (cycle_index + self.seq_len) % self.cycle_len
        future_cycle = self.cycleQueue(future_cycle_index, self.pred_len)

        y_raw = future_residual_pred + future_cycle   # 初步预测 (B, pred_len, D)

        # ----- 后置多尺度通道修正-----
        correction = self.output_refiner.compute_correction(y_raw)  
        y = y_raw + correction                # 残差连接

        # 逆归一化
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        if return_components:
            return y, y_raw, correction
        else:
            return y