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
        # x: (batch, seq_len, num_nodes)  此处为周期分量序列
        batch_size, seq_len, num_nodes = x.shape
        original_input = x

        # (B, L, N) -> (B, N, L)
        x = x.permute(0, 2, 1)
        x = self.node_encoder(x)          # (B, N, H)
        x = x.permute(0, 2, 1)            # (B, H, N)
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
        output = self.output_conv(recalibrated)  # (B, seq_len, N)

        return original_input + output           # 残差连接，输出修正后的周期分量


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

        # 注意：此处 refiner 用于修正周期分量（输入部分）
        self.cycle_refiner = STR(
            num_nodes=self.enc_in,
            seq_len=self.seq_len,
            refiner_dim=256
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

    def forward(self, x, cycle_index, return_residuals=False):
        cycle_index = cycle_index % self.cycle_len

        # 实例归一化
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # 获取原始周期分量
        current_cycle = self.cycleQueue(cycle_index, self.seq_len)   # (B, L, D)

        # 用 refiner 修正周期分量（预处理周期分量）
        refined_cycle = self.cycle_refiner(current_cycle)             # (B, L, D)

        # 计算残差
        residual_x = x - refined_cycle

        # 趋势预测（与原始 CycleNet 相同）
        residual_permuted = residual_x.permute(0, 2, 1)               # (B, D, L)
        future_residual_pred = self.trend_model(residual_permuted)    # (B, D, H)
        future_residual_pred = future_residual_pred.permute(0, 2, 1)  # (B, H, D)

        # 未来周期分量（不修正，直接使用原始周期队列）
        future_cycle_index = (cycle_index + self.seq_len) % self.cycle_len
        future_cycle = self.cycleQueue(future_cycle_index, self.pred_len)

        # 最终预测
        y = future_residual_pred + future_cycle

        # 实例反归一化
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        if return_residuals:
            return y, residual_x, refined_cycle   # 返回修正后的残差和周期分量供分析
        else:
            return y