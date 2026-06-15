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
    def __init__(self, num_nodes, seq_len, refiner_dim=512):
        super(STR, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.refiner_dim = refiner_dim

        
        self.node_encoder = nn.Linear(seq_len, refiner_dim)

        # 通道调整层
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

        # 通道注意力重标定
        self.feature_recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(refiner_dim, refiner_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(refiner_dim // 4, refiner_dim, 1),
            nn.Sigmoid()
        )

        # 输出卷积，映射回原始序列长度
        self.output_conv = nn.Conv1d(refiner_dim, seq_len, kernel_size=1)

    def forward(self, residual_x):
        batch_size, seq_len, num_nodes = residual_x.shape
        original_residual = residual_x

        # 形状变换: (B, L, N) -> (B, N, L)
        x = residual_x.permute(0, 2, 1)  # (B, N, L)

        # 节点编码：将 L 维映射到 H 维
        x = self.node_encoder(x)        # (B, N, H)

        # 转换为 (B, H, N) 便于卷积处理
        x = x.permute(0, 2, 1)          # (B, H, N)

        # 通道调整
        x = self.channel_adjust(x)      # (B, H, N)

        
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

        # 通道注意力重标定
        attention_weights = self.feature_recalibrate(fused)
        recalibrated = fused * attention_weights

        # 输出映射回原始序列长度
        x = self.output_conv(recalibrated)  # (B, seq_len, N)

        # 残差连接
        return original_residual + x


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

        # 周期队列
        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        
        self.residual_refiner = STR(
            num_nodes=self.enc_in,
            seq_len=self.seq_len,
            refiner_dim=512
        )

        # 趋势预测模型
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

        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        current_cycle = self.cycleQueue(cycle_index, self.seq_len)
        residual_x = x - current_cycle
        refined_residual = self.residual_refiner(residual_x)
        refined_residual_permuted = refined_residual.permute(0, 2, 1)
        future_residual_pred = self.trend_model(refined_residual_permuted)
        future_residual_pred = future_residual_pred.permute(0, 2, 1)
        future_cycle_index = (cycle_index + self.seq_len) % self.cycle_len
        future_cycle = self.cycleQueue(future_cycle_index, self.pred_len)
        y = future_residual_pred + future_cycle

        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        if return_residuals:
            return y, residual_x, refined_residual
        else:
            return y