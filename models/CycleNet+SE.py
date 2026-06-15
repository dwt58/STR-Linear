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


import torch
import torch.nn as nn


class STR(nn.Module):
    
    def __init__(self, num_nodes, seq_len, refiner_dim=256):
        super(STR, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.refiner_dim = refiner_dim

        # 1. 节点编码器：将时序预测投影到高维特征空间
        self.node_encoder = nn.Linear(seq_len, refiner_dim)

        # 2. SE 注意力重标定模块（唯一保留的跨通道操作）
        self.feature_recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),                      # (B, H, N) → (B, H, 1)
            nn.Conv1d(refiner_dim, refiner_dim // 4, 1),   # 降维
            nn.ReLU(inplace=True),
            nn.Conv1d(refiner_dim // 4, refiner_dim, 1),   # 升维
            nn.Sigmoid()                                   # 生成 0~1 权重
        )

        # 3. 输出投影层：将特征映射回原始预测长度
        self.output_conv = nn.Conv1d(refiner_dim, seq_len, kernel_size=1)

    def forward(self, x):
        # x: (batch, seq_len, num_nodes)
        original_input = x

        # 编码：时序压缩
        x = x.permute(0, 2, 1)          # (B, N, L)
        x = self.node_encoder(x)        # (B, N, H)
        x = x.permute(0, 2, 1)          # (B, H, N)

        # SE 注意力重标定
        attention_weights = self.feature_recalibrate(x)  # (B, H, 1)
        x = x * attention_weights                        # 通道级加权

        # 输出投影
        output = self.output_conv(x)    # (B, seq_len, N)

        # 残差连接
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

        # ----- 后置多尺度通道维度修正模块（作用于预测结果）-----
        self.output_refiner = STR(
            num_nodes=self.enc_in,
            seq_len=self.pred_len,                       # 注意：此处序列长度改为预测长度
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

    def forward(self, x, cycle_index, return_debug=False):
        cycle_index = cycle_index % self.cycle_len

        # 实例归一化
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # 原始 CycleNet 流程（无前置 refiner）
        current_cycle = self.cycleQueue(cycle_index, self.seq_len)
        residual_x = x - current_cycle

        residual_permuted = residual_x.permute(0, 2, 1)          # (B, D, L)
        future_residual_pred = self.trend_model(residual_permuted)  # (B, D, H)
        future_residual_pred = future_residual_pred.permute(0, 2, 1)  # (B, H, D)

        future_cycle_index = (cycle_index + self.seq_len) % self.cycle_len
        future_cycle = self.cycleQueue(future_cycle_index, self.pred_len)

        y_raw = future_residual_pred + future_cycle   # 初步预测 (B, pred_len, D)

        # ----- 后置多尺度通道修正（精炼预测结果）-----
        y = self.output_refiner(y_raw)                # 残差连接已在 refiner 内部完成

        # 逆归一化
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        if return_debug:
            return y, {'raw_pred': y_raw, 'residual': residual_x}
        else:
            return y 