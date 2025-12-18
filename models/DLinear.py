import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric.nn as gnn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class SpatialTemporalGNN(nn.Module):
    """增强的空间-时序联合建模模块（作为趋势修正器）- 保持你的原始实现"""
    def __init__(self, num_nodes, seq_len, gnn_hidden=256, graph_type='ring'):
        super(SpatialTemporalGNN, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        
        # 1. 空间建模层
        self.spatial_conv = gnn.SAGEConv(seq_len, gnn_hidden)
        
        # 2. 时序建模层 - 基于原始[1,2,4,8]序列的精细化优化
        # 首先使用1x1卷积调整通道数
        self.channel_adjust = nn.Conv1d(
            in_channels=gnn_hidden,
            out_channels=gnn_hidden,
            kernel_size=1
        )
        
        # 回到原始最优的空洞率序列
        self.dilation_rates = [1, 2, 4, 8]  # 恢复为经典的指数序列
        
        # 精细化设计的空洞卷积块
        self.dilated_convs = nn.ModuleList()
        for i, dilation in enumerate(self.dilation_rates):
            conv_block = nn.Sequential(
                # 使用分组卷积减少参数量，增强泛化能力
                nn.Conv1d(
                    in_channels=gnn_hidden,
                    out_channels=gnn_hidden,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    padding_mode='replicate',
                    groups=min(gnn_hidden, 8)  # 分组卷积，平衡表达能力和效率
                ),
                nn.LeakyReLU(0.1, inplace=True),  # 使用inplace节省内存
                nn.BatchNorm1d(gnn_hidden),
                # 针对不同层使用不同的dropout率
                nn.Dropout(0.03 + 0.02 * i)  # 深层使用稍大的dropout
            )
            self.dilated_convs.append(conv_block)
        
        # 简化的自适应权重机制
        self.adaptive_weights = nn.Parameter(torch.ones(len(self.dilation_rates)))
        
        # 轻量级特征重校准
        self.feature_recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化
            nn.Conv1d(gnn_hidden, gnn_hidden // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(gnn_hidden // 4, gnn_hidden, 1),
            nn.Sigmoid()
        )
        
        # 最终输出层
        self.output_conv = nn.Conv1d(
            in_channels=gnn_hidden,
            out_channels=seq_len,
            kernel_size=1
        )
        
        # 图结构选择
        self.graph_type = graph_type
        if graph_type == 'adaptive':
            self.edge_weights = nn.Parameter(torch.randn(num_nodes, num_nodes))
        else:
            self.register_buffer('edge_index', self.create_graph_structure(num_nodes))
    
    def create_graph_structure(self, num_nodes):
        """创建环形图结构"""
        src, dst = [], []
        for i in range(num_nodes):
            neighbors = [(i + j) % num_nodes for j in [-1, 1] if j != 0]
            for n in neighbors:
                src.append(i)
                dst.append(n)
        return torch.tensor([src, dst], dtype=torch.long)
    
    def forward(self, trend_component):
        """
        输入: [batch_size, seq_len, num_nodes] - 趋势成分
        输出: [batch_size, seq_len, num_nodes] - 修正后的趋势成分
        """
        batch_size, seq_len, num_nodes = trend_component.shape
        device = trend_component.device
        
        # 保存原始趋势用于后续连接
        original_trend = trend_component
        
        # 转换维度 [batch_size, num_nodes, seq_len]
        x = trend_component.permute(0, 2, 1)
        
        # 空间建模
        if self.graph_type == 'adaptive':
            adj = torch.softmax(self.edge_weights, dim=-1)
            x = torch.matmul(adj, x)
        else:
            if self.edge_index.device != device:
                self.edge_index = self.edge_index.to(device)
            
            # 重塑为图卷积需要的格式 [batch_size * num_nodes, seq_len]
            x_flat = x.reshape(-1, seq_len)
            
            # 为每个batch复制图结构
            edge_list = []
            for i in range(batch_size):
                offset = i * num_nodes
                src = self.edge_index[0] + offset
                dst = self.edge_index[1] + offset
                edge_list.append(torch.stack([src, dst]))
            edge_index = torch.cat(edge_list, dim=1).to(device)
            
            # 应用空间图卷积
            x_flat = self.spatial_conv(x_flat, edge_index)
            x = x_flat.view(batch_size, num_nodes, -1)  # [batch_size, num_nodes, gnn_hidden]
        
        # 时序建模
        # 调整维度: [batch_size, num_nodes, gnn_hidden] -> [batch_size, gnn_hidden, num_nodes]
        x = x.permute(0, 2, 1)
        
        # 调整通道数
        x = self.channel_adjust(x)
        
        # 保存输入用于特征重校准
        temporal_features = []
        current_x = x
        
        # 应用空洞卷积块
        for i, dilated_conv in enumerate(self.dilated_convs):
            conv_output = dilated_conv(current_x)
            # 使用自适应权重
            weighted_output = self.adaptive_weights[i] * conv_output
            temporal_features.append(weighted_output)
            current_x = conv_output  
        
        # 多尺度特征融合
        if len(temporal_features) > 1:
            # 加权融合
            fused = sum(temporal_features) / len(temporal_features)
        else:
            fused = temporal_features[0]
        
        # 特征重校准
        attention_weights = self.feature_recalibrate(fused)
        recalibrated = fused * attention_weights
        
        # 最终输出层
        x = self.output_conv(recalibrated)
        
        # 残差连接 - 保持趋势的整体特性
        return original_trend + x


class TrendRefinedDLinear(nn.Module):
    """
    只修正趋势部分的DLinear改进版本
    使用STGNN专门修正趋势成分，季节性成分保持不变
    """
    def __init__(self, configs):
        super(TrendRefinedDLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.individual = getattr(configs, 'individual', False)
        
        # 分解核大小
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        
        # STGNN趋势修正模块
        self.trend_refiner = SpatialTemporalGNN(
            num_nodes=self.enc_in,
            seq_len=self.seq_len,
            gnn_hidden=getattr(configs, 'gnn_hidden', 256),                                                          #!!!!
            graph_type=getattr(configs, 'graph_type', 'ring')
        )
        
        # 预测网络（保持DLinear原有结构）
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.enc_in):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        """
        前向传播流程：
        1. 分解为季节性和趋势成分
        2. 只使用STGNN修正趋势成分
        3. 分别预测季节性和修正后的趋势
        4. 合并得到最终预测
        """
        # x: [Batch, Input length, Channel]
        
        # 1. 分解得到季节性和趋势成分
        seasonal_init, trend_init = self.decompsition(x)
        
        # 2. 只修正趋势成分
        refined_trend = self.trend_refiner(trend_init)
        
        # 3. 分别预测季节性和趋势
        # 转置维度: [batch, seq_len, channel] -> [batch, channel, seq_len]
        seasonal_perm = seasonal_init.permute(0, 2, 1)
        trend_perm = refined_trend.permute(0, 2, 1)
        
        if self.individual:
            # 各通道独立预测
            seasonal_output = torch.zeros([seasonal_perm.size(0), seasonal_perm.size(1), self.pred_len], 
                                        dtype=seasonal_perm.dtype).to(seasonal_perm.device)
            trend_output = torch.zeros([trend_perm.size(0), trend_perm.size(1), self.pred_len], 
                                     dtype=trend_perm.dtype).to(trend_perm.device)
            
            for i in range(self.enc_in):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_perm[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_perm[:, i, :])
        else:
            # 共享参数预测
            seasonal_output = self.Linear_Seasonal(seasonal_perm)
            trend_output = self.Linear_Trend(trend_perm)
        
        # 4. 合并预测结果
        x = seasonal_output + trend_output
        
        # 转置回原始维度: [batch, channel, pred_len] -> [batch, pred_len, channel]
        return x.permute(0, 2, 1)


class Model(nn.Module):
    """
    轻量级趋势修正版本
    使用更小的STGNN配置专门处理趋势
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.individual = getattr(configs, 'individual', False)
        
        # 分解核大小
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        
        # 轻量级STGNN趋势修正模块
        self.trend_refiner = SpatialTemporalGNN(
            num_nodes=self.enc_in,
            seq_len=self.seq_len,
            gnn_hidden=getattr(configs, 'gnn_hidden', 256),  # 使用更小的隐藏层                                    #!!!!
            graph_type=getattr(configs, 'graph_type', 'ring')
        )
        
        # 预测网络
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.enc_in):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            
        # 趋势平滑权重（可选）
        self.trend_smoothing = nn.Parameter(torch.tensor(0.5))  # 控制趋势修正强度

    def forward(self, x):
        """
        轻量级版本的前向传播
        """
        # 1. 分解
        seasonal_init, trend_init = self.decompsition(x)
        
        # 2. 轻量级趋势修正
        refined_trend = self.trend_refiner(trend_init)
        
        # 3. 可调节的趋势融合
        # 原始趋势和修正趋势的加权组合
        final_trend = (1 - self.trend_smoothing) * trend_init + self.trend_smoothing * refined_trend
        
        # 4. 预测
        seasonal_perm = seasonal_init.permute(0, 2, 1)
        trend_perm = final_trend.permute(0, 2, 1)
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_perm.size(0), seasonal_perm.size(1), self.pred_len], 
                                        dtype=seasonal_perm.dtype).to(seasonal_perm.device)
            trend_output = torch.zeros([trend_perm.size(0), trend_perm.size(1), self.pred_len], 
                                     dtype=trend_perm.dtype).to(trend_perm.device)
            
            for i in range(self.enc_in):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_perm[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_perm[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_perm)
            trend_output = self.Linear_Trend(trend_perm)
        
        # 5. 合并
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)