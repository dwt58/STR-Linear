import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class SpatialTemporalGNN(nn.Module):
    """独立的空间-时序GNN预测模型（用于消融实验）"""
    def __init__(self, configs):
        super(SpatialTemporalGNN, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.use_revin = getattr(configs, 'use_revin', False)
        
        # 模型参数（可根据需要调整）
        gnn_hidden = getattr(configs, 'gnn_hidden', 256)
        graph_type = getattr(configs, 'graph_type', 'ring')
        
        # 1. 空间建模层
        self.spatial_conv = gnn.SAGEConv(self.seq_len, gnn_hidden)
        
        # 2. 时序建模层
        # 首先使用1x1卷积调整通道数
        self.channel_adjust = nn.Conv1d(
            in_channels=gnn_hidden,
            out_channels=gnn_hidden,
            kernel_size=1
        )
        
        # 空洞卷积序列
        self.dilation_rates = [1, 2, 4, 8]
        
        # 空洞卷积块
        self.dilated_convs = nn.ModuleList()
        for i, dilation in enumerate(self.dilation_rates):
            conv_block = nn.Sequential(
                nn.Conv1d(
                    in_channels=gnn_hidden,
                    out_channels=gnn_hidden,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    padding_mode='replicate',
                    groups=min(gnn_hidden, 8)
                ),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm1d(gnn_hidden),
                nn.Dropout(0.03 + 0.02 * i)
            )
            self.dilated_convs.append(conv_block)
        
        # 自适应权重机制
        self.adaptive_weights = nn.Parameter(torch.ones(len(self.dilation_rates)))
        
        # 特征重校准
        self.feature_recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(gnn_hidden, gnn_hidden // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(gnn_hidden // 4, gnn_hidden, 1),
            nn.Sigmoid()
        )
        
        # 3. 输出层 - 直接预测未来序列
        self.output_conv = nn.Conv1d(
            in_channels=gnn_hidden,
            out_channels=self.pred_len,  # 直接输出预测长度
            kernel_size=1
        )
        
        # 图结构
        self.graph_type = graph_type
        if graph_type == 'adaptive':
            self.edge_weights = nn.Parameter(torch.randn(self.enc_in, self.enc_in))
        else:
            self.register_buffer('edge_index', self.create_graph_structure(self.enc_in))
    
    def create_graph_structure(self, num_nodes):
        """创建环形图结构"""
        src, dst = [], []
        for i in range(num_nodes):
            neighbors = [(i + j) % num_nodes for j in [-1, 1] if j != 0]
            for n in neighbors:
                src.append(i)
                dst.append(n)
        return torch.tensor([src, dst], dtype=torch.long)
    
    def forward(self, x):
        """
        输入: [batch_size, seq_len, num_nodes] - 历史序列
        输出: [batch_size, pred_len, num_nodes] - 预测序列
        """
        batch_size, seq_len, num_nodes = x.shape
        device = x.device
        
        # 实例归一化（可选）
        if self.use_revin:
            # 计算每个序列的均值和方差
            seq_mean = torch.mean(x, dim=1, keepdim=True)  # [batch_size, 1, num_nodes]
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5  # [batch_size, 1, num_nodes]
            x_normalized = (x - seq_mean) / torch.sqrt(seq_var)
        else:
            x_normalized = x
        
        # 转换维度 [batch_size, num_nodes, seq_len]
        x_processed = x_normalized.permute(0, 2, 1)
        
        # 空间建模
        if self.graph_type == 'adaptive':
            adj = torch.softmax(self.edge_weights, dim=-1)
            x_processed = torch.matmul(adj, x_processed)
        else:
            if self.edge_index.device != device:
                self.edge_index = self.edge_index.to(device)
            
            # 重塑为图卷积需要的格式 [batch_size * num_nodes, seq_len]
            x_flat = x_processed.reshape(-1, seq_len)
            
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
            x_processed = x_flat.view(batch_size, num_nodes, -1)  # [batch_size, num_nodes, gnn_hidden]
        
        # 时序建模
        # 调整维度: [batch_size, num_nodes, gnn_hidden] -> [batch_size, gnn_hidden, num_nodes]
        x_processed = x_processed.permute(0, 2, 1)
        
        # 调整通道数
        x_processed = self.channel_adjust(x_processed)
        
        # 应用空洞卷积块
        temporal_features = []
        current_x = x_processed
        
        for i, dilated_conv in enumerate(self.dilated_convs):
            conv_output = dilated_conv(current_x)
            weighted_output = self.adaptive_weights[i] * conv_output
            temporal_features.append(weighted_output)
            current_x = conv_output
        
        # 多尺度特征融合
        if len(temporal_features) > 1:
            fused = sum(temporal_features) / len(temporal_features)
        else:
            fused = temporal_features[0]
        
        # 特征重校准
        attention_weights = self.feature_recalibrate(fused)
        recalibrated = fused * attention_weights
        
        # 最终输出层 - 直接预测未来序列
        output = self.output_conv(recalibrated)  # [batch_size, pred_len, num_nodes]
        
        # 调整维度: [batch_size, pred_len, num_nodes] -> [batch_size, pred_len, num_nodes]
        # 注意：这里不需要permute，因为output_conv的输出已经是正确的形状
        
        # 反归一化（可选）
        if self.use_revin:
            # 确保维度匹配：output是 [batch_size, pred_len, num_nodes]
            # seq_mean和seq_var是 [batch_size, 1, num_nodes]，可以广播
            output = output * torch.sqrt(seq_var) + seq_mean
        
        return output

# 为了兼容实验框架，创建Model包装类
class Model(nn.Module):
    """独立STGNN预测模型（用于消融实验）"""
    def __init__(self, configs):
        super(Model, self).__init__()
        self.stgnn = SpatialTemporalGNN(configs)
    
    def forward(self, x, cycle_index=None):
        """
        兼容原框架的前向传播
        输入: 
            x: [batch_size, seq_len, num_nodes] - 历史序列
            cycle_index: 为了兼容性保留，但不会使用
        输出: [batch_size, pred_len, num_nodes] - 预测序列
        """
        return self.stgnn(x)