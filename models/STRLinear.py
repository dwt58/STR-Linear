import torch
import torch.nn as nn
import torch_geometric.nn as gnn

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

class SpatialTemporalGNN(nn.Module):
    
    def __init__(self, num_nodes, seq_len, gnn_hidden=256, graph_type='ring'):                                                                         
        super(SpatialTemporalGNN, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        
        
        self.spatial_conv = gnn.SAGEConv(seq_len, gnn_hidden)
        
        
        self.channel_adjust = nn.Conv1d(
            in_channels=gnn_hidden,
            out_channels=gnn_hidden,
            kernel_size=1
        )
        
        
        self.dilation_rates = [1, 2, 4, 8]  

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
        
        
        self.adaptive_weights = nn.Parameter(torch.ones(len(self.dilation_rates)))
        
        
        self.feature_recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  
            nn.Conv1d(gnn_hidden, gnn_hidden // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(gnn_hidden // 4, gnn_hidden, 1),
            nn.Sigmoid()
        )
        
        self.output_conv = nn.Conv1d(
            in_channels=gnn_hidden,
            out_channels=seq_len,
            kernel_size=1
        )
        
        
        self.graph_type = graph_type
        if graph_type == 'adaptive':
            self.edge_weights = nn.Parameter(torch.randn(num_nodes, num_nodes))
        else:
            self.register_buffer('edge_index', self.create_graph_structure(num_nodes))
    
    def create_graph_structure(self, num_nodes):
        
        src, dst = [], []
        for i in range(num_nodes):
            neighbors = [(i + j) % num_nodes for j in [-1, 1] if j != 0]
            for n in neighbors:
                src.append(i)
                dst.append(n)
        return torch.tensor([src, dst], dtype=torch.long)
    
    def forward(self, residual_x):
        
        batch_size, seq_len, num_nodes = residual_x.shape
        device = residual_x.device
        
        
        original_residual = residual_x
        
        
        x = residual_x.permute(0, 2, 1)
        
        
        if self.graph_type == 'adaptive':
            adj = torch.softmax(self.edge_weights, dim=-1)
            x = torch.matmul(adj, x)
        else:
            if self.edge_index.device != device:
                self.edge_index = self.edge_index.to(device)
            
            
            x_flat = x.reshape(-1, seq_len)
            
            edge_list = []
            for i in range(batch_size):
                offset = i * num_nodes
                src = self.edge_index[0] + offset
                dst = self.edge_index[1] + offset
                edge_list.append(torch.stack([src, dst]))
            edge_index = torch.cat(edge_list, dim=1).to(device)
            
            
            x_flat = self.spatial_conv(x_flat, edge_index)
            x = x_flat.view(batch_size, num_nodes, -1)  
        
        
        x = x.permute(0, 2, 1)
        
        
        x = self.channel_adjust(x)
        
        
        temporal_features = []
        current_x = x
        
        
        for i, dilated_conv in enumerate(self.dilated_convs):
            conv_output = dilated_conv(current_x)
            
            weighted_output = self.adaptive_weights[i] * conv_output
            temporal_features.append(weighted_output)
            current_x = conv_output  
        
        '''
        if len(temporal_features) > 1:
            
            fused = sum(temporal_features) / len(temporal_features)
        else:
            fused = temporal_features[0]
        '''
        
        if len(temporal_features) > 1:
            
            normalized_weights = torch.softmax(self.adaptive_weights, dim=0)
            
            fused = sum(normalized_weights[i] * temporal_features[i] 
                       for i in range(len(temporal_features)))
        else:
            fused = temporal_features[0]

        attention_weights = self.feature_recalibrate(fused)
        recalibrated = fused * attention_weights
        
        
        x = self.output_conv(recalibrated)
        
        
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
        
        
        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)
        
        self.residual_refiner = SpatialTemporalGNN(
            num_nodes=self.enc_in,
            seq_len=self.seq_len,
            gnn_hidden=256,                                                                                
            graph_type='ring'  
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

    def forward(self, x, cycle_index, return_residuals=False ):
      
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