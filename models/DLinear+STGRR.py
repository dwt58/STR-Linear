import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric.nn as gnn

class moving_avg(nn.Module):
  
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
   
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


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
    
    def forward(self, trend_component):

        batch_size, seq_len, num_nodes = trend_component.shape
        device = trend_component.device
        

        original_trend = trend_component
        
 
        x = trend_component.permute(0, 2, 1)
        
 
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
            x = x_flat.view(batch_size, num_nodes, -1)  # [batch_size, num_nodes, gnn_hidden]
        
 
        x = x.permute(0, 2, 1)
        

        x = self.channel_adjust(x)
        
   
        temporal_features = []
        current_x = x
        
  
        for i, dilated_conv in enumerate(self.dilated_convs):
            conv_output = dilated_conv(current_x)
 
            weighted_output = self.adaptive_weights[i] * conv_output
            temporal_features.append(weighted_output)
            current_x = conv_output  
        
 
        if len(temporal_features) > 1:

            fused = sum(temporal_features) / len(temporal_features)
        else:
            fused = temporal_features[0]
        

        attention_weights = self.feature_recalibrate(fused)
        recalibrated = fused * attention_weights
        

        x = self.output_conv(recalibrated)
        

        return original_trend + x


class TrendRefinedDLinear(nn.Module):

    def __init__(self, configs):
        super(TrendRefinedDLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.individual = getattr(configs, 'individual', False)
        
      
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        

        self.trend_refiner = SpatialTemporalGNN(
            num_nodes=self.enc_in,
            seq_len=self.seq_len,
            gnn_hidden=getattr(configs, 'gnn_hidden', 256),
            graph_type=getattr(configs, 'graph_type', 'ring')
        )
 
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
 
        seasonal_init, trend_init = self.decompsition(x)
        

        refined_trend = self.trend_refiner(trend_init)
        
 
        seasonal_perm = seasonal_init.permute(0, 2, 1)
        trend_perm = refined_trend.permute(0, 2, 1)
        
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
  
        x = seasonal_output + trend_output
        
 
        return x.permute(0, 2, 1)


class Model(nn.Module):
 
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.individual = getattr(configs, 'individual', False)
        
       
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        
   
        self.trend_refiner = SpatialTemporalGNN(
            num_nodes=self.enc_in,
            seq_len=self.seq_len,
            gnn_hidden=getattr(configs, 'gnn_hidden', 256), 
            graph_type=getattr(configs, 'graph_type', 'ring')
        )
        
       
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.enc_in):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            
   
        self.trend_smoothing = nn.Parameter(torch.tensor(0.5))  

    def forward(self, x):
      
        seasonal_init, trend_init = self.decompsition(x)
        
      
        refined_trend = self.trend_refiner(trend_init)
        
        
        final_trend = (1 - self.trend_smoothing) * trend_init + self.trend_smoothing * refined_trend
        
      
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