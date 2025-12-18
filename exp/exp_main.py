from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, SegRNN, CycleNet, \
    LDLinear, SparseTSF, RLinear, RMLP, CycleiTransformer, iTransformer, CrossGNN, FourierGNN, StemGNN, STRLinear,STGNN,TimeMixer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler


import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

# 添加残差可视化函数
def plot_residual_comparison(original_residual, refined_residual, 
                           node_idx=0, batch_idx=0, save_dir='./residual_plots/', 
                           model_id='default', pred_len=96):
    """
    绘制原始残差和修正后残差的时序对比图
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 转换为numpy数组
    if torch.is_tensor(original_residual):
        original_residual = original_residual.detach().cpu().numpy()
    if torch.is_tensor(refined_residual):
        refined_residual = refined_residual.detach().cpu().numpy()
    
    # 提取指定节点和批次的数据
    orig_residual_node = original_residual[batch_idx, :, node_idx]
    refined_residual_node = refined_residual[batch_idx, :, node_idx]
    
    # 创建时间序列
    time_steps = np.arange(len(orig_residual_node))
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制残差对比
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, orig_residual_node, 'b-', linewidth=2, label='R_raw', alpha=0.8)
    plt.plot(time_steps, refined_residual_node, 'r-', linewidth=2, label='R_plus', alpha=0.8)
    plt.xlabel('Timesteps')
    plt.ylabel('Residual value')
    plt.title(f'{model_id} - node {node_idx} - R_raw vs R_plus')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制残差差异
    plt.subplot(2, 1, 2)
    residual_diff = refined_residual_node - orig_residual_node
    plt.plot(time_steps, residual_diff, 'g-', linewidth=2, label='R_refind', alpha=0.8)
    plt.xlabel('Timesteps')
    plt.ylabel('Residual value')
    plt.title(f'{model_id} - node {node_idx} - R_refind')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, f'{model_id}_node_{node_idx}_predlen_{pred_len}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"残差对比图已保存至: {save_path}")
    plt.close()  # 关闭图形，避免内存泄漏
    
    # 打印统计信息
    print(f"\n统计信息 - {model_id} - 节点 {node_idx}:")
    print(f"原始残差 - 均值: {np.mean(orig_residual_node):.4f}, 标准差: {np.std(orig_residual_node):.4f}")
    print(f"修正残差 - 均值: {np.mean(refined_residual_node):.4f}, 标准差: {np.std(refined_residual_node):.4f}")
    print(f"修正量   - 均值: {np.mean(residual_diff):.4f}, 标准差: {np.std(residual_diff):.4f}")

def plot_original_residual_neighbor_comparison(original_residual, center_node=0, batch_idx=0, 
                                            model_id='default', save_dir='./residual_plots/'):
    """
    绘制原始残差的邻居对比图 (R_raw)
    """
    if torch.is_tensor(original_residual):
        original_residual = original_residual.detach().cpu().numpy()
    
    num_nodes = original_residual.shape[2]
    
    # 在环形图中，邻居是左右相邻的节点
    neighbors = [(center_node - 1) % num_nodes, (center_node + 1) % num_nodes]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    time_steps = np.arange(original_residual.shape[1])
    
    # 中心节点：原始残差
    center_r_raw = original_residual[batch_idx, :, center_node]
    axes[0, 0].plot(time_steps, center_r_raw, 
                   'b-', label='Center Channel R_raw', alpha=0.7)
    axes[0, 0].set_title(f'Center Channel vs Neighbors R_raw Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 邻居节点：原始残差
    '''
    for i, neighbor in enumerate(neighbors):
        neighbor_r_raw = original_residual[batch_idx, :, neighbor]
        axes[0, 1].plot(time_steps, neighbor_r_raw, 
                       label=f'Neighbor Channel {neighbor} R_raw', alpha=0.7)
                       '''
    for i, neighbor in enumerate(neighbors):
        neighbor_r_raw = original_residual[batch_idx, :, neighbor]
        
        if i == 0:
            label_name = 'C_{i-1} R_raw'
        else:
            label_name = 'C_{i+1} R_raw'
        
        axes[0, 1].plot(time_steps, neighbor_r_raw, 
                    label=label_name, alpha=0.7)
    axes[0, 1].set_title(f'Neighbor Channel - R_raw')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 中心节点和邻居节点的原始残差对比
    axes[1, 0].plot(time_steps, center_r_raw, 
                   'k-', linewidth=2, label=f'C_i R_raw', alpha=0.8)
    for i, neighbor in enumerate(neighbors):
        neighbor_r_raw = original_residual[batch_idx, :, neighbor]
        if i == 0:
            label_name = 'C_{i-1} R_raw'
        else:
            label_name = 'C_{i+1} R_raw'
        axes[1, 0].plot(time_steps, neighbor_r_raw, 
                        linewidth=1.5, label=label_name, alpha=0.6)
    axes[1, 0].set_title('Center Channel vs Neighbors R_raw Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 计算相关系数并绘制条形图
    neighbor_correlations = []
    for neighbor in neighbors:
        neighbor_r_raw = original_residual[batch_idx, :, neighbor]
        corr = np.corrcoef(center_r_raw, neighbor_r_raw)[0, 1]
        neighbor_correlations.append(corr)
    
    if len(neighbor_correlations) == 2:
        axes[1, 1].bar(['C_{i-1}', 'C_{i+1}'], neighbor_correlations, 
                      color=['skyblue', 'lightcoral'])
        axes[1, 1].set_ylim(-1, 1)
        axes[1, 1].set_title('R_raw Correlation with Neighbors')
        axes[1, 1].set_ylabel('Correlation Coefficient')
        for i, v in enumerate(neighbor_correlations):
            axes[1, 1].text(i, v + 0.05 * np.sign(v), f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'original_neighbor_comparison_node_{center_node}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def analyze_original_residual_correlation(original_residual, batch_idx=0):
    """
    分析原始残差的节点间相关性
    """
    if torch.is_tensor(original_residual):
        original_residual = original_residual.detach().cpu().numpy()
    
    # 计算节点间原始残差的相关性矩阵
    residual_data = original_residual[batch_idx]  # [seq_len, num_nodes]
    correlation_matrix = np.corrcoef(residual_data.T)  # [num_nodes, num_nodes]
    
    return correlation_matrix

def plot_original_residual_spatial_correlation(correlation_matrix, graph_type, model_id, save_dir):
    """
    绘制原始残差的空间相关性热力图
    """
    plt.figure(figsize=(10, 8))
    
    # 使用 origin='lower' 来修正y轴方向
    im = plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, 
                   aspect='auto', origin='lower')
    plt.colorbar(im, label='Correlation Coefficient')
    plt.title(f'R_raw Spatial Correlation\n(Before STGNN Refinement)')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    
    save_path = os.path.join(save_dir, f'{model_id}_original_spatial_correlation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def plot_original_residual_correlation(original_residual, batch_idx=0, 
                                     model_id='default', save_dir='./residual_plots/'):
    """
    Plot spatial correlation heatmap of original residuals (before STGNN refinement)
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy array
    if torch.is_tensor(original_residual):
        original_residual = original_residual.detach().cpu().numpy()
    
    # Extract data for specified batch
    residual_data = original_residual[batch_idx]  # [seq_len, num_nodes]
    
    # Calculate correlation matrix between nodes for original residuals
    correlation_matrix = np.corrcoef(residual_data.T)  # [num_nodes, num_nodes]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Define colormap: red for positive, blue for negative, white near zero
    # Use RdBu_r colormap: Red for positive, Blue for negative
    im = plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, 
                   aspect='auto', origin='lower')
    
    # Add colorbar with clear labels
    cbar = plt.colorbar(im, label='Correlation Coefficient')
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(['-1.0', '-0.5', '0.0', 
                        '0.5', '1.0'])
    
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.title(f'R_raw Spatial Correlation\n(Before STGRR Refinement)')
    
    # Add grid for better readability
    plt.grid(False)
    
    plt.tight_layout()
    
    # Save image
    save_path = os.path.join(save_dir, f'{model_id}_R_raw Spatial Correlation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"R_raw Spatial Correlation heatmap saved: {save_path}")
    plt.close()
    
    # Print correlation statistics
    print(f"R_raw Spatial Correlation statistics:")
    print(f"  - Mean correlation: {np.mean(correlation_matrix):.4f}")
    print(f"  - Std correlation: {np.std(correlation_matrix):.4f}")
    print(f"  - Min correlation: {np.min(correlation_matrix):.4f}")
    print(f"  - Max correlation: {np.max(correlation_matrix):.4f}")
    print(f"  - Positive correlations: {np.sum(correlation_matrix > 0) / correlation_matrix.size:.1%}")
    print(f"  - Negative correlations: {np.sum(correlation_matrix < 0) / correlation_matrix.size:.1%}")
    
    return save_path, correlation_matrix

def analyze_spatial_correlation(original_residual, refined_residual, batch_idx=0):
    """
    分析节点间残差修正的相关性
    """
    if torch.is_tensor(original_residual):
        original_residual = original_residual.detach().cpu().numpy()
    if torch.is_tensor(refined_residual):
        refined_residual = refined_residual.detach().cpu().numpy()
    
    # 计算残差修正量
    residual_diff = refined_residual[batch_idx] - original_residual[batch_idx]  # [seq_len, num_nodes]
    
    # 计算节点间修正量的相关性矩阵
    correlation_matrix = np.corrcoef(residual_diff.T)  # [num_nodes, num_nodes]
    
    return correlation_matrix, residual_diff



def plot_spatial_correlation(correlation_matrix, graph_type, model_id, save_dir):
    """
    Plot heatmap of spatial correlation between nodes
    """
    plt.figure(figsize=(10, 8))
    
    # 使用 origin='lower' 来修正y轴方向
    im = plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, 
                   aspect='auto', origin='lower')
    plt.colorbar(im, label='Correlation Coefficient')
    plt.title(f'R_plus Spatial Correlation\n(After STGRR Refinement)')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    
    # 标记邻居关系（修正y轴方向后）
    '''
    if graph_type == 'ring':
        num_nodes = correlation_matrix.shape[0]
        for i in range(num_nodes):
            neighbors = [(i-1) % num_nodes, (i+1) % num_nodes]
            for n in neighbors:
                plt.plot(i, n, 'ro', markersize=2, alpha=0.6)  # 注意：现在是 (x, y) = (i, n)
    '''
    save_path = os.path.join(save_dir, f'{model_id}_spatial_correlation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def plot_neighbor_comparison(original_residual, refined_residual, center_node=0, batch_idx=0, 
                           model_id='default', save_dir='./residual_plots/'):
    """
    Plot comparison between center node and its neighbors
    """
    if torch.is_tensor(original_residual):
        original_residual = original_residual.detach().cpu().numpy()
    if torch.is_tensor(refined_residual):
        refined_residual = refined_residual.detach().cpu().numpy()
    
    num_nodes = original_residual.shape[2]
    
    # In ring graph, neighbors are left and right adjacent nodes
    neighbors = [(center_node - 1) % num_nodes, (center_node + 1) % num_nodes]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    time_steps = np.arange(original_residual.shape[1])
    
    # Center node: original vs refined residual
    axes[0, 0].plot(time_steps, original_residual[batch_idx, :, center_node], 
                   'b-', label='R_raw', alpha=0.7)
    axes[0, 0].plot(time_steps, refined_residual[batch_idx, :, center_node], 
                   'r-', label='R_plus', alpha=0.7)
    axes[0, 0].set_title(f'C_i Residual Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Center node: residual correction
    center_diff = refined_residual[batch_idx, :, center_node] - original_residual[batch_idx, :, center_node]
    axes[0, 1].plot(time_steps, center_diff, 'g-', label='R_refined')
    axes[0, 1].set_title(f'C_i Residual Refinement')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Neighbor nodes: correction comparison
    for i, neighbor in enumerate(neighbors):
        neighbor_diff = refined_residual[batch_idx, :, neighbor] - original_residual[batch_idx, :, neighbor]
        if i == 0:
            label_name = 'C_{i-1} R_plus'
        else:
            label_name = 'C_{i+1} R_plus'
        axes[1, 0].plot(time_steps, neighbor_diff, label=label_name, alpha=0.7)
    axes[1, 0].plot(time_steps, center_diff, 'k-', label='C_i R_plus', linewidth=2)
    axes[1, 0].set_title('Center Channel vs Neighbors R_plus Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Calculate correlations
    neighbor_diffs = []
    for neighbor in neighbors:
        neighbor_diff = refined_residual[batch_idx, :, neighbor] - original_residual[batch_idx, :, neighbor]
        neighbor_diffs.append(neighbor_diff)
    
    if len(neighbor_diffs) == 2:
        corr1 = np.corrcoef(center_diff, neighbor_diffs[0])[0, 1]
        corr2 = np.corrcoef(center_diff, neighbor_diffs[1])[0, 1]
        axes[1, 1].bar(['C_{i-1}', 'C_{i+1}'], [corr1, corr2], color=['skyblue', 'lightcoral'])
        axes[1, 1].set_ylim(-1, 1)
        axes[1, 1].set_title('R_plus Correlation with Neighbors')
        axes[1, 1].set_ylabel('Correlation Coefficient')
        for i, v in enumerate([corr1, corr2]):
            axes[1, 1].text(i, v + 0.05 * np.sign(v), f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'neighbor_comparison_node_{center_node}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def analyze_spatial_propagation(original_residual, refined_residual, batch_idx=0, 
                              model_id='default', save_dir='./residual_plots/'):
    """
    Analyze spatial propagation patterns of residual corrections
    """
    if torch.is_tensor(original_residual):
        original_residual = original_residual.detach().cpu().numpy()
    if torch.is_tensor(refined_residual):
        refined_residual = refined_residual.detach().cpu().numpy()
    
    residual_diff = refined_residual[batch_idx] - original_residual[batch_idx]  # [seq_len, num_nodes]
    
    # 计算每个节点的平均修正强度
    node_correction_strength = np.mean(np.abs(residual_diff), axis=0)
    
    # 计算空间自相关性（Moran's I）
    num_nodes = residual_diff.shape[1]
    W = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        W[i, (i-1) % num_nodes] = 1  # left neighbor
        W[i, (i+1) % num_nodes] = 1  # right neighbor
    
    # Moran's I calculation
    z = node_correction_strength - np.mean(node_correction_strength)
    morans_i = (z @ W @ z) / (z @ z) / np.sum(W)
    
    # 绘制空间模式
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(node_correction_strength, 'o-', linewidth=2, markersize=4)
    plt.xlabel('Node Index')
    plt.ylabel('Average Correction Strength')
    plt.title('Residual Correction Strength by Node')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # 绘制修正量的空间-时间热力图，使用 origin='lower'
    plt.imshow(residual_diff.T, aspect='auto', cmap='RdBu_r', 
               interpolation='nearest', origin='lower')
    plt.colorbar(label='Residual Correction Amount')
    plt.xlabel('Time Steps')
    plt.ylabel('Node Index')
    plt.title(f'Spatio-temporal Distribution of Corrections\nMoran\'s I = {morans_i:.4f}')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_id}_spatial_propagation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path, morans_i

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'SegRNN': SegRNN,
            'CycleNet': CycleNet,
            'STRLinear':STRLinear,
            'CrossGNN':CrossGNN,
            'TimeMixer':TimeMixer,
            'FourierGNN':FourierGNN,
            'StemGNN':StemGNN,
            'STGNN':STGNN,
            'LDLinear': LDLinear,
            'SparseTSF': SparseTSF,
            'RLinear': RLinear,
            'RMLP': RMLP,
            'iTransformer':iTransformer,
            'CycleiTransformer': CycleiTransformer
        }
        # 确保模型名称有效
        if self.args.model not in model_dict:
            raise ValueError(f"Unsupported model: {self.args.model}")
        model = model_dict[self.args.model].Model(self.args).float()  
       
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Cycle','STRLinear'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF','CrossGNN','FourierGNN', 'StemGNN','STGNN'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Cycle','STRLinear'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF','CrossGNN','FourierGNN', 'StemGNN','STGNN'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def _visualize_residuals(self, setting):
        """
        简化的残差可视化方法 - 只保留核心可视化功能
        """
        # 检查是否启用残差可视化
        if not getattr(self.args, 'residual_visualization', 1):
            print("Residual visualization is disabled")
            return
            
        if self.args.model != 'STRLinear':
            return
                
        print("Starting residual visualization...")
        
        try:
            test_data, test_loader = self._get_data(flag='test')
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                if i >= 1:
                    break
                    
                batch_x = batch_x.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    predictions, original_residual, refined_residual = self.model(
                        batch_x, batch_cycle, return_residuals=True
                    )
                
                save_dir = f'./residual_plots/{setting}/'
                os.makedirs(save_dir, exist_ok=True)
                # 0. Original residual heatmap (新增)
                original_corr_path, original_corr_matrix = plot_original_residual_correlation(
                original_residual, batch_idx=0, model_id=setting, save_dir=save_dir
            )
                print(f"Original residual spatial correlation heatmap saved: {original_corr_path}")
                
                # 1. Original time series comparison plots
                num_nodes_to_visualize = min(3, self.args.enc_in)
                for node_idx in range(num_nodes_to_visualize):
                    plot_residual_comparison(
                        original_residual, refined_residual,
                        node_idx=node_idx, batch_idx=0,
                        model_id=setting, pred_len=self.args.pred_len,
                        save_dir=save_dir
                    )
                
                # 2. Spatial correlation analysis for original residual (新增)
                original_correlation_matrix = analyze_original_residual_correlation(
                    original_residual
                )
                original_corr_path = plot_original_residual_spatial_correlation(
                    original_correlation_matrix, self.args.graph_type, setting, save_dir
                )
                print(f"Original R_raw spatial correlation plot saved: {original_corr_path}")
                
                # 3. Neighbor comparison analysis for original residual - 输出5个节点的对比图 (新增)
                num_center_nodes = min(5, self.args.enc_in)
                for center_node in range(num_center_nodes):
                    original_neighbor_path = plot_original_residual_neighbor_comparison(
                        original_residual, center_node,
                        batch_idx=0, model_id=setting, save_dir=save_dir
                    )
                    print(f"Original R_raw neighbor comparison plot saved: {original_neighbor_path}")

                # 2. Spatial correlation analysis
                correlation_matrix, residual_diff = analyze_spatial_correlation(
                    original_residual, refined_residual
                )
                corr_path = plot_spatial_correlation(
                    correlation_matrix, self.args.graph_type, setting, save_dir
                )
                print(f"Spatial correlation plot saved: {corr_path}")
                
                # 3. Neighbor comparison analysis - 输出5个节点的对比图
                num_center_nodes = min(5, self.args.enc_in)
                for center_node in range(num_center_nodes):
                    neighbor_path = plot_neighbor_comparison(
                        original_residual, refined_residual, center_node,
                        batch_idx=0, model_id=setting, save_dir=save_dir
                    )
                    print(f"Neighbor comparison plot saved: {neighbor_path}")
                
                print("Residual visualization completed!")
                break
                    
        except Exception as e:
            print(f"Residual visualization failed: {e}")

    
            

    def train(self, setting):
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Cycle','STRLinear'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF','CrossGNN','FourierGNN', 'StemGNN','STGNN'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {'Cycle','STRLinear'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF','CrossGNN','FourierGNN', 'StemGNN','STGNN'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                # max_memory = max(max_memory, current_memory)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # 训练完成后进行残差可视化
        self._visualize_residuals(setting)

        # print(f"Max Memory (MB): {max_memory}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Cycle','STRLinear'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF','CrossGNN','FourierGNN', 'StemGNN','STGNN'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Cycle','STRLinear'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF','CrossGNN','FourierGNN', 'StemGNN','STGNN'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                # inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    np.savetxt(os.path.join(folder_path, str(i) + '.txt'), pd)
                    np.savetxt(os.path.join(folder_path, str(i) + 'true.txt'), gt)

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # 测试完成后进行残差可视化
        self._visualize_residuals(setting)

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Cycle','STRLinear'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF','CrossGNN','FourierGNN', 'StemGNN','STGNN'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Cycle','STRLinear'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF','CrossGNN','FourierGNN', 'StemGNN','STGNN'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return