from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, SegRNN, CycleNet, \
    LDLinear, SparseTSF, RLinear, RMLP, CycleiTransformer, iTransformer, CrossGNN, FourierGNN, STRLinear, STGNN, AdaptiveCycleNet,CycleRing,PANDA,CMoS,TQNet,SOFTS
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
import seaborn as sns
import numpy as np

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 14,           # 全局默认字体
    'axes.titlesize': 18,      # 子图标题
    'axes.labelsize': 16,      # xlabel/ylabel
    'xtick.labelsize': 14,     # x轴刻度数字
    'ytick.labelsize': 14,     # y轴刻度数字
    'legend.fontsize': 12,     # 图例
    'figure.titlesize': 20     # 总图标题（suptitle）
})

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
            'STRLinear': STRLinear,
            'CrossGNN':CrossGNN,
            'FourierGNN':FourierGNN,
            'STGNN':STGNN,
            'LDLinear': LDLinear,
            'SparseTSF': SparseTSF,
            'RLinear': RLinear,
            'RMLP': RMLP,
            'iTransformer':iTransformer,
            'CycleiTransformer': CycleiTransformer,
            'AdaptiveCycleNet':AdaptiveCycleNet,
            'CycleRing':CycleRing,
            'PANDA':PANDA,
            'CMoS':CMoS,
            'TQNet':TQNet,
            'SOFTS':SOFTS
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
    

    def visualize_strlinear_components(self, setting):
        """
        为 STRLinear 模型生成可视化图表：
        1. 前5个通道的 y_raw, correction, y 预测曲线对比图
        2. 通道间相关性热力图（y_raw vs y）
        """
        if self.args.model != 'STRLinear':
            return

        print("Generating STRLinear component visualizations...")
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()

        # 收集一个 batch 的数据用于可视化
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                if i >= 1:   # 只使用第一个batch
                    break
                batch_x = batch_x.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # 获取模型组件
                y, y_raw, correction = self.model(batch_x, batch_cycle, return_components=True)

                # 转到CPU numpy
                y = y.cpu().numpy()                # (B, pred_len, D)
                y_raw = y_raw.cpu().numpy()
                correction = correction.cpu().numpy()

                # 创建保存目录
                save_dir = f'./strlinear_viz/{setting}/'
                os.makedirs(save_dir, exist_ok=True)

                # 选择第一个样本（batch_idx=0）
                batch_idx = 0
                pred_len = y.shape[1]
                num_nodes = y.shape[2]
                time_axis = np.arange(pred_len)

                # ========== 1. 前5个通道的时间序列对比图 ==========
                num_channels_to_plot = min(5, num_nodes)
                fig, axes = plt.subplots(num_channels_to_plot, 1, figsize=(12, 4*num_channels_to_plot))
                if num_channels_to_plot == 1:
                    axes = [axes]
                for ch in range(num_channels_to_plot):
                    ax = axes[ch]
                    ax.plot(time_axis, y_raw[batch_idx, :, ch], label='y_raw (CycleNet)', color='blue', alpha=0.7)
                    ax.plot(time_axis, y_raw[batch_idx, :, ch] + correction[batch_idx, :, ch], 
                            label='y (final)', color='red', linestyle='--')
                    # 单独绘制修正量（右轴）
                    ax2 = ax.twinx()
                    ax2.bar(time_axis, correction[batch_idx, :, ch], alpha=0.3, color='green', label='y_refined')
                    ax2.set_ylabel('y_refined magnitude', color='green')
                    ax2.tick_params(axis='y', labelcolor='green')
                    # 合并图例
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
                    ax.set_title(f'Channel {ch} - Predictions Comparison')
                    ax.set_xlabel('Time step')
                    ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'channel_curves_batch{batch_idx}.png'), dpi=300)
                plt.close()
                print(f"Saved channel curves to {save_dir}")

                # ========== 2. 通道相关性热力图 ==========
                # 计算 y_raw 和 y 的通道间相关系数矩阵（使用第一个样本）
                # 将 (pred_len, D) 转为 (D, pred_len) 计算通道间相关性
                y_raw_2d = y_raw[batch_idx].T   # (D, pred_len)
                y_2d = y[batch_idx].T           # (D, pred_len)

                corr_y_raw = np.corrcoef(y_raw_2d)
                corr_y = np.corrcoef(y_2d)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                # 热力图 y_raw
                sns.heatmap(corr_y_raw, ax=ax1, cmap='coolwarm', center=0, 
                            square=True, cbar_kws={"shrink": 0.8})
                ax1.set_title(f'Channel Correlation - y_raw (CycleNet)\nSample {batch_idx}')
                ax1.set_xlabel('Channel index')
                ax1.set_ylabel('Channel index')
                # 热力图 y
                sns.heatmap(corr_y, ax=ax2, cmap='coolwarm', center=0, 
                            square=True, cbar_kws={"shrink": 0.8})
                ax2.set_title(f'Channel Correlation - y (After Refiner)\nSample {batch_idx}')
                ax2.set_xlabel('Channel index')
                ax2.set_ylabel('Channel index')

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'channel_correlation_heatmap_batch{batch_idx}.png'), dpi=300)
                plt.close()
                print(f"Saved correlation heatmaps to {save_dir}")

         # ---------- 第二阶段：遍历整个测试集，统计每个通道的平均修正幅度 ----------
        print("Computing channel‑wise average Refinement magnitude on full test set ...")
        total_abs_correction = None          # 累加每个通道的绝对值均值 (shape: num_nodes,)
        total_samples = 0

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # 获取修正量
                _, _, correction = self.model(batch_x, batch_cycle, return_components=True)
                correction = correction.cpu().numpy()   # (B, pred_len, N)

                # 对每个样本，先在时间维上求平均绝对修正，得到 (B, N)，再沿 batch 求和
                sample_mean_abs = np.mean(np.abs(correction), axis=1)  # (B, N)
                batch_sum = np.sum(sample_mean_abs, axis=0)            # (N,)

                if total_abs_correction is None:
                    total_abs_correction = batch_sum
                else:
                    total_abs_correction += batch_sum
                total_samples += correction.shape[0]

        avg_correction_magnitude = total_abs_correction / total_samples  # (N,)

        # ---------- 绘制通道修正幅度矩形热力图（10列 × 17行）----------
        num_nodes = self.args.enc_in
        # 固定行数为 10，自动计算列数（170 → 17）
        n_rows = 10
        n_cols = int(np.ceil(num_nodes / n_rows))

        # 创建全 NaN 矩阵，形状 (n_rows, n_cols)
        heatmap_data = np.full((n_rows, n_cols), np.nan)
        # 按行填充前 num_nodes 个值
        for idx in range(num_nodes):
            row, col = divmod(idx, n_cols)
            heatmap_data[row, col] = avg_correction_magnitude[idx]

        # 生成通道索引标签
        annot_array = np.empty_like(heatmap_data, dtype=object)
        for idx in range(num_nodes):
            row, col = divmod(idx, n_cols)
            annot_array[row, col] = str(idx)
        annot_array[np.isnan(heatmap_data)] = ""

        mask = np.isnan(heatmap_data)

        fig, ax = plt.subplots(figsize=(max(12, n_cols*1.0), max(8, n_rows*0.8)))
        sns.heatmap(
            heatmap_data,
            annot=annot_array if num_nodes <= 200 else False,
            fmt='',
            mask=mask,
            cmap='YlOrRd',
            square=False,          # 此处改为 False，允许非正方形格子
            cbar_kws={'label': 'Mean Abs Refinement', 'shrink': 0.8},
            linewidths=0.5,
            linecolor='grey',
            ax=ax
        )
        ax.set_title(f'Channel‑wise Average Refinement Magnitude (Full Test Set)\n'
                    f'Grid {n_rows}×{n_cols} (Total {num_nodes} channels)')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        heatmap_path = os.path.join(save_dir, 'channel_Refinement_magnitude_10x17.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved 10×17 Refinement magnitude heatmap to {heatmap_path}")       
        
        # ---------- 第三阶段：通道影响力分析（针对 MAR 最大的几个目标通道） ----------
        print("Computing channel influence via gradient attribution ...")
        # 确定要分析的目标通道（例如修正幅度最大的 5 个）
        
        top_targets = np.argsort(avg_correction_magnitude)[::-1][:5]

        influence_matrix, top_sources = self.compute_channel_influence(
            test_loader, target_channels=top_targets.tolist(), top_k_sources=10
        )

        # ----- 可视化 1：影响力矩阵热图（目标通道 vs 所有源通道）-----
        # 如果通道数太多，只显示源通道中影响力较大的子集（比如全部 N 太大时可裁剪）
        num_nodes = self.args.enc_in
        if num_nodes > 50:
            # 选取在任意目标通道中影响力排名前 40 的源通道
            importance_sum = influence_matrix.sum(axis=0)  # (N,)
            top_src_indices = np.argsort(importance_sum)[::-1][:40]
            influence_plot = influence_matrix[:, top_src_indices]
            xticklabels = top_src_indices
        else:
            influence_plot = influence_matrix
            xticklabels = np.arange(num_nodes)

        fig, ax = plt.subplots(figsize=(max(10, influence_plot.shape[1]*0.4),
                                    max(4, influence_plot.shape[0]*0.8)))
        sns.heatmap(influence_plot,
                    xticklabels=xticklabels,
                    yticklabels=top_targets,
                    cmap='YlOrRd',
                    annot=influence_plot.shape[1] <= 30,  # 列数少时显示数值
                    fmt='.4f',
                    linewidths=0.5,
                    cbar_kws={'label': 'Avg Gradient Influence'},
                    ax=ax)
        ax.set_title('Channel Influence Matrix (Target vs. Source)\nTargets: top-5 MAR channels')
        ax.set_xlabel('Source channel index')
        ax.set_ylabel('Target channel index')
        plt.tight_layout()
        influence_path = os.path.join(save_dir, 'channel_influence_matrix.png')
        plt.savefig(influence_path, dpi=300)
        plt.close()
        print(f"Saved channel influence matrix to {influence_path}")

        # ----- 可视化 2：单个目标通道的 top 影响源条形图 -----
        for target_ch in top_targets:
            sources, scores = zip(*top_sources[target_ch])
            # 调整图形尺寸，宽度适中，高度根据 source 数量动态变化
            fig, ax = plt.subplots(figsize=(8, max(4, len(sources) * 0.4)))
            y_pos = np.arange(len(sources))
            ax.barh(y_pos, scores, align='center', color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f'Ch {s}' for s in sources], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Influence Score', fontsize=10)
            ax.set_ylabel('Source Channel', fontsize=10)
            ax.set_title(f'Top influencing sources for Target Channel {target_ch}', fontsize=12)
            # 统一刻度字号
            ax.tick_params(axis='x', labelsize=8)
            # 添加细网格，便于读值
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            bar_path = os.path.join(save_dir, f'influence_target_ch{target_ch}.png')
            plt.savefig(bar_path, dpi=300)
            plt.close()
            print(f"Saved influence bar chart for target Ch{target_ch} to {bar_path}")

        print("STRLinear visualization completed.")
    
    


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
                                 {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF','CrossGNN','FourierGNN','STGNN', 'PANDA','CMoS','TQNet'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Cycle','STRLinear'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF','CrossGNN','FourierGNN','STGNN', 'PANDA','CMoS','TQNet'}):
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

    def compute_channel_influence(self, test_loader, target_channels, top_k_sources=10):
        """
        计算指定目标通道的修正量对输入 y_raw 各通道的梯度归因。
        
        参数:
            test_loader: 测试集 DataLoader
            target_channels: list of int, 要分析的目标通道索引
            top_k_sources: int, 每个目标通道保留影响力最大的源通道数（用于可视化）
        
        返回:
            influence_matrix: ndarray (len(target_channels), num_nodes)
                            每行是一个目标通道受到各源通道的影响力分数
            top_sources: dict, key=目标通道, value=list of (源通道, influence_score)
        """
        self.model.eval()
        num_nodes = self.args.enc_in
        # 初始化累计影响力
        total_influence = np.zeros((len(target_channels), num_nodes))
        total_samples = 0

        with torch.no_grad():   # 注意：梯度计算需要 enable_grad，我们在内部单独处理
            pass

        # 实际上为了计算梯度，需要临时开启梯度
        for i, (batch_x, batch_y, _, _, batch_cycle) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_cycle = batch_cycle.int().to(self.device)
            # 我们不需要 batch_x 的梯度，但需要 y_raw 的梯度，所以直接调用模型获取 y_raw
            # 注意：模型 forward 里 y_raw 是基于 batch_x 计算出来的，因此我们需要让 batch_x 可导
            batch_x.requires_grad = True

            # 前向传播，拿到 y_raw 和 correction（保留计算图）
            y, y_raw, correction = self.model(batch_x, batch_cycle, return_components=True)

            for t_idx, target_ch in enumerate(target_channels):
                # 目标通道修正量在所有时间步上取平均作为标量
                target_corr = correction[:, :, target_ch].mean()
                # 计算 y_raw 的梯度（注意 retain_graph 需要保留整个图以计算多个目标通道）
                grads = torch.autograd.grad(target_corr, y_raw,
                                            retain_graph=True,
                                            create_graph=False)[0]
                # grads shape: (B, pred_len, N)
                # 对时间维度取绝对值并求平均 -> (B, N)
                influence = grads.abs().mean(dim=1).detach().cpu().numpy()  # (B, N)
                total_influence[t_idx] += influence.sum(axis=0)   # 沿 batch 求和

            total_samples += batch_x.size(0)
            # 清除梯度占用，节省显存
            self.model.zero_grad()
            batch_x.requires_grad = False

        # 归一化为平均影响力
        avg_influence = total_influence / total_samples   # (len(target), N)

        # 获取每个目标通道的 top-k 源通道
        top_sources = {}
        for t_idx, target_ch in enumerate(target_channels):
            inf = avg_influence[t_idx]
            sorted_idx = np.argsort(inf)[::-1]
            # 排除目标通道自身（通常自影响最大，可以保留也可以排除，这里保留）
            top_list = [(src, inf[src]) for src in sorted_idx[:top_k_sources]]
            top_sources[target_ch] = top_list

        return avg_influence, top_sources

    
            

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
                                 {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF','CrossGNN','FourierGNN','STGNN', 'PANDA','CMoS','TQNet'}):
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
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF','CrossGNN','FourierGNN','STGNN', 'PANDA','CMoS','TQNet'}):
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
                                 {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF','CrossGNN','FourierGNN','STGNN', 'PANDA','CMoS','TQNet'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Cycle','STRLinear'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF','CrossGNN','FourierGNN','STGNN', 'PANDA','CMoS','TQNet'}):
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

        
        
        if self.args.model == 'STRLinear':
           self.visualize_strlinear_components(setting)
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
                                 {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF','CrossGNN','FourierGNN','STGNN', 'PANDA','CMoS','TQNet'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Cycle','STRLinear'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF','CrossGNN','FourierGNN','STGNN', 'PANDA','CMoS','TQNet'}):
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