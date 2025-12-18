import numpy as np
import warnings


def RSE(pred, true):
    """计算相对平方误差"""
    numerator = np.sqrt(np.sum((true - pred) ** 2))
    denominator = np.sqrt(np.sum((true - true.mean()) ** 2))
    return numerator / denominator


def CORR(pred, true):
    """计算相关系数 - 使用更内存高效的方法"""
    # 展平时间步和特征维度，只保留样本维度
    pred_flat = pred.reshape(pred.shape[0], -1)
    true_flat = true.reshape(true.shape[0], -1)
    
    # 计算相关系数
    corr_coeffs = []
    for i in range(pred_flat.shape[0]):
        if i % 100 == 0:  # 每100个样本打印进度
            print(f"Calculating CORR for sample {i}/{pred_flat.shape[0]}")
        
        pred_sample = pred_flat[i]
        true_sample = true_flat[i]
        
        # 计算单个样本的相关系数
        if np.std(pred_sample) > 1e-12 and np.std(true_sample) > 1e-12:
            corr = np.corrcoef(pred_sample, true_sample)[0, 1]
            corr_coeffs.append(corr)
        else:
            corr_coeffs.append(0.0)
    
    return 0.01 * np.mean(corr_coeffs)


def MAE(pred, true):
    """计算平均绝对误差 - 内存高效版本"""
    total_mae = 0.0
    count = 0
    
    # 分批计算
    batch_size = 100  # 每次处理100个样本
    for i in range(0, pred.shape[0], batch_size):
        end_idx = min(i + batch_size, pred.shape[0])
        pred_batch = pred[i:end_idx]
        true_batch = true[i:end_idx]
        
        batch_mae = np.mean(np.abs(pred_batch - true_batch))
        total_mae += batch_mae * (end_idx - i)
        count += (end_idx - i)
    
    return total_mae / count


def MSE(pred, true):
    """计算均方误差 - 内存高效版本"""
    total_mse = 0.0
    count = 0
    
    # 分批计算
    batch_size = 100
    for i in range(0, pred.shape[0], batch_size):
        end_idx = min(i + batch_size, pred.shape[0])
        pred_batch = pred[i:end_idx]
        true_batch = true[i:end_idx]
        
        batch_mse = np.mean((pred_batch - true_batch) ** 2)
        total_mse += batch_mse * (end_idx - i)
        count += (end_idx - i)
    
    return total_mse / count


def RMSE(pred, true):
    """计算均方根误差"""
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """计算平均绝对百分比误差 - 完全重写以避免内存问题"""
    total_mape = 0.0
    count = 0
    
    # 逐样本计算，避免创建巨大掩码数组
    for i in range(pred.shape[0]):
        if i % 100 == 0:  # 每100个样本打印进度
            print(f"Calculating MAPE for sample {i}/{pred.shape[0]}")
        
        pred_sample = pred[i]
        true_sample = true[i]
        
        # 只为当前样本创建掩码
        mask = true_sample != 0
        if np.any(mask):
            # 只处理非零值
            pred_nonzero = pred_sample[mask]
            true_nonzero = true_sample[mask]
            
            sample_mape = np.mean(np.abs((pred_nonzero - true_nonzero) / true_nonzero))
            total_mape += sample_mape
            count += 1
        else:
            # 如果所有值都是零，跳过这个样本
            continue
    
    return total_mape / count if count > 0 else 0.0


def MSPE(pred, true):
    """计算均方百分比误差 - 内存高效版本"""
    total_mspe = 0.0
    count = 0
    
    # 逐样本计算
    for i in range(pred.shape[0]):
        if i % 100 == 0:  # 每100个样本打印进度
            print(f"Calculating MSPE for sample {i}/{pred.shape[0]}")
        
        pred_sample = pred[i]
        true_sample = true[i]
        
        # 只为当前样本创建掩码
        mask = true_sample != 0
        if np.any(mask):
            # 只处理非零值
            pred_nonzero = pred_sample[mask]
            true_nonzero = true_sample[mask]
            
            sample_mspe = np.mean(np.square((pred_nonzero - true_nonzero) / true_nonzero))
            total_mspe += sample_mspe
            count += 1
        else:
            # 如果所有值都是零，跳过这个样本
            continue
    
    return total_mspe / count if count > 0 else 0.0


def metric(pred, true):
    """
    计算所有评估指标 - 完全重写为内存友好版本
    """
    print("Starting metric calculation...")
    
    # 计算相对简单的指标
    print("Calculating MAE...")
    mae = MAE(pred, true)
    
    print("Calculating MSE...")
    mse = MSE(pred, true)
    
    print("Calculating RMSE...")
    rmse = RMSE(pred, true)
    
    print("Calculating RSE...")
    rse = RSE(pred, true)
    
    # 计算需要更多内存的指标
    print("Calculating MAPE...")
    mape = MAPE(pred, true)
    
    print("Calculating MSPE...")
    mspe = MSPE(pred, true)
    
    print("Calculating CORR...")
    corr = CORR(pred, true)
    
    print("Metric calculation completed!")
    
    return mae, mse, rmse, mape, mspe, rse, corr


# 添加一个专门处理超大数据的评估函数
def metric_large_data(pred, true, sample_ratio=0.1):
    """
    针对超大数据的评估函数，通过采样减少内存使用
    """
    num_samples = pred.shape[0]
    sample_size = max(1, int(num_samples * sample_ratio))
    
    print(f"Sampling {sample_size} out of {num_samples} samples for metric calculation")
    
    # 随机选择部分样本
    indices = np.random.choice(num_samples, sample_size, replace=False)
    pred_sample = pred[indices]
    true_sample = true[indices]
    
    # 计算指标
    return metric(pred_sample, true_sample)