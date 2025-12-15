# bnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import math
import random


class BayesianLinear(nn.Module):
    """改进的贝叶斯线性层"""

    def __init__(self, in_features, out_features, weight_prior_std=None, bias_prior_std=0.1, layer_depth=0):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.layer_depth = layer_depth

        # 自适应先验标准差（基于He初始化）
        if weight_prior_std is None:
            self.prior_weight_std = math.sqrt(2.0 / in_features) * 0.05  # 减小先验标准差
        else:
            self.prior_weight_std = weight_prior_std

        self.prior_bias_std = bias_prior_std

        # 权重参数：均值和对数标准差
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * math.sqrt(2.0 / in_features))

        # 深度相关的初始不确定性 - 减小初始不确定性
        base_logvar = -4 - layer_depth * 0.2
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.05 + base_logvar)

        # 偏置参数：均值和对数标准差
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.01)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.05 + base_logvar)

        # 先验分布参数
        self.prior_weight_mu = 0
        self.prior_bias_mu = 0

    def forward(self, x):
        """前向传播，使用重参数化技巧"""
        # 权重采样
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight_eps = torch.randn_like(weight_std)
        weight = self.weight_mu + weight_eps * weight_std

        # 偏置采样
        bias_std = torch.exp(0.5 * self.bias_logvar)
        bias_eps = torch.randn_like(bias_std)
        bias = self.bias_mu + bias_eps * bias_std

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """计算KL散度"""
        # 权重的KL散度
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu - self.prior_weight_mu) ** 2 / self.prior_weight_std ** 2 +
            weight_var / self.prior_weight_std ** 2 -
            1 - self.weight_logvar + 2 * math.log(self.prior_weight_std)
        )

        # 偏置的KL散度
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu - self.prior_bias_mu) ** 2 / self.prior_bias_std ** 2 +
            bias_var / self.prior_bias_std ** 2 -
            1 - self.bias_logvar + 2 * math.log(self.prior_bias_std)
        )

        return weight_kl + bias_kl


class BayesianConv1d(nn.Module):
    """改进的贝叶斯1D卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 weight_prior_std=None, bias_prior_std=0.1, layer_depth=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.layer_depth = layer_depth

        # 自适应先验
        if weight_prior_std is None:
            fan_in = in_channels * kernel_size
            self.prior_weight_std = math.sqrt(2.0 / fan_in) * 0.05  # 减小先验标准差
        else:
            self.prior_weight_std = weight_prior_std

        self.prior_bias_std = bias_prior_std

        # 权重参数
        fan_in = in_channels * kernel_size
        self.weight_mu = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size) * math.sqrt(2.0 / fan_in))

        base_logvar = -4 - layer_depth * 0.2  # 减小初始不确定性
        self.weight_logvar = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size) * 0.05 + base_logvar)

        # 偏置参数
        self.bias_mu = nn.Parameter(torch.randn(out_channels) * 0.01)
        self.bias_logvar = nn.Parameter(torch.randn(out_channels) * 0.05 + base_logvar)

        # 先验分布参数
        self.prior_weight_mu = 0
        self.prior_bias_mu = 0

    def forward(self, x):
        """前向传播"""
        # 权重采样
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight_eps = torch.randn_like(weight_std)
        weight = self.weight_mu + weight_eps * weight_std

        # 偏置采样
        bias_std = torch.exp(0.5 * self.bias_logvar)
        bias_eps = torch.randn_like(bias_std)
        bias = self.bias_mu + bias_eps * bias_std

        return F.conv1d(x, weight, bias, self.stride, self.padding)

    def kl_divergence(self):
        """计算KL散度"""
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu - self.prior_weight_mu) ** 2 / self.prior_weight_std ** 2 +
            weight_var / self.prior_weight_std ** 2 -
            1 - self.weight_logvar + 2 * math.log(self.prior_weight_std)
        )

        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu - self.prior_bias_mu) ** 2 / self.prior_bias_std ** 2 +
            bias_var / self.prior_bias_std ** 2 -
            1 - self.bias_logvar + 2 * math.log(self.prior_bias_std)
        )

        return weight_kl + bias_kl


class TemperatureScaling(nn.Module):
    """温度校准模块"""

    def __init__(self, num_params=5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(num_params))

    def forward(self, mean, var):
        calibrated_var = var * self.temperature.unsqueeze(0) ** 2
        return mean, calibrated_var


class BayesianNeuralNetwork(nn.Module):
    """改进的贝叶斯神经网络"""

    def __init__(self,
                 in_channel: int = 1,
                 out_channel: int = 5,
                 spectrum_size: int = 3516,
                 hidden_dims: list = [1024, 512, 256, 128]):
        super().__init__()

        self.out_channel = out_channel
        self.spectrum_size = spectrum_size

        # 多尺度卷积特征提取 - 捕获不同尺度的光谱特征
        self.multi_scale_conv = nn.ModuleList([
            # 小尺度特征 - 捕获精细光谱线
            nn.Sequential(
                BayesianConv1d(in_channel, 32, kernel_size=3, stride=1, padding=1, layer_depth=0),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ),
            # 中尺度特征 - 捕获中等宽度特征
            nn.Sequential(
                BayesianConv1d(in_channel, 32, kernel_size=9, stride=1, padding=4, layer_depth=0),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ),
            # 大尺度特征 - 捕获宽谱特征
            nn.Sequential(
                BayesianConv1d(in_channel, 32, kernel_size=27, stride=1, padding=13, layer_depth=0),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
        ])

        # 主卷积路径 - 更深的网络
        self.conv1 = BayesianConv1d(96, 128, kernel_size=16, stride=4, padding=8, layer_depth=1)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = BayesianConv1d(128, 256, kernel_size=16, stride=4, padding=8, layer_depth=2)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = BayesianConv1d(256, 512, kernel_size=8, stride=2, padding=4, layer_depth=3)
        self.bn3 = nn.BatchNorm1d(512)

        self.conv4 = BayesianConv1d(512, 1024, kernel_size=4, stride=2, padding=2, layer_depth=4)
        self.bn4 = nn.BatchNorm1d(1024)

        # 注意力机制 - 让模型关注重要的光谱区域
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(1024, 1024 // 16),
            nn.ReLU(),
            nn.Linear(1024 // 16, 1024),
            nn.Sigmoid()
        )

        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 计算特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, in_channel, spectrum_size)
            feature_dim = self._get_feature_dim(dummy)

        # 改进的全连接层 - 使用残差连接
        self.bayesian_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        prev_dim = feature_dim

        for i, hidden_dim in enumerate(hidden_dims):
            self.bayesian_layers.append(BayesianLinear(prev_dim, hidden_dim, layer_depth=i + 5))
            # 添加跳跃连接（当维度匹配时）
            if prev_dim == hidden_dim:
                self.skip_connections.append(nn.Identity())
            else:
                self.skip_connections.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # 输出层
        self.mu_layer = BayesianLinear(prev_dim, out_channel, layer_depth=len(hidden_dims) + 5)

        # 参数特定的噪声方差（基于实验调优）
        self.param_log_noise_var = nn.Parameter(torch.tensor([-3.5, -2.5, -3.5, -5.0, -2.5]))

        # 温度校准
        self.temperature_scaling = TemperatureScaling(out_channel)

        # 改进的dropout
        self.dropout = nn.Dropout(0.1)  # 进一步降低dropout率

    def _get_feature_dim(self, x):
        # 多尺度特征提取
        multi_features = []
        for conv_branch in self.multi_scale_conv:
            branch_out = conv_branch(x)
            multi_features.append(branch_out)

        x = torch.cat(multi_features, dim=1)  # 拼接多尺度特征

        # 主卷积路径
        x = F.relu(self.bn1(F.conv1d(x, torch.zeros(128, 96, 16), stride=4, padding=8)))
        x = F.relu(self.bn2(F.conv1d(x, torch.zeros(256, 128, 16), stride=4, padding=8)))
        x = F.relu(self.bn3(F.conv1d(x, torch.zeros(512, 256, 8), stride=2, padding=4)))
        x = F.relu(self.bn4(F.conv1d(x, torch.zeros(1024, 512, 4), stride=2, padding=2)))

        x = self.global_pool(x)
        return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        """前向传播"""
        batch_size = x.size(0)

        # 多尺度特征提取
        multi_features = []
        for conv_branch in self.multi_scale_conv:
            branch_out = conv_branch(x)
            multi_features.append(branch_out)

        x = torch.cat(multi_features, dim=1)  # [B, 96, L/2]

        # 主卷积路径
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = F.relu(self.bn4(self.conv4(x)))

        # 注意力机制
        attention_weights = self.attention(x)
        x = x * attention_weights.unsqueeze(-1)
        x = self.dropout(x)

        # 全局池化
        x = self.global_pool(x)
        x = x.view(batch_size, -1)

        # 改进的全连接层（带残差连接）
        for i, (layer, skip) in enumerate(zip(self.bayesian_layers, self.skip_connections)):
            residual = skip(x)
            x = F.relu(layer(x))
            if x.shape == residual.shape:  # 只有当形状匹配时才添加残差连接
                x = x + residual
            x = self.dropout(x)

        # 输出层
        mu = self.mu_layer(x)

        # 参数特定的噪声方差
        noise_var = torch.exp(self.param_log_noise_var).expand(batch_size, -1)

        return mu, noise_var

    def kl_divergence(self):
        """计算总的KL散度"""
        kl_sum = 0

        # 多尺度卷积层的KL散度
        for conv_branch in self.multi_scale_conv:
            for layer in conv_branch:
                if isinstance(layer, BayesianConv1d):
                    kl_sum += layer.kl_divergence()

        # 主卷积层的KL散度
        kl_sum += self.conv1.kl_divergence()
        kl_sum += self.conv2.kl_divergence()
        kl_sum += self.conv3.kl_divergence()
        kl_sum += self.conv4.kl_divergence()

        # 全连接层的KL散度
        for layer in self.bayesian_layers:
            kl_sum += layer.kl_divergence()

        # 输出层的KL散度
        kl_sum += self.mu_layer.kl_divergence()

        return kl_sum

    def predict_with_uncertainty(self, x, n_samples=100, use_calibration=True):
        """预测时的不确定性量化"""
        self.train()  # 保持Dropout开启用于采样

        predictions = []
        noise_vars = []

        with torch.no_grad():
            for _ in range(n_samples):
                mu, noise_var = self.forward(x)
                predictions.append(mu.detach())
                noise_vars.append(noise_var.detach())

        predictions = torch.stack(predictions, dim=0)  # [n_samples, B, out_channel]
        noise_vars = torch.stack(noise_vars, dim=0)  # [n_samples, B, out_channel]

        # 预测均值
        mean_prediction = torch.mean(predictions, dim=0)  # [B, out_channel]

        # 认知不确定性
        epistemic_uncertainty = torch.var(predictions, dim=0)  # [B, out_channel]

        # 偶然不确定性
        aleatoric_uncertainty = torch.mean(noise_vars, dim=0)  # [B, out_channel]

        # 温度校准
        if use_calibration:
            with torch.enable_grad():
                mean_prediction_temp = mean_prediction.clone().detach().requires_grad_(False)
                aleatoric_uncertainty_temp = aleatoric_uncertainty.clone().detach().requires_grad_(False)
                mean_prediction, aleatoric_uncertainty = self.temperature_scaling(
                    mean_prediction_temp, aleatoric_uncertainty_temp
                )
                mean_prediction = mean_prediction.detach()
                aleatoric_uncertainty = aleatoric_uncertainty.detach()

        # 总不确定性
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

        return mean_prediction, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty


def get_kl_weight(epoch, total_epochs, strategy='aggressive'):
    """改进的动态KL权重调度"""
    if strategy == 'aggressive':
        if epoch < total_epochs * 0.6:  # 前60%轮次：极小KL权重
            return 1e-12
        elif epoch < total_epochs * 0.85:  # 60%-85%：逐步增加
            progress = (epoch - total_epochs * 0.6) / (total_epochs * 0.25)
            return 1e-12 * (1e6 ** progress)  # 从1e-12到1e-6
        else:  # 后15%：稳定KL权重
            return 1e-6
    else:
        # 原始策略
        return np.logspace(-8, -5, total_epochs)[min(epoch, total_epochs - 1)]


def adaptive_kl_weight(val_nll, val_mse, base_weight=1e-6):
    """自适应KL权重"""
    if val_nll > 2 * val_mse:
        return base_weight * 0.1
    return base_weight


def bnn_loss(mu, noise_var, target, kl_divergence, n_samples, kl_weight=1e-6, l2_weight=1e-7, model=None):
    """改进的贝叶斯神经网络损失函数"""
    # 加权MSE损失（对不同参数使用不同权重，基于重要性）
    param_weights = torch.tensor([1.2, 1.5, 1.0, 0.8, 1.3]).to(mu.device)  # E(B-V), 12+log(O/H), R3, SFR, stellar mass
    diff = target - mu
    weighted_mse = torch.mean(param_weights * (diff ** 2))

    # 负对数似然损失
    nll_loss = 0.5 * torch.sum(torch.log(2 * math.pi * noise_var) + diff ** 2 / noise_var)
    nll_loss = nll_loss / mu.size(0)  # 平均到批次

    # KL散度损失
    kl_loss = kl_divergence / n_samples

    # L2正则化
    l2_reg = 0
    if model is not None:
        l2_reg = sum(p.pow(2.0).sum() for p in model.parameters() if p.requires_grad)

    # 结合MSE和NLL损失 - 以MSE为主
    total_loss = 0.8 * weighted_mse + 0.2 * nll_loss + kl_weight * kl_loss + l2_weight * l2_reg

    return total_loss, nll_loss, kl_loss


# 测试代码
if __name__ == "__main__":
    # 测试改进的贝叶斯网络
    model = BayesianNeuralNetwork(in_channel=1, out_channel=5, spectrum_size=3516)

    # 创建随机输入
    x = torch.randn(8, 1, 3516)
    target = torch.randn(8, 5)

    # 前向传播
    mu, noise_var = model(x)
    print(f"Mu shape: {mu.shape}")
    print(f"Noise var shape: {noise_var.shape}")

    # 计算KL散度
    kl_div = model.kl_divergence()
    print(f"KL divergence: {kl_div.item()}")

    # 计算损失
    total_loss, nll_loss, kl_loss = bnn_loss(mu, noise_var, target, kl_div, n_samples=1000, model=model)
    print(f"Total loss: {total_loss.item()}")
    print(f"NLL loss: {nll_loss.item()}")
    print(f"KL loss: {kl_loss.item()}")

    # 不确定性预测
    mean_pred, epistemic, aleatoric, total = model.predict_with_uncertainty(x, n_samples=50)
    print(f"Mean prediction shape: {mean_pred.shape}")
    print(f"Epistemic uncertainty: {torch.mean(epistemic, dim=0)}")
    print(f"Aleatoric uncertainty: {torch.mean(aleatoric, dim=0)}")
    print(f"Total uncertainty: {torch.mean(total, dim=0)}")
