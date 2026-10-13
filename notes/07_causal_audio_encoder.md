
# CausalAudioEncoder 详解

> 适用模型：Wan2.2-TI2V-5B-Audio
> 最后更新：2026-03-22

---

## 一、CausalAudioEncoder 完整结构

### 1.1 类定义

**位置**：`diffsynth/models/wan_video_dit_s2v.py:322-340`

```python
class CausalAudioEncoder(nn.Module):
    def __init__(self, dim=5120, num_layers=25, out_dim=2048, num_token=4, need_global=False):
        super().__init__()
        self.encoder = MotionEncoder_tc(in_dim=dim, hidden_dim=out_dim, num_heads=num_token, need_global=need_global)
        weight = torch.ones((1, num_layers, 1, 1)) * 0.01
        self.weights = torch.nn.Parameter(weight)  # 可学习的层权重
        self.act = torch.nn.SiLU()

    def forward(self, features):
        # features: (B, 25, 1024, T)  ← MERT 的 25 层输出
        weights = self.act(self.weights)  # (1, 25, 1, 1) → SiLU 激活
        weights_sum = weights.sum(dim=1, keepdims=True)  # 逐层求和，用于归一化
        weighted_feat = ((features * weights) / weights_sum).sum(dim=1)  # (B, 1024, T)
        weighted_feat = weighted_feat.permute(0, 2, 1)  # (B, T, 1024)
        res = self.encoder(weighted_feat)  # 传给 MotionEncoder_tc
        return res  # (B, T/4, num_token, out_dim)
```

### 1.2 MERT 加权平均机制

**输入**：MERT-v1-330M 的 25 层隐状态
```
(B, 25, 1024, T)  ← 25 层，每层 1024 维，T 时间步
```

**可学习权重**：
```
self.weights: (1, 25, 1, 1)
初始化：全 0.01
训练过程中学习各层贡献度
```

**加权求和**：
```python
# 元素级乘法
weighted = features * weights  # 广播：(B, 25, 1024, T)

# 按层求和
weighted_feat = weighted.sum(dim=1)  # (B, 1024, T)  ← 25层合并为1层
```

**机制含义**：
- 初始所有层等权（0.01）
- 训练时学习各层的相对重要性
- 例：w_0=0.08, w_1=0.03, ..., w_24=0.02，权重不均匀分布

---

## 二、MotionEncoder_tc（因果音频编码器）

### 2.1 因果卷积设计

**为什么"因果"？** 在时间处理中，因果性意味着 t 时刻的输出仅依赖于 ≤t 的输入，不依赖未来信息。

**CausalConv1d 实现**（位置：`wan_video_dit_s2v.py:86-99`）：

```python
class CausalConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        # 只在左侧填充（时间轴往后），不在右侧填充
        padding = (kernel_size - 1, 0)  # (左, 右) = (2, 0)
        self.time_causal_padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):  # x: (B, C, T)
        x = F.pad(x, self.time_causal_padding, mode='replicate')  # 左侧复制填充
        return self.conv(x)  # 标准卷积
```

**例子**（kernel_size=3）：
```
原始序列:  [t-2, t-1, t, t+1, t+2, ...]
填充后:    [复, t-2, t-1, t, t+1, t+2, ...]  ← 在左侧补充 1 个token
卷积输出:  output[t] = f(padding[0], t-2, t-1) ← 只看当前及过去
          （未来 t+1, t+2 被卷积核覆盖但在时间上落在t之后，因此从不用到）
```

### 2.2 MotionEncoder_tc 详细结构

**位置**：`wan_video_dit_s2v.py:102-170`

**初始化**：
```python
class MotionEncoder_tc(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 2048, num_heads: int = 4, need_global: bool = True):
        # 局部路径：拆分成 num_heads 个分支
        self.conv1_local = CausalConv1d(1024, 2048//4 * 4, kernel_size=3, stride=1)
                           # 输出 hidden_dim//4 * num_heads = 512 * 4 = 2048

        # 全局路径：单一分支
        if need_global:
            self.conv1_global = CausalConv1d(1024, 2048//4, kernel_size=3, stride=1)
                               # 输出 hidden_dim//4 = 512

        # 层规范化
        self.norm1 = nn.LayerNorm(512, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(2048, elementwise_affine=False, eps=1e-6)

        # 下采样卷积
        self.conv2 = CausalConv1d(512, 1024, kernel_size=3, stride=2)      # /2
        self.conv3 = CausalConv1d(1024, 2048, kernel_size=3, stride=2)    # /4

        # 激活函数
        self.act = nn.SiLU()

        # 全局分支最后的线性层
        if need_global:
            self.final_linear = nn.Linear(2048, 2048)

        # Padding token（添加到 num_heads 维度）
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, 2048))
```

### 2.3 Forward 执行流程

**输入**：(B, T, 1024)  ← 来自加权 MERT

```python
def forward(self, x):
    x = rearrange(x, 'b t c -> b c t')  # (B, 1024, T)

    # ===== 局部分支 (num_heads 路径) =====
    x_local = self.conv1_local(x)  # (B, 2048, T)

    # 拆分为 4 个并行分支
    x_local = rearrange(x_local, 'b (n c) t -> (b n) t c', n=4)
    # (B, 2048, T) → (B×4, T, 512)

    x_local = self.norm1(x_local)
    x_local = self.act(x_local)

    # 第二层卷积（stride=2，下采样）
    x_local = rearrange(x_local, 'b t c -> b c t')
    x_local = self.conv2(x_local)  # (B×4, 1024, T//2)

    x_local = rearrange(x_local, 'b c t -> b t c')
    x_local = self.norm2(x_local)
    x_local = self.act(x_local)

    # 第三层卷积（stride=2，再下采样）
    x_local = rearrange(x_local, 'b t c -> b c t')
    x_local = self.conv3(x_local)  # (B×4, 2048, T//4)

    x_local = rearrange(x_local, 'b c t -> b t c')
    x_local = self.norm3(x_local)
    x_local = self.act(x_local)

    # 恢复为 (B, T/4, 4, 2048)
    x_local = rearrange(x_local, '(b n) t c -> b t n c', b=B)

    # 添加 padding token（维度变为 5）
    padding = self.padding_tokens.repeat(B, x_local.shape[1], 1, 1)
    x_local = torch.cat([x_local, padding], dim=-2)  # (B, T/4, 5, 2048)

    # ===== 全局分支 (need_global=True) =====
    if self.need_global:
        x_global = self.conv1_global(x_ori)  # (B, 512, T)
        # 同样的 3 层卷积 + norm 链
        # 最后：final_linear → (B, T/4, 2048)
        # Reshape → (B, T/4, 1, 2048)

        return x_local, x_global  # 两个输出
    else:
        return x_local  # 仅本地分支
```

### 2.4 输出形状映射

```
输入:  (B, T, 1024)

经过 3 层 stride=2 卷积：
  T → T/2 → T/4

输出（need_global=False）:
  (B, T/4, 4, 2048)  ← 4 个 token/帧，总维度 2048

输出（need_global=True）:
  audio_emb:         (B, T/4, 4, 2048)   ← CrossAttention 用
  audio_emb_global:  (B, T/4, 1, 2048)   ← AdaLayerNorm 调制用

示例（T=240）:
  输入:  (1, 240, 1024)
  输出:  (1, 60, 4, 2048) 和 (1, 60, 1, 2048)
```

### 2.5 num_audio_token=4 的设计理由

| 方面 | 说明 |
|------|------|
| **值来源** | MotionEncoder_tc 的 num_heads=4（ 拆分 4 个分支） |
| **为什么是 4** | 经验值，足以表示音频的多维特征（频率、节奏、能量等） |
| **稳定性** | 固定 token 数比变长序列更稳定（对比 MERT 原始 25 层） |
| **参数效率** | 降低计算量：CrossAttention(H×W, 4) vs (H×W, 25) |
| **类比** | 类似 IP-Adapter 的固定 image token 数 |

---

## 三、音频形状变换总结

```
MERT 提取:
  原始音频 24kHz → MERT-v1-330M
  输出：(25 层, 75fps, 1024 维)

线性插值 (75fps → 30fps):
  (25, 75, 1024) → (25, 30, 1024)
  对应于视频 30fps 采样率

进入 CausalAudioEncoder:
  1. 加权平均 25 层
     (25, 30, 1024) → (1, 30, 1024)

  2. MotionEncoder_tc 处理
     (1, 30, 1024) → (1, 7-8, 4, 2048) + (1, 7-8, 1, 2048)
     （30 / 4 ≈ 7.5，向上取整 8）

最终输出:
  audio_emb:        (B, T_compressed, 4, dim)
  audio_emb_global: (B, T_compressed, 1, dim)

  其中 T_compressed = T_30fps // 4
```

---

## 四、关键代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| CausalAudioEncoder 定义 | wan_video_dit_s2v.py | 322-340 |
| MotionEncoder_tc 定义 | wan_video_dit_s2v.py | 102-170 |
| CausalConv1d 定义 | wan_video_dit_s2v.py | 86-99 |
| 音频初始化调用 | wan_video_dit_audio.py | ~120 |
| 加权 MERT 使用 | diffsynth/pipelines/wan_video.py | ~850 |
