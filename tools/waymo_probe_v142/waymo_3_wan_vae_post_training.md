# Waymo 3-Channel Wan LiDAR VAE Post-Training Note

## 1. 结论

这条路线可以做，而且比“单通道强改 Wan 输入输出层”更省事。

第一版最推荐的定义是：

- 数据源：Waymo 官方 `TFRecord`
- LiDAR：`TOP` lidar
- return：只用 `first return`
- 输入通道：`range + intensity + valid_mask`
- VAE：直接复用 `Wan2.1` 原生 `3-channel in / 3-channel out` 结构
- latent 目标：保持和 video VAE 一样的时空压缩比，尽量得到 `29 raw frames -> 8 latent frames`
- joint training 前：做 `ULA / affine latent alignment`

这版的核心优点不是“3 通道所以能 concat”，而是：

1. 你可以最大化复用 Wan 原生 VAE 权重，不需要先改首尾层 shape。
2. `range + intensity + valid_mask` 比伪 RGB 更贴近 LiDAR 物理语义。
3. 后续如果它稳定，接入 CoLiGen 会比当前 `CI8x8 + 29 -> 8 temporal adapter` 更干净。

---

## 2. 先钉死的设计决策

下面这几个点必须先定，不然后面实现会来回改。

### 2.1 不是所有 3 通道方案都一样

这里建议第一版固定为：

- channel 1: `range`
- channel 2: `intensity`
- channel 3: `valid_mask`

不建议第一版直接用：

- `range + intensity + elongation`
- `x + y + z`
- 把 first / second return 全部堆成 6 到 8 通道

原因很简单：

- `valid_mask` 对稀疏观测更稳定
- 更适合作为 reconstruction 里的 occupancy / alpha 信号
- 不需要改 Wan 的通道数

`elongation` 建议留成后续 ablation。

### 2.2 和 video latent 对齐靠什么

这点要写得非常明确：

**是否能和 video latent 融合，不取决于输入是不是 3 通道，而取决于下面四件事：**

- latent `shape`
- `temporal / spatial stride`
- latent normalization
- `ULA`

所以做 3 通道的真正意义是：

- 让 LiDAR 输入接口和 Wan 图像 / 视频 VAE 的原生输入更接近
- 让你更容易继承 pretrained weights

而不是“因为 3 通道所以自然能 concat”。

### 2.3 第一版只用 TOP lidar + first return

第一版先固定：

- `LaserName.TOP`
- `first return`

这样做的原因：

- 最容易先跑通
- 时序稳定性最好
- 不会把 second return、多激光器融合、不同投影策略混在一起

后续如果第一版有效，再考虑：

- second return 做辅助监督
- 或者把多激光器统一投影成你自己的 panoramic grid

---

## 3. 关于 TFRecord 直读

## 3.1 可行，但它不是“零成本更换数据源”

直接读 Waymo 官方 `TFRecord` 是可行的，但和当前 tar 版相比，会新增三个工程块：

1. `TFRecord -> Frame` 解码
2. 从压缩 LiDAR 表征恢复 range image
3. clip 级 frame sampling / cache / index 构建

也就是说，TFRecord 直读不只是“把 dataloader path 改一下”，而是要新增一整层数据读取逻辑。

## 3.2 当前环境里缺依赖

我查过当前环境：

- `tensorflow` 未安装
- `waymo_open_dataset` 未安装

所以如果要走 TFRecord 直读，文档里必须把“依赖安装”和“索引缓存”明确写进方案，否则执行时会卡住。

## 3.3 推荐的工程策略

建议不要把 TFRecord 直读做成“每个 epoch 从头扫完整文件”。

推荐做法：

1. 先扫描所有 `TFRecord`
2. 生成一个 `frame index cache`
3. 训练时按 cache 随机访问 clip

这个 cache 至少要记录：

- `segment path`
- `frame offset`
- `timestamp`
- `scene / segment id`
- 可选的 ego pose / calibration shortcut

否则 dataloader 吞吐会很差。

---

## 4. 输入定义

## 4.1 第一版的输入张量

建议定义为：

```text
B x 3 x T x H x W
```

其中：

- `B`: batch
- `3`: `range, intensity, valid_mask`
- `T = 29`
- `H, W`: 由你最终选择的 range-image 表达决定

## 4.2 关键分叉：用官方 TOP 原生 range image，还是统一投影网格

这里是最需要提前决定的点。

### 方案 A：保留 Waymo TOP lidar 原生 range image

优点：

- 最接近 TFRecord 原始观测
- 不需要自己重新投影点云
- 信息损失最少

缺点：

- shape 会受官方 range image 规范约束
- 未必和你当前 `128 x 3600 -> 512 x 1800/896/1792` 这条旧管线完全一致
- 后续和现有 LiDAR repo / joint pipeline 的对接要多做一次 shape 适配

### 方案 B：从 TFRecord 解码点云 / range features 后，再统一投影到你当前 panoramic grid

优点：

- 可以继续沿用你熟悉的 `128 x 3600 -> row repeat x4 -> crop`
- 更容易和现有 LiDAR tokenizer / joint training 对比
- 更容易直接对齐当前 CoLiGen 方案

缺点：

- 多了一次“官方表示 -> 自定义投影”的转换
- 工程实现更长

### 我的建议

如果你的目标是“尽快比较 3-channel Wan LiDAR VAE 和现有 CI8x8 baseline”：

- **优先方案 B**

如果你的目标是“尽量忠实保留官方 TFRecord 里的原始观测定义”：

- **走方案 A**

从你现在的项目上下文看，我更建议 **方案 B**，因为它更容易和现有 CoLiGen 主线对齐。

---

## 5. 为什么这次可以不改 Wan 的首尾层 shape

这版和你之前单通道方案最大的不同在这里。

如果输入输出都定义成：

- input: `3 channels`
- output: `3 channels`

那么 Wan2.1 原生 VAE 的接口正好就是：

- encoder 第一层 `3 -> hidden`
- decoder 最后一层 `hidden -> 3`

也就是说，**第一版不需要为了通道数去改模型结构**。

你可以直接：

- 完整加载 Wan2.1 VAE 权重
- 保持 backbone 不变
- 先只做 LiDAR domain adaptation

这比“单通道版本先做权重 surgery”更简单，也更稳。

## 5.1 这不代表不用微调

虽然不用改通道数，但仍然需要微调，因为：

- LiDAR 统计和 RGB 差别很大
- `valid_mask` 这种 occupancy-like 通道不是天然 RGB 纹理
- 时序结构也和自然视频不同

所以推荐的训练流程还是：

### Stage A: 轻量适配

- 先冻结大部分 backbone
- 只放开浅层和输出头附近
- 让模型先学会“这不是 RGB，但仍然是 3 通道结构化观测”

### Stage B: 全量微调

- 再解冻全模型
- 把 reconstruction 和时序稳定性训上来

---

## 6. TFRecord 直读的数据流

第一版推荐的数据流写成：

```text
Waymo TFRecord
  -> parse Frame
  -> extract TOP lidar first-return range image / features
  -> project to target range grid
  -> build 3 channels:
       [range, intensity, valid_mask]
  -> temporal clip sampling (29 frames)
  -> normalization
  -> Wan LiDAR VAE
```

如果走你当前更熟悉的 panoramic grid 规范，可以写成：

```text
Waymo TFRecord
  -> parse point/range features
  -> project to 128 x 3600
  -> downsample width: 3600 -> 1800
  -> repeat rows: 128 -> 512
  -> crop width: 896 or 1792
  -> T = 29
  -> build [range, intensity, valid_mask]
  -> B x 3 x 29 x 512 x W
```

---

## 7. latent 目标

这部分仍然建议和当前 CoLiGen 主线对齐：

- temporal compression: `4x`
- spatial compression: `8x8`

如果输入是：

- `B x 3 x 29 x 512 x 896`

那么目标 latent shape 是：

- `B x C_z x 8 x 64 x 112`

如果输入是：

- `B x 3 x 29 x 512 x 1792`

那么目标 latent shape 是：

- `B x C_z x 8 x 64 x 224`

其中 `C_z` 尽量保持和 Wan video VAE 一致。

---

## 8. 损失建议

第一版先用简单、稳定、好实现的目标。

推荐：

```text
L = lambda_r * L_range
  + lambda_i * L_intensity
  + lambda_m * L_mask
  + lambda_kl * L_KL
  + lambda_p * L_perc
```

建议的默认定义：

- `L_range`: valid mask 内的 `L1` 或 `Charbonnier`
- `L_intensity`: valid mask 内的 `L1`
- `L_mask`: `BCE`
- `L_KL`: 标准 VAE KL
- `L_perc`: LPIPS，对 `range/intensity/mask` 拼成的 3 通道图像计算

一个保守的初始权重可以是：

- `lambda_r = 1.0`
- `lambda_i = 0.3`
- `lambda_m = 0.2`
- `lambda_kl = 1e-6 ~ 1e-5`
- `lambda_p = 0.1 ~ 0.3`

## 8.1 第一版先不加的东西

先不加：

- GAN loss
- 点云 Chamfer 反传
- second return reconstruction
- 多激光器融合 loss

原因：

- 先把 TFRecord 直读跑通
- 先把 3-channel Wan LiDAR VAE 训稳
- 先得到一个可导出的 tokenizer checkpoint

---

## 9. 训练建议

## 9.1 第一阶段：small pilot

先做一个低风险 pilot：

- source: TFRecord
- lidar: TOP
- return: first return
- grid: 先统一到你当前 panoramic grid
- input: `29 x 512 x 896`
- batch: 小 batch
- 目标：验证收敛、检查 recon、确认 latent shape

## 9.2 第二阶段：主训练

pilot 成功后再扩大到：

- 更长训练
- 更稳定的 validation
- optional `1792` 宽度 refinement

## 9.3 joint training 接入前必须检查

必须满足：

1. latent time dim 真的是 `8`
2. recon 不明显崩
3. latent normalization 已稳定
4. `ULA` 统计量已导出

---

## 10. 接入 CoLiGen 时要怎么想

如果这版 3-channel Wan LiDAR VAE 成功，后续接 CoLiGen 的思路是：

1. 用新 tokenizer 提 LiDAR latents
2. 做 `ULA / affine alignment`
3. 再和 video latents 融合

这里建议的表述是：

**“3-channel LiDAR VAE 解决的是 LiDAR tokenizer 与 Wan backbone 的接口兼容性；真正解决跨模态融合稳定性的，仍然是 latent shape 对齐 + normalization + ULA。”**

这句话最好在文档里一直保持，不然后面很容易把“3-channel”误当成“跨模态自然对齐”。

---

## 11. 我对这版的总评价

如果你现在想在两个方向里二选一：

- 单通道 UniDriveDreamer-style LiDAR VAE
- 三通道 TFRecord-direct Wan LiDAR VAE

我会这样判断：

- **想更贴论文**：单通道方案更正宗
- **想更快开工、更多继承 Wan 原始权重**：三通道方案更实用

所以这版不是不行，反而是个很不错的工程版路线。

我唯一建议你一定先写清楚的两句话是：

1. 第一版固定 `TOP + first return + range/intensity/valid_mask`
2. 第一版优先把 TFRecord 解码后的数据重新投影到你当前的 `128 x 3600 -> 512 x ...` 规范上

只要这两句写死，这个方案就基本能直接往下实现了。
