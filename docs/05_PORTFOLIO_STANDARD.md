# GitHub 作品集标准

## 招聘者应在两分钟内看到

- 项目解决的问题
- 为什么这个问题重要
- 基线与改进方法
- 最主要的量化结果
- 一张核心结果图
- 如何复现实验
- 你的真实贡献与项目限制

## README 最终应包含

- 项目摘要
- 核心结果表
- 方法概览图
- 数据与实验设置
- 快速开始
- 主要实验与消融
- 错误案例
- 仓库结构
- 复现说明
- 局限与后续工作

## 提交记录标准

推荐使用小而明确的提交，例如：

- `docs: define baseline experiment protocol`
- `feat: add cifar10 data module`
- `test: add smoke test for one training step`
- `exp: compare sgd and adam baselines`
- `report: add calibration study results`

避免把整个项目一次性提交为 `first commit`，因为清晰的提交历史本身也是工程能力证据。

## 不应进入 GitHub

- 数据集原文件
- 大型 checkpoint
- TensorBoard 日志
- 本机绝对路径
- API 密钥和账号信息
- 临时图片和无解释的实验文件
- 无法复现的“最好结果”

## 面试表达框架

1. 问题：为什么只看准确率不够？
2. 基线：如何保证比较公平？
3. 改进：具体改变了什么机制？
4. 证据：指标、消融和重复实验说明什么？
5. 失败：什么方法没有奏效，原因是什么？
6. 权衡：精度、鲁棒性、校准和速度如何取舍？
