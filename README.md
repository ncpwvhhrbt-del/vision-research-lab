# Vision Research Lab

一个面向计算机视觉算法岗的可复现研究项目，重点研究卷积神经网络在准确性、鲁棒性、可信度与效率之间的权衡。

> 当前状态：Phase 3 完成 - 已完成可复现训练、消融实验、鲁棒性、校准和高置信错误分析。

## 项目目标

本项目不以网页展示或单次训练结果为终点，而是建立一个可以持续扩展的视觉实验平台。核心任务是：在统一数据、训练预算和评测协议下，对模型结构、训练策略、鲁棒性、置信度校准和模型压缩方法进行公平比较。

首个研究问题：

> 在有限显存和训练预算下，哪些训练与模型改进能同时提升图像分类的准确率、分布偏移鲁棒性和置信度可靠性？

## 首期研究范围

- 任务：通用图像分类
- 数据：CIFAR-10 起步，随后扩展至 CIFAR-100
- 基线：简单 CNN、ResNet 系列
- 重点：数据增强、损失函数、正则化、鲁棒性、校准、蒸馏和轻量化
- 暂不重点研究：Transformer、大型前后端系统、超大规模预训练
- 计算约束：单张 NVIDIA RTX 4050 Laptop GPU，6GB 显存

## 预期成果

- 可复现的训练与评测流程
- 统一实验配置与结果记录
- 基线模型排行榜
- 消融实验与错误案例分析
- 鲁棒性和校准评测报告
- 面向 GitHub 与求职作品集的完整 README 和技术报告

## 仓库结构

```text
vision-research-lab/
├── .github/          # Issue 与 PR 模板
├── benchmarks/       # 基线和研究结果清单
├── configs/          # 数据、模型与实验配置
├── data/             # 本地数据目录，原始数据不进入 Git
├── docs/             # 研究问题、系统设计、协议与路线图
├── notebooks/        # 仅用于探索与可视化
├── outputs/          # 日志、权重、指标与预测结果，不进入 Git
├── reports/          # 可上传的图表、表格和研究报告
├── scripts/          # 后续命令入口
├── src/              # 后续正式源码
└── tests/            # 单元、集成与冒烟测试
```

## 研究原则

1. 先建立可信基线，再提出改进。
2. 每个实验只改变少量变量。
3. 不只报告最高准确率，还报告均值、波动、效率和失败案例。
4. 训练集统计不得泄漏到验证集或测试集。
5. 所有图表和结论都能追溯到实验配置与原始指标。
6. Notebook 不作为最终训练入口。

## 文档入口

- [项目章程](docs/00_PROJECT_CHARTER.md)
- [研究问题](docs/01_RESEARCH_QUESTIONS.md)
- [系统设计](docs/02_SYSTEM_DESIGN.md)
- [实验协议](docs/03_EXPERIMENT_PROTOCOL.md)
- [开发路线图](docs/04_ROADMAP.md)
- [作品集标准](docs/05_PORTFOLIO_STANDARD.md)
- [贡献规范](CONTRIBUTING.md)

## 当前里程碑

- [x] 项目定位
- [x] 仓库目录设计
- [x] 研究问题与评测协议
- [x] 环境与依赖定义
- [x] 数据模块
- [x] 基线训练管线
- [x] 统一评测模块
- [x] 第一轮可复现实验
- [x] 学习率调度消融
- [x] 数据增强消融
- [x] 类别级误差分析
- [x] 鲁棒性评测
- [x] ECE、Brier Score 与可靠性图
- [x] 温度缩放
- [x] 高置信错误分析

## 最终报告

- [最终技术报告](docs/FINAL_REPORT.md)
- [学习率策略比较](benchmarks/SCHEDULER_COMPARISON.md)
- [数据增强比较](benchmarks/AUGMENTATION_COMPARISON.md)
- [可靠性图](reports/crop_flip_reliability.png)

## License

MIT License。数据集和预训练权重仍遵循其各自许可证。
