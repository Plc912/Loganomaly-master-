# LogAnomaly MCP 封装说明文档

原作者：Openaiops

项目原地址：[AIOps-NanKai / model / LogAnomaly · 极狐GitLab](https://www.aiops.cn/gitlab/aiops-nankai/model/loganomaly)

MCP封装作者：庞力铖

GitHub：

## 📋 项目概述

**LogAnomaly** 是一个基于双向LSTM（Bidirectional LSTM）的深度学习日志异常检测系统，通过MCP（Model Context Protocol）封装，提供自然语言交互接口。

### 核心特性

- ✅ **双向LSTM模型**: 强大的序列建模能力
- ✅ **语义增强**: Word2Vec + 同义词/反义词信息
- ✅ **增量学习**: 支持在线学习和模板匹配
- ✅ **候选集机制**: Top-N候选提高容错性
- ✅ **完整流程**: 9步全流程自动化---

## 🔧 环境要求

### Python 版本

- **要求**: Python 3.11+ （已适配 Python 3.11.13）
- **推荐**: Python 3.11.13 或 3.12+
- **说明**:
  - 原项目使用 Python 3.8 和 TensorFlow 2.5
  - MCP 库要求 Python >= 3.11
  - 已升级 TensorFlow 到 2.15+ 以支持 Python 3.11

### 安装步骤

**推荐方式**: 使用虚拟环境

```bash
# 创建虚拟环境
python3.11 -m venv loganomaly_env
source loganomaly_env/bin/activate  # Linux/Mac
loganomaly_env\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

**生产环境**: 使用 Docker 隔离环境

### 系统依赖

#### 1. C 编译环境（必需）

用于编译 LRCWE 词向量训练程序

**Windows**:

- MinGW-w64: https://www.mingw-w64.org/
- 或 Visual Studio Build Tools

**Linux**:

```bash
sudo apt-get install build-essential
```

**Mac**:

```bash
xcode-select --install
```

#### 2. NLTK WordNet 数据（可选）

用于搜索同义词和反义词

```bash
python -m nltk.downloader wordnet omw-1.4
```

---

## 📊 数据格式要求

### 1. 原始日志文件 (`.log`)

**格式**: 纯文本文件，每行一条日志

**示例**:

```
2023-10-01 10:23:45 node-1 kernel: INFO task completed successfully
2023-10-01 10:23:46 node-2 kernel: WARNING high memory usage detected
2023-10-01 10:23:47 node-1 kernel: ERROR connection timeout
```

**要求**:

- ✅ **无列头**: 纯日志文本，无 CSV 表头
- ✅ **每行一条**: 不能有多行日志
- ✅ **时间戳**: 建议包含时间信息（第一列）
- ✅ **结构化**: 日志应有一定的结构（便于模板提取）

**常见日志类型**:

- ✅ BGL（Blue Gene/L）超算日志
- ✅ HDFS 分布式文件系统日志
- ✅ Apache/Nginx Web 服务器日志
- ✅ 系统日志（syslog）

### 2. 标签文件 (`.label`)

**格式**: 纯文本文件，每行一个标签

**示例**:

```
0
0
1
0
1
```

**要求**:

- ✅ **无列头**: 纯数字，无表头
- ✅ **行数匹配**: 必须与日志文件行数完全一致
- ✅ **标签值**:
  - `0` = 正常日志
  - `1` = 异常日志
- ✅ **编码**: UTF-8（推荐）

### 3. 数据集示例

**BGL 数据集**:

```
原始日志: bgl.log（4,747,963 条日志）
标签文件: bgl.label（4,747,963 行）
异常率: ~0.5%
```

**HDFS 数据集**:

```
原始日志: hdfs.log（11,175,629 条日志）
标签文件: hdfs.label（11,175,629 行）
异常率: ~3%
```

---

## 📦 安装与配置

### 1. 安装依赖

```bash
# 进入项目目录
cd E:\software\MCP__Proj\log\loganomaly-master

# 安装 Python 依赖
pip install -r requirements.txt

# 下载 NLTK 数据（可选）
python -m nltk.downloader wordnet omw-1.4
```

### 2. 编译 C 程序

```bash
# 进入词向量训练目录
cd template2Vector/src

# 编译
make

# 验证编译结果
ls -l lrcwe  # Linux/Mac
dir lrcwe.exe  # Windows
```

### 3. 配置 MCP 客户端

将 `loganomaly_config_example.json` 的内容添加到你的 MCP 客户端配置文件中。

**Cursor IDE** 配置路径:

- Windows: `%APPDATA%\Cursor\User\globalStorage\rooveterinaryinc.roo-cline\settings\cline_mcp_settings.json`
- Mac: `~/Library/Application Support/Cursor/User/globalStorage/rooveterinaryinc.roo-cline/settings/cline_mcp_settings.json`
- Linux: `~/.config/Cursor/User/globalStorage/rooveterinaryinc.roo-cline/settings/cline_mcp_settings.json`

### 4. 启动服务器

**Windows**:

```cmd
python loganomaly_mcp_server.py
```

**Linux/Mac**:

```bash
python3 loganomaly_mcp_server.py
```

---

## 💬 自然语言使用话术

### 完整流程（推荐新手）

```
帮我对 BGL 日志进行完整的异常检测分析，日志文件是 data/bgl.log，标签文件是 data/bgl.label
```

### 分步执行（高级用户）

#### 步骤1: 提取模板

```
从 data/bgl.log 中提取日志模板，保存到 middle 目录
```

#### 步骤2: 匹配模板

```
将 data/bgl.log 与模板 middle/bgl_log.template 匹配，生成日志序列
```

#### 步骤3: 过滤正常日志

```
从 middle/bgl_log.seq 中过滤出正常日志，标签文件是 data/bgl.label
```

#### 步骤4: 搜索同义词反义词

```
基于 middle/bgl_log.template 搜索同义词和反义词
```

#### 步骤5: 转换模板格式

```
将 middle/bgl_log.template 转换为词向量训练格式
```

#### 步骤6: 训练词向量

```
使用 LRCWE 算法训练词向量，输入文件在 middle 目录
```

#### 步骤7: 生成模板向量

```
从词向量模型生成模板向量
```

#### 步骤8: 训练LSTM模型

```
使用正常日志训练双向LSTM异常检测模型，训练30个epoch
```

#### 步骤9: 异常检测

```
使用训练好的模型检测 middle/bgl_log.seq 中的异常，使用15个候选
```

### 任务管理

#### 查看任务列表

```
查看所有 LogAnomaly 任务
列出所有后台任务
```

#### 查看任务详情

```
查看任务 <task_id> 的详细信息
获取任务进度
```

#### 取消任务

```
取消任务 <task_id>
停止正在运行的任务
```

---

## 🎯 使用场景与数据要求

### 适用场景

#### ✅ 推荐使用

1. **复杂日志语义**: 需要理解日志语义，而不仅仅是模式匹配
2. **动态环境**: 日志模板会随时间变化
3. **高精度要求**: 需要精细的异常检测
4. **大规模日志**: 百万级以上日志量

#### ❌ 不太适合

1. **简单日志模式**: 如果只是统计分析，LightAD更快
2. **实时性要求极高**: LSTM推理速度不如TCN
3. **资源受限环境**: 需要更多计算资源
4. **极小数据集**: 少于1000条日志效果不佳

### 数据规模建议

| 数据集大小        | 训练时间     | 推荐epochs | 推荐seq_length |
| ----------------- | ------------ | ---------- | -------------- |
| < 10,000          | 5-10分钟     | 10-20      | 5-10           |
| 10,000-100,000    | 10-30分钟    | 20-30      | 10-15          |
| 100,000-1,000,000 | 30分钟-2小时 | 30-50      | 10-20          |
| > 1,000,000       | 2-8小时      | 30-50      | 10-20          |

### 数据质量要求

#### 1. 日志结构化程度

- ✅ **高**: 有明确的字段分隔（如 Apache 日志）
- ⚠️ **中**: 有一定结构但不规范
- ❌ **低**: 自由文本，无明显结构

#### 2. 异常比例

- ✅ **推荐**: 0.1% - 10%
- ⚠️ **可接受**: 10% - 30%
- ❌ **不推荐**: > 30%（训练困难）

#### 3. 模板数量

- ✅ **推荐**: 10 - 1000 个模板
- ⚠️ **可接受**: 1000 - 5000 个模板
- ❌ **过多**: > 5000 个模板（训练缓慢）

---

## ⚙️ 参数调优指南

### 序列长度 (`seq_length`)

**默认值**: 10

**调整建议**:

- **短日志**: 5-10（如系统日志）
- **长日志**: 15-30（如应用日志）
- **权衡**: 更长的序列能捕获更多上下文，但训练更慢

### 训练轮数 (`epochs`)

**默认值**: 30

**调整建议**:

- **小数据集**: 10-20
- **中数据集**: 20-40
- **大数据集**: 30-50
- **观察**: 监控loss，提前停止过拟合

### 候选集大小 (`n_candidates`)

**默认值**: 15

**调整建议**:

- **高精度**: 5-10（更严格）
- **平衡**: 10-20（推荐）
- **高召回**: 20-30（更宽松）

### 独热编码 (`use_onehot`)

**默认值**: True

**选择建议**:

- **True**: 适合模板数量 < 1000
- **False**: 适合模板数量 > 1000（使用词向量）

### 计数矩阵 (`use_count_matrix`)

**默认值**: True

**选择建议**:

- **True**: 能捕获模板共现信息（推荐）
- **False**: 仅使用模板向量（更快）---

## 🚨 常见问题与解决方案

### 1. Python 版本要求

**说明**: LogAnomaly MCP 已适配 Python 3.11+

**安装依赖**:

```bash
# 确保使用 Python 3.11+
python --version  # 应显示 3.11.13 或更高

# 安装依赖
pip install -r requirements.txt

# 如果遇到版本冲突，尝试升级 pip
python -m pip install --upgrade pip
```

### 2. C 编译失败

**问题**: `make: gcc: command not found`

**解决方案**:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential

# Mac
xcode-select --install

# Windows
# 下载安装 MinGW-w64 或 Visual Studio Build Tools
```

### 3. WordNet 数据缺失

**问题**: `LookupError: Resource wordnet not found`

**解决方案**:

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 4. 内存不足

**问题**: `MemoryError` 或系统卡死

**解决方案**:

- 减小 `batch_size`（默认64 → 32或16）
- 减小 `seq_length`
- 减小训练集大小
- 增加系统内存或使用更大内存的机器

### 5. 训练过慢

**问题**: 训练时间过长

**解决方案**:

- 使用 GPU 版本的 TensorFlow

```bash
pip install tensorflow-gpu==2.5.0
```

- 减小 `epochs`
- 增大 `batch_size`
- 使用更少的训练数据

### 6. 检测精度低

**问题**: Precision/Recall/F1 分数很低

**解决方案**:

- 增加训练数据量
- 调整 `n_candidates`
- 增加 `epochs`
- 检查数据标签是否正确
- 尝试不同的 `seq_length`

### 7. 模板提取失败

**问题**: 生成的模板数量异常（过多或过少）

**解决方案**:

- 检查日志格式是否规范
- 调整 FT-tree 参数（需修改源码）
- 预处理日志（去除噪声字段）

---

## 📈 性能指标参考

### BGL 数据集

| 指标      | 数值     | 说明        |
| --------- | -------- | ----------- |
| Precision | 0.95+    | 精确率      |
| Recall    | 0.90+    | 召回率      |
| F1-Score  | 0.92+    | F1分数      |
| 训练时间  | 2-4小时  | GTX 1080 Ti |
| 检测时间  | 5-10分钟 | 1M条日志    |

### HDFS 数据集

| 指标      | 数值      | 说明        |
| --------- | --------- | ----------- |
| Precision | 0.98+     | 精确率      |
| Recall    | 0.95+     | 召回率      |
| F1-Score  | 0.96+     | F1分数      |
| 训练时间  | 1-2小时   | GTX 1080 Ti |
| 检测时间  | 10-15分钟 | 2M条日志    |

---

## 🔬 技术细节

### 双向LSTM架构

```
输入层 (seq_length, vector_dim)
    ↓
双向LSTM层 (128单元) × 2
    ↓
Dropout层 (0.2)
    ↓
拼接层 (vector分支 + count分支)
    ↓
全连接层 (n_templates)
    ↓
Softmax激活
    ↓
输出层 (预测下一个模板)
```

### LRCWE 词向量算法

**特点**:

- 基于 Word2Vec CBOW 模型
- 引入同义词约束（相似性损失）
- 引入反义词约束（对比损失）
- 多任务学习

**损失函数**:

```
Loss = α_rel · L_rel + α_syn · L_syn + α_ant · L_ant

其中:
- L_rel: 关系损失（上下文预测）
- L_syn: 同义词损失
- L_ant: 反义词损失
```

### FT-tree 日志解析

**特点**:

- 基于频繁子树挖掘
- 支持增量学习
- 自动模板提取

---

## 📚 参考资料

### 论文

- **LogAnomaly**: Du, M., Li, F., Zheng, G., & Srikumar, V. (2017). "DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning." *CCS 2017*.

### 相关项目

- **FT-tree**: https://github.com/logpai/logparser
- **Word2Vec**: https://code.google.com/archive/p/word2vec/
- **DeepLog**: https://github.com/wuyifan18/DeepLog

### 数据集

- **LogHub**: https://github.com/logpai/loghub
  - BGL
  - HDFS
  - Spark
  - Hadoop
  - OpenStack

---

## 🎉 开始使用

1. ✅ 安装依赖: `pip install -r requirements.txt`
2. ✅ 编译C程序: `cd template2Vector/src && make`
3. ✅ 下载NLTK数据: `python -m nltk.downloader wordnet omw-1.4`
4. ✅ 启动服务器: `python3 loganomaly_mcp_server.py`（或 `python loganomaly_mcp_server.py`）
5. ✅ 配置MCP客户端: 复制 `loganomaly_config_example.json` 内容
6. ✅ 开始使用: 发送自然语言指令！
