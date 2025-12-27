# CUREBench 生物医学AI竞赛解决方案

[![ProjectPage](https://img.shields.io/badge/CUREBench-Page-red)](https://curebench.ai) [![ProjectPage](https://img.shields.io/badge/CUREBench-Kaggle-green)](https://www.kaggle.com/competitions/cure-bench)

## 项目概述

本项目为 **CUREBench 生物医学AI竞赛**的完整解决方案，基于 **GRPO (Group Relative Policy Optimization)** 算法对轻量级语言模型进行全参数微调，专门针对药理学、临床医学和药物安全领域的问答任务进行优化。

### 核心贡献

- **轻量级高效方案**: 基于 Qwen3-0.6B (约6亿参数)，在消费级显卡(RTX 3090 24GB)上完成训练
- **GRPO全参数训练**: 采用组相对策略优化算法，通过自定义奖励函数引导模型学习
- **结构化推理输出**: 强制 `<think>...</think><answer>X</answer>` 格式，提升推理可解释性
- **医学领域适配**: 通过Few-shot提示词和奖励函数设计，强化医学术语和逻辑推理能力

---

## 方法论

### 1. 训练算法：GRPO (Group Relative Policy Optimization)

GRPO 是一种基于组的强化学习策略优化算法，相比传统PPO更适合语言模型微调：

```
                    ┌─────────────────────────────────────────┐
                    │              GRPO 训练流程               │
                    └─────────────────────────────────────────┘

┌──────────┐      ┌──────────────┐      ┌─────────────────┐      ┌──────────┐
│  Prompt  │ ───▶ │ 生成8个候选  │ ───▶ │  奖励函数打分   │ ───▶ │ 策略更新 │
│  输入    │      │   响应       │      │  (组内排序)     │      │          │
└──────────┘      └──────────────┘      └─────────────────┘      └──────────┘
                         │                      │
                         ▼                      ▼
                  ┌─────────────────────────────────────────┐
                  │ 组内相对比较，选择最优响应作为正样本     │
                  │ 其他响应作为负样本，计算对比损失         │
                  └─────────────────────────────────────────┘
```

**关键参数**:
- `num_generations = 8`: 每个prompt生成8个候选响应
- 通过组内相对排序确定正负样本，无需外部奖励模型

### 2. 奖励函数设计

自定义奖励函数综合评估格式合规性、答案正确性和推理质量：

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        奖励函数结构 (总分: 0-2.5分)                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ 1. 格式合规性 (0-0.5分)                                              │ │
│  │    ├─ <think>标签包含有效内容 (>10字符): +0.3分                      │ │
│  │    └─ <answer>标签包含A-E选项: +0.2分                                │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ 2. 答案正确性 (0-1.5分)                                              │ │
│  │    ├─ 正确答案: +1.5分                                               │ │
│  │    ├─ 错误答案: +0.0分                                               │ │
│  │    └─ 缺少<answer>标签: -0.3分 (多选题)                              │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ 3. 推理质量 (0-0.3分)                                                │ │
│  │    ├─ 医学术语 (≥3个): +0.2分                                        │ │
│  │    │   关键词: patient, treatment, drug, medication, symptom,        │ │
│  │    │          diagnosis, dose, effect, clinical, adverse...          │ │
│  │    └─ 逻辑连接词: +0.1分                                             │ │
│  │        关键词: first, second, therefore, because, since, thus        │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ 4. 长度惩罚                                                          │ │
│  │    └─ 输出<20字符: -0.5分                                            │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**奖励函数代码实现**:
```python
def reward_function(prompts, completions, ground_truths, question_types):
    rewards = []
    for completion in completions:
        reward = 0.0

        # 1. 格式合规性检查
        reward += check_format_compliance(completion)  # 0-0.5

        # 2. 答案正确性 (仅从<answer>标签提取)
        extracted_answer = extract_answer_from_response(completion)
        if extracted_answer == ground_truth:
            reward += 1.5
        elif not extracted_answer:
            reward -= 0.3  # 缺少标签惩罚

        # 3. 推理质量评估
        reward += evaluate_reasoning_quality(completion)  # 0-0.3

        # 4. 长度惩罚
        if len(completion) < 20:
            reward -= 0.5

        rewards.append(max(0.0, min(2.5, reward)))
    return rewards
```

### 3. 提示词工程

采用医学专家系统提示词 + 2-shot示例的策略：

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           System Prompt 结构                             │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ 角色定义:                                                          │  │
│  │ "You are a medical expert assistant specializing in                │  │
│  │  pharmacology, clinical medicine, and drug safety."                │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ 输出格式要求:                                                      │  │
│  │ <think>                                                            │  │
│  │ Your step-by-step reasoning process here.                          │  │
│  │ </think>                                                           │  │
│  │ <answer>                                                           │  │
│  │ The option letter (A, B, C, D, or E) here.                         │  │
│  │ </answer>                                                          │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Few-shot 示例 (2个):                                               │  │
│  │ • Example 1: 痤疮治疗药物识别 (水杨酸)                             │  │
│  │ • Example 2: 哺乳期用药考虑 (沙丁胺醇)                             │  │
│  │                                                                    │  │
│  │ 示例来源: 从训练集中选取，训练时排除以避免数据泄露                 │  │
│  │ 排除ID: ["U9PHZ83RKYV8", "wzkMQ7uHtlLs"]                           │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 模型架构

### 基座模型: Qwen3-0.6B

| 参数 | 值 |
|------|-----|
| **模型类型** | Qwen3ForCausalLM |
| **隐藏层大小** | 1024 |
| **层数** | 28 |
| **注意力头数** | 16 |
| **KV头数** | 8 (GQA) |
| **头维度** | 128 |
| **中间层大小** | 3072 |
| **最大位置编码** | 40960 |
| **词表大小** | 151936 |
| **激活函数** | SiLU |
| **数据类型** | bfloat16 |
| **总参数量** | ~600M |

### 训练后模型

```
models/
├── Qwen3-0.6B/                          # 基座模型
├── Qwen3-0.6B-GRPO-CUREBench/           # GRPO训练后 (3 epochs)
│   └── checkpoint-114/                  # 最佳检查点
└── Qwen3-0.6B-GRPO-CUREBench-2epoch/    # GRPO训练后 (2 epochs)
    └── checkpoint-228/
```

---

## 训练配置

### 硬件环境

| 配置项 | 详情 |
|--------|------|
| **GPU** | NVIDIA RTX 3090 24GB |
| **精度** | bfloat16 |
| **梯度检查点** | 启用 (节省显存) |

### 训练超参数

```json
{
  "training": {
    "model_path": "models/Qwen3-0.6B",
    "dataset_path": "curebench_valset_pharse1.jsonl",
    "output_dir": "models/Qwen3-0.6B-GRPO-CUREBench",
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 8,
    "max_prompt_length": 512,
    "max_completion_length": 1024,
    "num_generations": 8,
    "training_mode": "full_parameter"
  },
  "grpo": {
    "learning_rate": "TRL默认值",
    "beta": "TRL默认值",
    "temperature": "TRL默认值"
  },
  "vllm_acceleration": {
    "enabled": true,
    "mode": "colocate",
    "gpu_memory_utilization": 0.35
  }
}
```

### 训练流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            完整训练流程                                      │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌─────────────────┐
     │ 1. 数据加载     │
     │ JSONL (459样本) │
     │ 排除2个Few-shot │
     │ → 457训练样本   │
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │ 2. 数据预处理   │
     │ • 格式化问题    │
     │ • 构建对话模板  │
     │ • Tokenization  │
     │ • 添加<think>引导│
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │ 3. GRPO训练     │
     │ • 每prompt生成8个│
     │   候选响应      │
     │ • 奖励函数打分  │
     │ • 组内相对排序  │
     │ • 策略梯度更新  │
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │ 4. 保存模型     │
     │ • 每50步保存    │
     │ • 保留最近3个   │
     │ • 训练配置JSON  │
     └─────────────────┘

训练统计:
• 数据集: 457样本 (排除2个few-shot示例)
• 有效批大小: 8 (batch_size=4 × gradient_accumulation=2)
• 每epoch步数: ~57步
• 总训练步数: ~171步 (3 epochs)
• 检查点保存: 第50, 100, 150步
• 最佳检查点: checkpoint-114
```

---

## 数据集

### 数据格式

数据集采用JSONL格式，每行一个JSON对象：

```json
{
  "id": "unique_question_id",
  "question_type": "multi_choice | open_ended | open_ended_multi_choice",
  "question": "问题文本...",
  "options": {
    "A": "选项A文本",
    "B": "选项B文本",
    "C": "选项C文本",
    "D": "选项D文本",
    "E": "选项E文本"
  },
  "correct_answer": "A"
}
```

### 问题类型支持

| 类型 | 说明 | 处理方式 |
|------|------|---------|
| `multi_choice` | 标准多选题 | 直接选择A-E |
| `open_ended_multi_choice` | 开放式转多选 | 生成开放答案后转换为选项 |
| `open_ended` | 开放式问答 | 自由文本回答 |

### 数据集规模

| 数据集 | 样本数 | 用途 |
|--------|--------|------|
| `curebench_valset_pharse1.jsonl` | 459 | 训练/验证 |
| `curebench_testset_phase1.jsonl` | 2,079 | 测试集1 |
| `curebench_testset_phase2.jsonl` | 2,491 | 测试集2 |

---

## 推理流程

### 推理框架架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           推理框架 (eval_framework.py)                       │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌──────────────────────┐
                        │    CompetitionKit    │
                        │    (主控制器)        │
                        └──────────┬───────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
     ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
     │   LocalModel    │  │   VLLMModel     │  │  ChatGPTModel   │
     │ (HuggingFace)   │  │ (批量加速)      │  │  (Azure API)    │
     └─────────────────┘  └─────────────────┘  └─────────────────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │    答案提取器        │
                        │ • XML标签解析       │
                        │ • 多选题选项匹配    │
                        │ • 开放式答案处理    │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │    结果生成器        │
                        │ • submission.csv    │
                        │ • meta_data.json    │
                        │ • 打包为.zip        │
                        └──────────────────────┘
```

### 答案提取逻辑

```python
def extract_answer(response: str) -> str:
    """
    从模型响应中提取答案
    优先级: <answer>标签 > 正则匹配 > 默认值
    """
    # 1. 尝试从<answer>标签提取
    xml_match = re.search(r"<answer>\s*([A-E])\s*</answer>", response)
    if xml_match:
        return xml_match.group(1).upper()

    # 2. 回退到其他匹配模式...
    return ""
```

### 推理命令

```bash
# 本地模型推理
python run.py --config metadata_config_val.json

# vLLM加速推理 (推荐用于大规模测试)
python run.py --config metadata_config_vllm.json
```

---

## 输出格式

### submission.csv

| 字段 | 说明 | 示例 |
|------|------|------|
| `id` | 问题唯一标识 | "ABC123XYZ" |
| `prediction` | 模型完整输出 | "<think>...</think><answer>A</answer>" |
| `choice` | 提取的选项 | "A" / "NOTAVALUE" (开放题) |
| `reasoning` | 推理过程 (JSON) | "Let me analyze..." |

### meta_data.json

```json
{
  "meta_data": {
    "model_name": "Qwen3-0.6B-GRPO-CUREBench",
    "track": "internal_reasoning",
    "model_type": "LocalModel",
    "base_model_type": "Open weight model",
    "base_model_name": "Qwen3-0.6B-GRPO-CUREBench",
    "dataset": "cure_bench_pharse_1"
  }
}
```

---

## 项目结构

```
CUREBench-submission/
│
├── 训练相关
│   ├── train_grpo.py              # GRPO训练主脚本 (501行)
│   ├── train_config.json          # 训练配置文件
│   └── merge_lora.py              # LoRA权重合并工具
│
├── 推理评估
│   ├── eval_framework.py          # 评估框架核心 (1,317行)
│   ├── run.py                     # 推理入口脚本
│   ├── dataset_utils.py           # 数据集加载工具
│   ├── metadata_config_val.json   # 验证集配置
│   ├── metadata_config_vllm.json  # vLLM推理配置
│   └── metadata_config_test.json  # 测试集配置
│
├── 数据集
│   ├── curebench_valset_pharse1.jsonl    # 验证集 (459样本)
│   ├── curebench_testset_phase1.jsonl    # 测试集1 (2,079样本)
│   └── curebench_testset_phase2.jsonl    # 测试集2 (2,491样本)
│
├── 模型
│   └── models/
│       ├── Qwen3-0.6B/                   # 基座模型
│       ├── Qwen3-0.6B-GRPO-CUREBench/    # 训练后模型
│       └── Qwen3-0.6B-GRPO-CUREBench-2epoch/
│
├── 结果输出
│   ├── results/                   # 最终结果
│   ├── results-val-ori/           # 原始验证结果
│   └── results-val-2epoch/        # 2epoch验证结果
│
└── 文档与配置
    ├── README.md                  # 项目文档
    └── requirements.txt           # 依赖列表
```

---

## 快速开始

### 环境安装

```bash
# 1. 克隆仓库
git clone <repository-url>
cd CUREBench-submission

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载基座模型 (如果需要)
# 将Qwen3-0.6B模型放置于 models/Qwen3-0.6B/
```

### 训练模型

```bash
# 使用默认配置训练
python train_grpo.py

# 自定义参数训练
python train_grpo.py \
    --model_path models/Qwen3-0.6B \
    --dataset_path curebench_valset_pharse1.jsonl \
    --output_dir models/Qwen3-0.6B-GRPO-CUREBench \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_generations 8

# 禁用vLLM加速 (如遇兼容问题)
python train_grpo.py --no_vllm
```

### 运行推理

```bash
# 验证集推理
python run.py --config metadata_config_val.json

# 测试集推理 (vLLM加速)
python run.py --config metadata_config_vllm.json
```

---

## 依赖项

```txt
# 核心依赖
transformers>=4.21.0      # HuggingFace模型库
torch>=1.12.0             # PyTorch深度学习框架
trl                       # TRL库 (GRPO训练)
tqdm>=4.64.0              # 进度条
pandas>=1.3.0             # 数据处理

# 可选依赖
vllm>=0.6.0               # vLLM推理加速
openai>=1.0.0             # Azure OpenAI API
numpy>=1.21.0             # 数值计算
scikit-learn>=1.0.0       # 评估指标
```

---

## 技术亮点总结

| 特性 | 实现方案 |
|------|----------|
| **轻量级部署** | Qwen3-0.6B，单卡24GB即可训练 |
| **GRPO训练** | 组相对策略优化，无需外部奖励模型 |
| **自定义奖励** | 格式+正确性+推理质量综合评估 |
| **推理加速** | vLLM colocate模式，训练推理共用GPU |
| **结构化输出** | 强制XML格式，提升可解释性 |
| **医学适配** | 专家系统提示词 + Few-shot示例 |
| **混合精度** | bfloat16 + 梯度检查点，显存优化 |

---
