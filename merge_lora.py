import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# ================= 配置区域 =================

# 1. 基座模型名称 (必须与训练时一致)
BASE_MODEL_NAME = "models/Qwen3-0.6B"

# 2. 训练好的 Adapter 路径 (即训练脚本中的 OUTPUT_DIR)
ADAPTER_PATH = "models/Qwen3-0.6B-GRPO-CUREBench"

# 3. 合并后的模型保存路径
SAVE_PATH = "models/Qwen3-0.6B-GRPO-CUREBench-Full"

# ================= 合并逻辑 =================

def merge_model():
    print(f"Loading base model: {BASE_MODEL_NAME}")
    
    # 加载基座模型
    # 注意：合并时建议使用 float16 或 bfloat16，不要使用 4bit/8bit 量化加载，否则无法合并
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=torch.float16,  # 如果训练时用了 bf16，这里建议改为 torch.bfloat16
        device_map="auto",          # 显存不够时可改为 "cpu"
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
    # 将 Adapter 加载到基座模型上
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("Merging weights...")
    # 核心步骤：合并权重并卸载 LoRA 层
    model = model.merge_and_unload()

    print(f"Saving merged model to: {SAVE_PATH}")
    # 保存合并后的模型
    model.save_pretrained(SAVE_PATH)

    # 处理 Tokenizer
    print("Saving tokenizer...")
    try:
        # 优先尝试从 Adapter 目录加载 Tokenizer (因为训练时可能保存了特殊的 token)
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    except:
        # 如果 Adapter 目录没有 Tokenizer，则从基座加载
        print("Tokenizer not found in adapter path, loading from base model...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    tokenizer.save_pretrained(SAVE_PATH)
    
    print("✅ Merge completed successfully!")
    print(f"Model saved to: {os.path.abspath(SAVE_PATH)}")

if __name__ == "__main__":
    merge_model()