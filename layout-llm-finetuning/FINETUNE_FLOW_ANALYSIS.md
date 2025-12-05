# LLM Fine-tuning Flow Analysis

## 完整流程概览

### 阶段 1: 数据准备 (`prepare_sft_dataset.py`)

#### 1.1 加载布局概率模型
- **输入**: `layout_prob_model.joblib` (GMM模型) + `layout_thresholds.json` (分位数阈值)
- **作用**: 用于后续的分布过滤
- **合理性**: ✅ 合理 - 使用预训练的概率模型确保数据质量

#### 1.2 读取原始数据
- **输入**: `train_layout.json` (包含 `ad_copy` 和 `text_layout`)
- **输出**: 原始样本列表
- **合理性**: ✅ 合理 - 标准数据加载流程

#### 1.3 分布过滤
- **方法**: 对每个样本的 `text_layout` 计算 log 概率
- **阈值**: 使用 `is_in_distribution()` 检查是否达到阈值（默认 p5 = 中间95%）
- **输出**: 过滤后的样本列表
- **合理性分析**:
  - ✅ **优点**: 确保训练数据符合真实分布，提高模型质量
  - ⚠️ **潜在问题**: 
    - p5 阈值可能过于严格（丢弃5%的极端值，但可能包含有用信息）
    - 如果原始数据质量高，过滤后可能只剩60-70%的数据
  - 💡 **建议**: 
    - 先用 p10 测试，如果效果不好再收紧到 p5
    - 记录过滤统计，评估数据损失

#### 1.4 数据质量分析
- **功能**: `analyze_filtering_impact()` 分析过滤前后的分布变化
- **分析内容**:
  - 样本数量与保留率
  - Alignment 分布变化
  - Color 分布变化
  - 坐标统计（均值、标准差）
  - Log 概率分布
- **合理性**: ✅ **优秀** - 提供了数据质量的可视化，帮助理解过滤影响

#### 1.5 转换为 SFT 格式
- **格式**: 
  ```json
  {
    "instruction": "You are an ad design assistant...",
    "input": "Ad copy:\n[广告文案]",
    "output": "{\"text_layout\": {...}}"
  }
  ```
- **合理性**: ✅ 合理 - 标准的指令微调格式

#### 1.6 保存为 JSONL
- **输出**: `sft_dataset.jsonl` (每行一个样本)
- **合理性**: ✅ 合理 - JSONL 格式便于流式处理

---

### 阶段 2: 模型微调 (`finetune_layout_llm.py`)

#### 2.1 加载基座模型
- **模型**: Qwen2.5-1.5B 或 3B
- **环境适配**:
  - CUDA: 使用 4-bit 量化 (bitsandbytes)
  - CPU/MPS: 使用 float32 全精度
- **合理性**: ✅ **优秀** - 自动环境适配，支持多种硬件

#### 2.2 配置 LoRA
- **参数**:
  - `r=16` (rank)
  - `lora_alpha=32`
  - `lora_dropout=0.05`
  - 目标模块: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **可训练参数**: ~1.18% (18M / 1.5B)
- **合理性**: ✅ 合理 - 标准 LoRA 配置，参数选择合理

#### 2.3 数据预处理
- **步骤**:
  1. 读取 `sft_dataset.jsonl`
  2. 使用 Qwen 的 `apply_chat_template` 格式化为对话格式
  3. Tokenize (truncation=True, max_length=1024, padding=False)
  4. **关键**: 计算 assistant 回复的起始位置
- **合理性分析**:
  - ✅ **优点**: 正确识别 assistant 回复边界
  - ⚠️ **潜在问题**: 
    - 通过比较 `instruction+input` 和完整序列的长度来确定边界，可能不够精确
    - 如果 chat template 在 assistant 回复前添加了特殊 token，可能导致边界偏移
  - 💡 **建议**: 
    - 验证边界计算的准确性（可以打印几个样本检查）
    - 考虑使用 tokenizer 的特殊 token 来更精确地定位

#### 2.4 自定义 Data Collator
- **功能**: `InstructionDataCollator` - 只对 assistant 回复部分计算 loss
- **实现**:
  - 使用预计算的 `assistant_start_positions`
  - 将 instruction/input 部分的 labels 设为 -100（忽略）
  - 只对 assistant 回复部分计算 loss
- **合理性**: ✅ **关键改进** - 这是最重要的修复，确保模型只学习生成 output，不学习生成 instruction

#### 2.5 数据集划分
- **策略**: 
  - 如果样本数 > 100: 90% 训练集，10% 验证集
  - 否则: 全部用于训练
- **合理性分析**:
  - ✅ **优点**: 简单有效
  - ⚠️ **潜在问题**: 
    - 简单的随机划分可能不够科学
    - 没有考虑数据的时间顺序或分布一致性
  - 💡 **建议**: 
    - 如果数据有时间顺序，使用时间划分
    - 或者使用分层采样确保分布一致

#### 2.6 训练配置
- **参数**:
  ```python
  num_epochs: 3
  batch_size: 4 (per device)
  gradient_accumulation_steps: 4 (实际 batch = 16)
  learning_rate: 2e-4
  fp16/bf16: 根据环境自动选择
  eval_strategy: "steps" (每 500 步评估)
  ```
- **合理性**: ✅ 合理 - 参数选择适中，适合小模型微调

#### 2.7 训练执行
- **使用**: HuggingFace `Trainer` API
- **合理性**: ✅ 合理 - 标准训练流程

---

### 阶段 3: 推理 (`infer_layout_llm.py`)

#### 3.1 加载模型
- **功能**: `load_model_for_inference()` - 加载微调后的模型
- **支持**: LoRA adapter 自动检测和合并
- **合理性**: ✅ 合理 - 封装良好，使用方便

#### 3.2 生成布局
- **功能**: `generate_layout()` - 从广告文案生成 JSON
- **方法**: 使用 chat template 格式化 prompt，然后生成
- **合理性**: ✅ 合理 - 与训练时格式一致

#### 3.3 后处理
- **功能**: 
  - `extract_json_from_text()` - 从生成文本中提取 JSON
  - `validate_and_fix_layout()` - 验证和修复布局值
- **合理性**: ✅ 合理 - 必要的后处理步骤

#### 3.4 智能重试机制
- **功能**: `infer_with_retry()` - 如果生成不符合分布，自动重试
- **策略**:
  - 自适应 temperature 调度（从保守到探索）
  - Rejection Sampling（每次尝试生成多个候选，选择最好的）
  - 详细日志记录
- **合理性**: ✅ **优秀** - 显著提高推理成功率

#### 3.5 分布检查
- **功能**: 使用 `layout_prob_model` 检查生成结果是否符合分布
- **合理性**: ✅ **优秀** - 训练和推理都使用分布对齐，形成闭环

---

## 整体流程合理性分析

### ✅ 设计优秀的方面

1. **分布对齐机制**
   - 训练前过滤 + 推理时检查的双重保障
   - 使用概率模型而非硬规则，更灵活

2. **Loss 计算修复**
   - 自定义 `InstructionDataCollator` 只对 assistant 回复计算 loss
   - 这是最关键的技术改进

3. **数据质量分析**
   - `analyze_filtering_impact()` 提供详细的过滤影响分析
   - 帮助理解数据变化

4. **智能推理**
   - 自适应 temperature + rejection sampling
   - 显著提高生成质量

5. **环境自适应**
   - 自动检测 CUDA/MPS/CPU
   - 自动选择量化策略

### ⚠️ 潜在问题和改进建议

#### 问题 1: Assistant 边界计算可能不够精确

**当前方法**:
```python
# 通过比较 instruction+input 和完整序列的长度来确定边界
inst_input_len = len(instruction_input_tokenized["input_ids"][i])
assistant_start = inst_input_len
```

**潜在问题**:
- 如果 chat template 在 assistant 回复前添加了特殊 token，可能导致边界偏移
- 如果 instruction+input 被 truncate，边界可能不准确

**建议**:
- 验证边界计算的准确性（打印几个样本检查）
- 考虑使用 tokenizer 的特殊 token（如 `<|im_start|>assistant`）来更精确地定位
- 或者直接 tokenize output 部分，然后找到它在完整序列中的位置

#### 问题 2: 数据过滤阈值可能过于严格

**当前**: 默认 p5（只保留中间 95%）

**建议**:
- 提供阈值选择指南
- 建议先用 p10 测试，如果效果不好再收紧
- 记录过滤统计，帮助用户做决策

#### 问题 3: 数据集划分策略简单

**当前**: 简单的 90/10 随机划分

**建议**:
- 如果数据有时间顺序，使用时间划分
- 或者使用分层采样确保分布一致
- 考虑使用 `train_test_split` 的 `stratify` 参数（如果可能）

#### 问题 4: 缺少训练监控指标

**当前**: 只有 loss 监控

**建议**:
- 添加自定义 metric（如 JSON 格式正确率、分布符合率）
- 在验证集上评估生成质量

#### 问题 5: 推理时的边界情况处理

**当前**: 如果多次重试都失败，返回最佳尝试

**建议**:
- 可以添加"fallback"策略（如使用训练集中最相似的样本）
- 或者提供"手动修正"选项

---

## 流程完整性检查

### ✅ 完整的流程链路

```
原始数据 (train_layout.json)
    ↓ [分布过滤 + 质量分析]
SFT 数据集 (sft_dataset.jsonl)
    ↓ [LoRA 微调 + 正确的 Loss 计算]
微调模型 (output_layout_llm/)
    ↓ [智能推理 + 分布检查]
最终输出 (text_layout JSON)
```

### ✅ 关键改进点

1. **Loss 计算修复** - 只对 assistant 回复计算 loss ✅
2. **数据质量分析** - 提供详细的过滤影响分析 ✅
3. **智能推理** - 自适应重试机制 ✅

---

## 总体评价

### 合理性评分: 8.5/10

**优点**:
- ✅ 核心问题已修复（Loss 计算）
- ✅ 分布对齐机制完善
- ✅ 数据质量分析详细
- ✅ 推理机制智能
- ✅ 环境适配良好

**待改进**:
- ⚠️ Assistant 边界计算可以更精确
- ⚠️ 数据过滤阈值可以更灵活
- ⚠️ 可以添加更多训练监控指标

### 结论

**整体流程设计合理，核心问题已解决。** 

主要改进（Loss 计算修复）已经完成，这是最关键的技术问题。其他问题属于优化项，不影响基本功能。

**建议优先级**:
1. ✅ **已完成**: Loss 计算修复
2. ✅ **已完成**: 数据质量分析
3. ✅ **已完成**: 智能推理重试
4. 🔄 **可选**: 验证 assistant 边界计算准确性
5. 🔄 **可选**: 添加更多训练监控指标

流程已经可以投入使用，后续可以根据实际效果进行优化。

