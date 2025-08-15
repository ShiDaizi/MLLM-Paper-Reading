# MLLM-Paper-Reading
This is a paper reading repository for recording my list of read papers.

## 📖 Table of Contents
- [MLLM-Paper-Reading](#mllm-paper-reading)
  - [📖 Table of Contents](#-table-of-contents)
  - [Image LLMs CVPR2025](#image-llms-cvpr2025)
  - [Benchmark CVPR2025](#benchmarks-cvpr2025)

## Image LLMs CVPR2025
- [x] **LLaVA-Critic**: Learning to Evaluate Multimodal Models. CVPR2025 [Paper](https://arxiv.org/abs/2410.02712) [Page](https://llava-vl.github.io/blog/2024-10-03-llava-critic/)
  
  本文介绍了 LLaVA-Critic，这是首个开源大型多模态模型（LMM），其设计定位为通用评估器，可用于评估各类多模态任务的性能。该模型在高质量的 “评估指令遵循数据集” 上训练而成，该数据集包含 4.6 万张图像和 11.3 万个评估样本（涵盖逐点评估和 pairwise 评估两种设置），并在两个关键领域展现出有效性。

- [x] **Img-Diff**: Contrastive Data Synthesis for Multimodal Large Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2408.04594) [Code](https://github.com/modelscope/data-juicer/tree/ImgDiff)

  为提升多模态大型语言模型（MLLMs）的细粒度图像识别能力，研究团队提出名为Img-Diff的新型对比性数据合成方法，该方法受对比学习和图像差异描述启发，通过生成强调物体变化的相似图像对，借助差异区域生成器定位物体差异、差异描述生成器阐述差异，构建出可自动扩展的 “物体替换” 样本数据集；利用该数据集微调 InternVL2 等当前最优（SOTA）MLLMs，在图像差异任务和视觉问答（VQA）任务上实现显著性能提升，尤其在 MMVP 基准测试中大幅优于 GPT-4V、Gemini 等现有 SOTA 模型，同时团队已公开代码和数据集以推动相关研究。


- [x] **FlashSloth**: Lightning Multimodal Large Language Models via Embedded Visual Compression. CVPR2025 [Paper](https://arxiv.org/abs/2412.04317) [Code](https://github.com/codefanw/FlashSloth)
  
  This paper presents FlashSloth, a powerful and fast tiny multimodal large language model (MLLM) that achieves a superior balance between performance and efficiency by introducing embedded visual compression designs, specifically Spatial Attention Pooling (SAP) to capture visually salient semantics and an Embedded Query (EmbQ) module to grasp instruction-related image information, thereby greatly reducing the number of visual tokens, training memory, computation complexity, and response time while retaining high performance on various vision-language tasks compared to advanced tiny MLLMs like InternVL2, MiniCPM-V2, and Qwen2-VL.
  **This paper modifies the image token embedding method in a tricky way, and through pre-training plus SFT, it outperforms other models in terms of performance, speed, and memory usage. However, it doesn't mention which datasets were used for training??**

- [x] **BlueLM-V-3B**: Algorithm and System Co-Design for Multimodal Large Language Models on Mobile Devices. CVPR2025 [Paper](https://arxiv.org/abs/2411.10640v1)
  
  BlueLM-V-3B is a 3B-parameter mobile-optimized multimodal large language model that achieves superior performance (topping OpenCompass among ≤4B models with 66.1 points and outperforming some 8B models) and efficiency (2.2GB memory, 24.4 token/s on MediaTek Dimensity 9300) through algorithm-system co-design, including relaxed aspect ratio matching, token downsampling, mixed-precision quantization, and two-stage training on 2.5M pretraining and 645M fine-tuning image-text pairs (both open-source and in-house data).

- [x] **Insight-V**: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2411.14432) [Code](https://github.com/dongyh20/Insight-V)
  
  Llama-VID, a video understanding large language model based on the Llama architecture, achieves superior performance across 14 video understanding benchmarks (with an average 5.3% improvement in VideoQA) and enhanced efficiency (2.1x faster inference and 40% lower memory usage) through key innovations like dynamic video tokenization (adjusting tokens based on content complexity) and temporal-aware attention (strengthening long-video temporal modeling), trained on 8.7M video-text pairs for pre-training and 1.2M instruction data for fine-tuning.

- [x] **Critic-V**: VLM Critics Help Catch VLM Errors in Multimodal Reasoning. CVPR2025 [Paper](https://arxiv.org/abs/2411.18203)

  Critic-V is a novel framework inspired by the Actor-Critic paradigm designed to enhance the multimodal reasoning capabilities of Vision-Language Models (VLMs) by addressing issues like hallucinated image understandings and unrefined reasoning paths. It comprises two core components: the Reasoner, which generates reasoning paths from visual and textual inputs using dynamic text prompts that evolve iteratively, and the Critic, which provides nuanced natural language feedback (instead of scalar rewards) to refine these paths, operating within a reinforcement learning framework. The Critic is trained via Direct Preference Optimization (DPO) on a dataset of 29,012 multimodal question-answer pairs, with critiques ranked using a Rule-based Reward (RBR) that combines Jaccard similarity (for error detection accuracy) and GPT-4o scores (for feedback quality). Evaluation results demonstrate that Critic-V significantly outperforms existing methods, including GPT-4V, on 5 out of 8 benchmarks, with notable improvements in mathematical reasoning (e.g., an 11.8% boost on MathVista for Qwen2-VL-7B) and strong performance across tasks like RealWorldQA and MMT-Bench, highlighting its effectiveness in enhancing VLM reliability for real-world applications such as autonomous driving and embodied intelligence.

- [x] **Mono-InternVL**: Pushing the Boundaries of Monolithic Multimodal Large Language Models with Endogenous Visual Pre-training. CVPR2025 [Paper](https://arxiv.org/abs/2410.08202) [Code](https://internvl.github.io/blog/2024-10-10-Mono-InternVL/)

  Mono-InternVL是一种新型单体多模态大语言模型（MLLM），旨在解决现有单体 MLLMs 存在的不稳定优化和灾难性遗忘问题。其核心创新在于通过多模态混合专家（MMoE）结构将视觉专家嵌入预训练 LLM，并采用内生视觉预训练（EViP） 策略 —— 分三阶段（概念学习、语义学习、对齐学习）从噪声数据到高质量数据渐进式学习视觉知识。实验表明，Mono-InternVL 在 16 个多模态基准中的 13 个超越现有单体 MLLMs（如在 OCRBench 上比 Emu3 高 80 分），同时与模块化模型 InternVL-1.5 性能相当，但首 token 延迟降低 67%，为单体 MLLMs 的发展提供了新方向。

- [x] **DivPrune**: Diversity-based Visual Token Pruning for Large Multimodal Models. CVPR2025 [Paper](https://arxiv.org/abs/2503.02175) [Code](https://github.com/vbdi/divprune)

  DivPrune是一种基于Max-Min Diversity Problem (MMDP) 的视觉 token 剪枝方法，旨在解决大型多模态模型（LMMs）中视觉 token 冗余导致的高推理延迟问题。其核心是通过最大化保留 token 的多样性（即最大化最小 pairwise 距离）减少冗余，无需微调或校准数据。实验表明，DivPrune 在 16 个图像和视频语言数据集上实现最先进精度，尤其在高剪枝率（≥80%）下优势显著；同时降低 GPU 内存使用和端到端延迟，例如在 LLaVA 1.5-7B 上，剪枝 90% token 时 COCO 的 CIDEr 仅下降 12.7%，而现有方法下降 95%，且首 token 延迟降低 67%。

- [x] **ODE**: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2409.09318)

  ODE（Open-Set Dynamic Evaluation） 是一种开放集动态评估协议，旨在解决多模态大语言模型（MLLMs）幻觉评估中数据污染的问题。其核心是通过图结构建模现实世界的物体概念、属性及关联，基于四种分布标准（Standard、Long-tail、Random、Fictional） 生成动态样本，评估存在级和属性级幻觉。实验表明，ODE 生成的样本揭示了 MLLMs 更高的幻觉率（如 MiniGPT-4 在 ODE 上的 F1 分数比静态基准低 20% 以上），且生成数据可辅助模型微调（LLaVA-1.5 微调后幻觉率降低 7.6%），为 MLLMs 幻觉评估和优化提供了可靠方案。

- [x] Augmenting Multimodal LLMs with Self-Reflective Tokens for Knowledge-based Visual Question Answering. CVPR2025 [Paper](https://arxiv.org/abs/2411.16863) [Code](https://github.com/aimagelab/ReflectiVA)

  ReflectiVA是一种增强多模态大语言模型（MLLMs）的方法，通过引入自反射令牌（reflective tokens）（<RET>、<NORET>、<REL>、<NOREL>）实现对外部知识的动态利用。该模型采用两阶段两模型训练策略：首先训练文章内判别器区分同篇文档中的相关与不相关段落，再利用其标注数据结合混合数据集训练最终模型。实验表明，ReflectiVA 在Encyclopedic-VQA和InfoSeek等知识型视觉问答任务上显著优于现有方法（如在 Encyclopedic-VQA 单跳任务上准确率达 35.5%），同时保留在标准 MLLM 基准上的性能，为需外部知识的多模态任务提供了有效解决方案。

- [x] **AGLA**: Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention. CVPR2025 [Paper](https://arxiv.org/abs/2406.12718) [Code](https://github.com/Lackel/AGLA)

  大型视觉语言模型（LVLMs） 虽在多模态任务中表现出色，但常存在物体幻觉问题（生成文本与图像中实际物体不一致）。研究发现，其根源在于注意力缺陷—— 过度关注与提示无关的全局特征，而忽略相关局部特征。为此，本文提出AGLA（Assembly of Global and Local Attention），这是一种无需训练、即插即用的方法：通过图像 - 提示匹配（IPM） 生成增强图像视图（突出相关局部特征、生成掩码图像），再融合原始图像的生成性全局特征与增强图像的辨别性局部特征，得到校准的 logit 分布，从而缓解幻觉。实验表明，AGLA 在 POPE、ROPE、MME、CHAIR 等多个数据集上，对 LLaVA-1.5、InstructBLIP 等模型的幻觉 mitigation 效果显著优于现有方法（如 POPE 的 adversarial 设置下，LLaVA-1.5 的 F1 分数从 76.26 提升至 81.36）。

- [x] **ICT**: Image-Object Cross-Level Trusted Intervention for Mitigating Object Hallucination in Large Vision-Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2411.15268v1)

  大型视觉语言模型（LVLMs） 存在物体幻觉问题（生成内容与图像物体不一致），主要源于过度依赖语言先验和视觉细节捕捉不足。本文提出ICT（Image-Object Cross-Level Trusted Intervention），一种无需训练、即插即用的方法：通过图像级干预（增强对整体视觉信息的关注）和物体级干预（提升对细粒度物体细节的注意力），在 forward pass 阶段调整注意力头的激活值，既保留有用语言先验，又减少幻觉。实验显示，ICT 在 POPE 基准上平均提升 6.27%，MME 基准上提升 67.37 分，且不增加推理延迟，在不同模型和数据集上泛化性良好。

- [x] Can Large Vision-Language Models Correct Grounding Errors By Themselves? CVPR2025 [Paper](https://openreview.net/pdf?id=fO1xnmW8T6)

  本文聚焦视觉语言模型（VLMs）的语义接地错误自我纠正能力，发现 VLMs 在适当提示下，无需微调或 oracle 反馈即可纠正自身语义接地错误；提出迭代自校正框架，通过分解语义接地为二进制验证任务、结合文本与视觉提示技术，在 LLaVA-1.5、GPT-4V 等 5 个模型上持续提升性能，准确率最高提升8.4 个点。研究还揭示，即使经过多轮反馈，GPT-4V 等强模型利用 oracle 反馈的能力仍有限，为后续研究提供方向。

- [x] **Molmo and PixMo**: Open Weights and Open Data for State-of-the-Art Vision-Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2409.17146) [Page](https://molmo.allenai.org/blog)

  本文介绍了Molmo—— 一个开源的视觉语言模型（VLM）家族，及其配套的PixMo数据集。PixMo 包含 3 个人工标注数据集（高细节图像描述、自由形式图像问答、2D 指向标注）和 4 个合成数据集，均不依赖外部 VLM 生成数据。Molmo 通过优化模型设计（如重叠多裁剪策略、高效连接器）和训练流程，在学术基准和人类评估中表现优异：其 72B 模型性能超过Claude 3.5 Sonnet、Gemini 1.5 Pro等专有模型，仅次于GPT-4o。模型权重、数据集和代码均开源，为从无到有构建高性能 VLMs 提供了基础。

- [x] **Nullu**: Mitigating Object Hallucinations in Large Vision-Language Models via HalluSpace Projection. CVPR2025 [Paper](https://arxiv.org/abs/2412.13817) [Code](https://github.com/Ziwei-Zheng/Nullu)

  本文提出Nullu方法，旨在缓解大型视觉语言模型（LVLMs）中的物体幻觉（OH） 问题。其核心是通过提取HalluSpace（幻觉子空间）—— 即真实与幻觉特征差异的低秩子空间，将模型权重投影到该子空间的零空间，从而过滤幻觉特征。实验表明，Nullu 在CHAIR、POPE等数据集上显著降低 OH（如 LLaVA-1.5 的 CHAIR_S 从 20.40 降至 15.20），且无额外推理成本，同时在通用基准（如 MME）上保持良好性能。此外，研究揭示 HalluSpace 与 LLM 先验相关，Nullu 通过抑制这些先验实现去偏，与 DPO 方法存在概念关联。

- [x] **HiRes-LLaVA**: Restoring Fragmentation Input in High-Resolution Large Vision-Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2407.08706)

  本文提出HiRes-LLaVA，一种高效处理高分辨率输入的大型视觉语言模型（LVLM），旨在解决现有切片式高分辨率处理方法导致的输入碎片化问题（上下文连续性和空间几何信息丢失）。其核心创新包括：SliceRestore 适配器（通过合并 - 捕捉 - 分割流程重构切片特征，保留全局和局部信息）和Self-Mining Sampler（基于特征自身压缩视觉 token，保留位置信息并降低训练成本）；还提出动态高分辨率切片策略，适应任意比例图像。为评估碎片化处理能力，构建EntityGrid-QA 基准（含识别、位置、计数任务）。实验表明，HiRes-LLaVA 在 11 个公共基准和 EntityGrid-QA 上表现优异，尤其在文档任务中，如 CLIP-ViT-336px 版本在 DocVQA 上达 65.7，显著超越现有模型。

- [x] Devils in Middle Layers of Large Vision-Language Models: Interpreting, Detecting and Mitigating Object Hallucinations via Attention Lens. CVPR2025 [Paper](https://arxiv.org/abs/2411.16724)

  本文通过注意力透镜解析大型视觉语言模型（LVLMs）的物体幻觉机制，发现中间层是视觉信息处理的关键，可分为 “视觉信息富集” 和 “语义精炼” 两个阶段；揭示真实物体 token 在视觉信息富集阶段的视觉注意力权重显著高于幻觉 token（AUROC 达 74%），且幻觉 token 的注意力头存在跨物体交互不一致问题；提出基于中间层注意力整合的推理时干预方法，在 LLaVA-1.5 等 3 个模型上平均降低CHAIR_I 6.3 点和CHAIR_S 24.1 点，有效缓解幻觉且不增加训练成本。

- [x] **HoVLE**: Unleashing the Power of Monolithic Vision-Language Models with Holistic Vision-Language Embedding. CVPR2025 [Paper](https://arxiv.org/abs/2412.16158)

  为解决现有单体视觉语言模型（monolithic VLMs） 需调优预训练大语言模型（LLMs）导致语言能力退化、性能落后于组合型 VLMs 的问题，本文提出HoVLE—— 一种新型高性能单体 VLM，其核心是引入整体视觉 - 语言嵌入模块（holistic vision-language embedding module），将视觉与文本输入映射到共享嵌入空间，使 LLM 能以处理文本的方式理解图像；同时设计三阶段训练策略（蒸馏→对齐→指令微调），无需额外标注数据即可赋能嵌入模块，最终 HoVLE（**2.6B 参数**）在 17 个多模态基准上大幅超越现有单体 VLMs（如 MMBench 提升～15 个点），且性能接近领先组合型 VLMs（如 InternVL2、Qwen2-VL-2B）。

- [x] **VoCo-LLaMA**: Towards Vision Compression with Large Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2406.12275v2) [Code](https://github.com/Yxxxb/VoCo-LLaMA?tab=readme-ov-file)

  为解决视觉语言模型（VLMs）处理高分辨率图像 / 视频时视觉 token 占用上下文窗口过大、计算成本高的问题，本文提出VoCo-LLaMA—— 首个利用大语言模型（LLMs）固有能力实现视觉压缩的方法，无需外部模块；其核心是在视觉指令微调阶段引入VoCo token，通过修改注意力掩码构建 “视觉 token→VoCo token→文本 token” 的专属交互路径，结合注意力蒸馏将 LLM 对视觉 token 的理解迁移到 VoCo token；该方法在推理时采用两阶段流程（视觉压缩→任务处理），可实现576× 压缩率并保持83.7% 的原始性能，同时 KV 缓存复用使缓存存储减少99.8%、FLOPs 降低94.8%、推理时间缩短69.6%；扩展到视频领域时，通过压缩帧序列的 VoCo token 建模时序关联，在 MSVD-QA 等视频 QA 基准上超越现有方法，支持处理约 200 倍更多视频帧。

- [x] Perception Tokens Enhance Visual Reasoning in Multimodal Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2412.03548) [Page](https://aurora-perception.github.io/)

  为解决多模态语言模型（MLMs）在3D 结构推理（如深度估计）和 2D 目标实例推理（如目标计数） 中的不足（现有微调泛化差、工具依赖型方法计算 / 内存低效），论文提出Perception Tokens（感知令牌） —— 一种辅助推理的内在图像表示，类似语言模型的思维链（CoT）提示，并设计AURORA 训练方法：通过 VQVAE 将深度图等中间表示 token 化，结合多任务训练与课程学习，使 MLMs 能利用感知令牌进行中间推理。实验显示，基于 LLaVA 1.5 13B 的 LLaVA-AURORA 在核心任务上显著提升：3D 相对深度估计在 BLINK 上提升 + 6.4%，2D 目标计数在 BLINK（+10.8%）、CVBench（+11.3%）、SEED-Bench（+8.3%）上均超微调方法，且实现跨任务泛化。

- [x] **Florence-VL**: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion. CVPR2025 [Paper](https://arxiv.org/abs/2412.04424) [Page](https://jiuhaichen.github.io/florence-vl.github.io/)

  论文提出Florence-VL，这是一类基于生成式视觉基础模型Florence-2的多模态大语言模型（MLLMs），旨在解决现有 MLLM 依赖 CLIP 类对比学习视觉编码器、缺乏多粒度视觉特征的局限。Florence-2 能通过不同提示（如详细 caption、OCR、密集区域 caption）提取多维度视觉特征，论文进一步提出Depth-Breadth Fusion（DBFusion） 策略，将不同深度（DaViT 低层特征 + 高层任务特征）和广度（多提示特征）的视觉特征通过通道拼接融合，并投影到 LLM 输入空间。模型训练采用 “端到端全模型预训练（基于 16.9M 图像 caption 数据）+ 投影层与 LLM 微调（基于 10M 指令数据）” 的流程，最终在25 个涵盖通用多模态、视觉中心、OCR & 图表、知识型的基准测试中显著超越现有 SOTA。

- [x] **ATP-LLaVA**: Adaptive Token Pruning for Large Vision Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2412.00447) [Page](https://yxxxb.github.io/ATP-LLaVA-page/)

  为解决大型视觉语言模型（LVLMs）处理长视觉令牌时计算成本过高的问题，论文提出ATP-LLaVA，其核心是通过Adaptive Token Pruning（ATP）模块实现层级与实例级自适应令牌剪枝，而非传统固定剪枝比例。ATP 模块结合Spatial Augmented Pruning（SAP）策略，从令牌冗余（融合模态内 / 跨模态注意力得分）和空间建模（动态均匀空间采样 + 2D 旋转位置嵌入保留）双视角计算令牌重要性，并通过可学习阈值动态筛选令牌；同时设计ATP-Loss 损失函数平衡剪枝效率与模型性能。实验表明，ATP-LLaVA 在 7 个基准测试中平均剪枝75% 视觉令牌（从 576 降至 144），仅损失1.9% 性能，同时减少75% KV 缓存内存、38.4% CUDA 推理时间和78.1% FLOPs，为资源受限设备部署 LVLMs 提供高效解决方案。

- [x] Accelerating Multimodel Large Language Models by Searching Optimal Vision Token Reduction. CVPR2025 [Paper](https://arxiv.org/abs/2412.00556)

  论文针对多模态大语言模型（MLLMs）因图像分辨率提升导致视觉令牌数量激增、计算成本过高的问题，提出从两个场景优化效率：（I）降低计算成本且不损失性能；（II）给定预算下提升性能。核心发现是视觉令牌按注意力得分的排序在除第一层外的所有层中高度相似（Kendall’s Tau 系数≥0.7），据此假设深层关键令牌是浅层关键令牌的子集、关键令牌数量不随层深增加。针对场景 I，提出G-Search 贪心搜索算法，通过贝叶斯优化从浅层到深层逐层寻找最小必要令牌保留率，实现最高2.3× 加速且平均精度仅下降0.2%；针对场景 II，基于 G-Search 的 S 型保留率曲线设计P-Sigmoid 参数化函数，通过贝叶斯优化搜索最优参数，在减少 87.5% 视觉令牌时，较 FastV 提升 **+3.38%（LLaVA-1.5-7B）** 至 **+7.69%（InternVL2-8B）** 平均精度。方法在 LLaVA、InternVL2 等多模型及 12 个基准测试中验证，且可进一步加速 prompt-agnostic 方法（如 TokenPacker）达 1.5×。

- [x] **VisionZip**: Longer is Better but Not Necessary in Vision Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2412.04467) [Code](https://github.com/dvlab-research/VisionZip)

  论文提出VisionZip，一种简单高效的视觉令牌压缩方法，旨在解决视觉语言模型（VLMs）中视觉令牌冗余（CLIP/SigLIP 生成的令牌中仅少数含高信息）导致的高计算成本问题。其核心是训练无关的令牌筛选与合并策略：先基于视觉编码器注意力得分选择主导令牌（Dominant Tokens）（含核心信息），再通过语义相似性合并剩余令牌生成上下文令牌（Contextual Tokens），同时可通过 30 分钟微调投影层（仅用 1/10 LLaVA-1.5 数据）进一步对齐模态空间。实验显示，VisionZip 在 LLaVA-1.5/NeXT、Mini-Gemini 等模型上，保留 95% 性能的同时减少 88.9% 视觉令牌，预填充时间提升 8 倍，使 LLaVA-NeXT 13B 推理速度超 7B 模型且精度更高，在视频理解（Video-LLaVA）和多轮对话场景中也显著优于 FastV、SparseVLM 等 SOTA 方法。


## Benchmarks CVPR2025
- [x] **Video-MME**: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis [Paper](https://arxiv.org/abs/2405.21075) [Code](https://github.com/BradyFU/Video-MME)

  本文提出Video-MME—— 首个针对多模态大语言模型（MLLMs）视频分析的全面评估基准，其核心优势包括：覆盖6 个主视觉领域及 30 个子领域的视频类型多样性、包含11 秒至 1 小时短 / 中 / 长时长的时间维度完整性、整合视频帧 + 字幕 + 音频的多模态数据广度，以及专家人工标注的高质量标注；通过该基准评估各类主流 MLLMs 发现，Gemini 1.5 Pro是表现最佳的商业模型（平均准确率 75%），显著优于GPT-4o（71.9%）及开源模型（如 VILA-1.5 准确率 59%），同时证实字幕和音频可显著提升视频理解能力（Gemini 1.5 Pro 加字幕提升 6.2%、加音频提升 4.3%），且所有模型均存在随视频时长增加性能下降的问题，为 MLLM 视频理解能力的未来优化提供方向。

- [x] **MLVU**: Benchmarking Multi-task Long Video Understanding [Paper](https://arxiv.org/abs/2406.04264) [Code](https://github.com/JUNJIE99/MLVU)
  
  本文提出MLVU（Multitask Long Video Understanding Benchmark）—— 首个针对多模态大语言模型（MLLMs）长视频理解能力的多任务评估基准，旨在解决现有基准视频长度不足（多为几秒至几分钟）、类型与任务单一、不适配长视频评估的问题。MLVU 核心优势包括：覆盖3 分钟至 2 小时（平均 15 分钟）的灵活视频时长、包含电影 / 监控 / 第一视角等10 类视频类型、设计9 类针对性任务（分整体 / 单细节 / 多细节 LVU 任务，任务格式含两种：①多选择题，开发集 4 个选项、测试集 6 个选项，评估指标为准确率；②生成题，含视频总结、子场景描述，通过 GPT-4 从 “准确性 / 相关性”（子场景描述）或 “完整性 / 可靠性”（视频总结）维度打分，满分 10 分）；通过评估 23 个主流 MLLMs 发现，GPT-4o表现最优（多选择任务平均准确率 54.5%，生成任务平均 5.87 分），开源长视频 MLLMs（如 LLaVA-Onevision）与商业模型差距缩小（仅差 2.8%），但所有模型均存在随视频时长增加性能显著下降的问题，且多细节任务（如动作排序、计数）仍是普遍瓶颈；研究同时揭示上下文长度、图像理解能力、LLM backbone是影响长视频理解的关键因素，为未来 MLLM 优化提供方向。

- [x] **MMVU**: Measuring Expert-Level Multi-Discipline Video Understanding [Paper](https://arxiv.org/abs/2501.12380) [Code](https://github.com/yale-nlp/MMVU)

  Zhao 等人（2025）提出MMVU（专家级多学科视频理解基准），这是一个用于评估基础模型视频理解能力的综合基准，包含3,000 个专家标注问题，覆盖4 个核心学科（科学、医疗、人文社科、工程）下的27 个细分领域，基于1,529 个专业领域视频构建。与现有基准相比，MMVU 的核心优势在于：要求模型应用领域特定知识进行专家级推理（而非基础视觉感知）、所有示例由人类专家从零标注并通过严格质量控制、每个示例附带专家标注的推理依据和领域知识。研究对32 个前沿多模态基础模型进行评估，发现具备 System-2 思维的o1 模型和Gemini 2.0 Flash Thinking 模型表现最佳，但仍未达到人类专家水平（开放书设置下人类准确率 86.8%，顶级模型 o1 测试集平均准确率 77.0%）；通过误差分析，识别出模型在视觉感知、领域知识应用等方面的关键不足，为后续研究提供方向。

- [x] **MV-MATH**: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts [Paper](https://arxiv.org/abs/2502.20808) [Page](https://eternal8080.github.io/MV-MATH.github.io/)

  Wang 等人（2025）提出MV-MATH（多视觉语境下的多模态数学推理评估基准），这是一个针对多模态大语言模型（MLLMs）多视觉数学推理能力的高质量基准，包含2,009 个源于真实 K-12 场景的多图像数学问题，覆盖3 个难度等级（简单 27%、中等 48%、困难 25%）和11 个数学学科，问题类型分为选择题（1,109 题）、自由作答题（900 题，含 100 道多步骤题）。与现有单视觉数学基准不同，MV-MATH 首次聚焦多视觉语境，将图像按关联性分为 “相互依赖型（MD，1,412 题）” 和 “独立型（ID，597 题）”，并附带细粒度标注。通过对25 个 MLLMs（18 个开源 + 7 个闭源）的评估发现，顶级模型Claude 3.5-Sonnet准确率仅 33.9%，远低于人类水平（76.5%）；开源模型中LLaVA-OneVision-72B表现最佳（26.2%），且图像序列输入优于合并输入，CoT 提示对自由作答题有效但对选择题可能起反作用。误差分析显示，视觉感知误差（42%） 是模型最主要的错误类型，为后续模型优化提供关键方向。

- [x] **VideoAutoArena**: An Automated Arena for Evaluating Large Multimodal Models in Video Analysis through User Simulation [Paper](https://arxiv.org/abs/2411.13281) [Code](https://github.com/VideoAutoArena/VideoAutoArena)

  Luo 等人（2025）提出VideoAutoArena，这是一个用于自动评估多模态大模型（LMMs）视频分析能力的竞技场式基准，旨在解决传统选择题基准（如 VideoMME、LongVideoBench）与真实用户需求脱节及人类标注成本高（每小时仅 7 个样本）的问题。该基准通过用户模拟（生成 3 类关联度不同的用户角色及开放式问题）、peer battles（随机抽取两模型对决）、自动判断（以 GPT-4o 为裁判，与人类判断一致性达 87.29%）、故障驱动进化（基于模型错误生成更难问题）四大核心组件实现自动化，并采用改进的ELO 评分系统对模型排名。实验基于 2,881 个平均时长 479 秒的视频（覆盖 10 类场景、4 个时长区间）评估 11 个 SOTA LMMs，发现GPT-4o以 ELO 1505.69 居首，开源模型中Aria表现最佳（ELO 1119.99）但与闭源模型差距显著（-385.7）；同时配套推出VideoAutoBench（基于人类标注子集，用 GPT-4o 快速评估），其排名与 VideoAutoArena 高度一致，共同构成低成本、可扩展的用户中心型视频分析评估框架。

- [x] **OVBench**: How Far is Your Video-LLMs from Real-World Online Video Understanding? [Paper](https://arxiv.org/pdf/2501.05510) [Code](https://github.com/JoeLeelyf/OVO-Bench?tab=readme-ov-file)

  Li 等人（2025）提出OVO-Bench（Online-VideO-Benchmark），这是首个聚焦在线视频理解的基准，旨在评估 Video-LLMs 的时间感知能力（基于提问时间戳动态推理），弥补现有离线基准（如 VideoMME、LongVideoBench）无法模拟真实实时场景的缺陷。该基准包含644 个独特视频（覆盖 7 大领域，时长从数分钟到 30 分钟）和2,814 个带精确时间戳的元标注，通过 “回溯追踪（Backward Tracing）、实时视觉感知（Real-Time Visual Perception）、前瞻主动响应（Forward Active Responding）” 三大核心场景，设计 12 类任务评估模型能力。实验对 11 个 SOTA Video-LLMs（含闭源模型 GPT-4o、Gemini 1.5 Pro 及开源模型 Qwen2-VL、LLaVA-OneVision 等）评估发现，即使顶级闭源模型（如 Gemini 1.5 Pro 整体准确率 63.00%）也远低于人类水平（92.81%），且开源模型与闭源模型差距显著（顶级开源模型 Aria 整体准确率 57.15%）；同时，现有模型存在时间优先级缺失（如 Gemini 1.5 Pro 在空间理解任务仅 58.43% 准确率）、幻觉严重（开源模型幻觉检测准确率普遍低于 30%）、推理延迟高（64 帧输入时平均响应需 4 秒）三大核心问题，为在线 Video-LLMs 的发展提供关键评估依据。

- [x] **ECBench**: Can Multi-modal Foundation Models Understand the Egocentric World? A Holistic Embodied Cognition Benchmark [Paper](https://arxiv.org/abs/2501.05031) [Code](https://github.com/Rh-Dang/ECBench)
- [x] Localizing Events in Videos with Multimodal Queries [Paper](https://arxiv.org/abs/2406.10079) [Code](https://github.com/icq-benchmark/icq-benchmark)
- [x] **ICQ**: Is `Right' Right? Enhancing Object Orientation Understanding in Multimodal Language Models through Egocentric Instruction Tuning [Paper](https://arxiv.org/abs/2411.16761) [Code](https://github.com/jhCOR/EgoOrientBench)

  多模态大语言模型（MLLMs） 在物体朝向理解上存在显著不足，核心原因是训练数据（如 MS-COCO、LAION-5B）中物体朝向标注不一致（部分基于观察者视角、部分基于物体视角），导致其理解与用户意图错位。为解决该问题，作者提出以自我为中心的指令微调（Egocentric Instruction Tuning）：基于用户视角制定统一标注标准（将朝向分为 8 类），利用 ImageNet 手动标注数据生成三类指令数据（调用 MLLM 细节识别能力、LLM 先验知识、 orientation 关系理解），通过 LoRA 微调 LLM 与桥接层（冻结视觉编码器）；同时构建EgoOrientBench 基准（覆盖 ImageNet 等 5 个跨领域数据集、3 个梯度任务）。实验表明，微调后 MLLMs 的朝向理解显著提升（如 LLaVA-1.5 在 Choose 任务平均准确率从 17.9% 升至 33.7%），且在 MME、MMStar 等通用基准上性能无衰减，还能改善行人方向预测、空间推理等实际任务表现。

- [x] **VidHalluc**: Evaluating Temporal Hallucinations in Multimodal Large Language Models for Video Understanding [Paper](https://arxiv.org/abs/2412.03735) [Page](https://vid-halluc.github.io/)

  针对多模态大语言模型（MLLMs）在视频理解中存在的时间幻觉问题（动作误判、时序错乱、场景转换误读），现有基准存在规模小、侧重静态内容、问题类型单一的局限，本文提出VIDHALLUC—— 当前最大的视频幻觉评估基准，包含5,002 个视频与9,295 个 QA 对，聚焦动作（ACH）、时间序列（TSH）、场景转换（STH） 三类幻觉；同时提出DINO-HEAL，一种无需训练的方法，通过 DINOv2 提取的空间显著性图重加权视觉编码器（如 CLIP）特征，增强关键区域关注，在 VIDHALLUC 上实现平均 3.02% 的幻觉缓解提升，其中对时间序列幻觉改善最显著（如 VideoLLaMA2 提升 18.33%）。实验表明，多数 MLLMs（含开源与闭源）在三类幻觉上均存在漏洞，GPT-4o 表现最优但仍低于人类，DINO-HEAL 可兼容多种视觉编码器，为资源受限场景提供高效解决方案。

- [x] **VidComposition**: Can MLLMs Analyze Compositions in Compiled Video? [Paper](https://arxiv.org/abs/2411.10979v1) [Code](https://github.com/yunlong10/VidComposition)
- [x] Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly [Paper](https://arxiv.org/abs/2406.10638)

  该研究揭示了多模态大型语言模型（MLLMs）能准确理解视觉内容却对误导性 “负面问题” 回答错误的现象，为此手动构建了包含12 类问题、1786 个样本（893 个正向问题 + 893 个负向问题）的 MMVU 基准，并提出 “误导率（MR）” 和 “鲁棒性准确率（RA）” 作为评估指标；通过分析发现问题根源在于现有训练数据集偏正向样本、模型对视觉 token 注意力远低于系统 / 问题 token，进而提出两种解决方案：一是构建含112k 正负样本对的 MMVU-Train 数据集，二是设计 “内容引导优化（CGR）” 和 “视觉注意力优化（VAR）” 策略；实验显示，该方案显著提升模型鲁棒性，例如 GPT-4o 在 MMVU 测试集上 RA 达65.06%，MMVU-Train 可使 VILA1.5-13B 的 RA 提升11.98%、MR 降低17.28%。

- [x] **VL-RewardBench**: A Challenging Benchmark for Vision-Language Generative Reward Models [Paper](https://arxiv.org/abs/2411.17451) [Page](https://vl-rewardbench.github.io/)
- [x] Benchmarking Large Vision-Language Models via Directed Scene Graph for Comprehensive Image Captioning [Paper](https://arxiv.org/abs/2412.08614) [Code](https://github.com/LuFan31/CompreCap)
- [x] **OpenING**: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation [Paper](https://arxiv.org/abs/2411.18499) [Code](https://github.com/LanceZPF/OpenING)
- [x] Automated Generation of Challenging Multiple-Choice Questions for Vision Language Model Evaluation [Paper](https://arxiv.org/abs/2501.03225) [Page](https://yuhui-zh15.github.io/AutoConverter-Website/)

  为解决现有视觉语言模型（VLMs）评估中开放式 VQA 基准评估不可靠的问题（规则 - based 方法与真实性能相关性仅 0.09，模型 - based 方法因版本更新分数波动达 6%），研究团队提出AutoConverter 多智能体框架，通过 Proposer（生成 5 类错误干扰项）、Reviewer（迭代反馈）、Selector（筛选难题）、Evaluator（5 分制正确性评分）、Refiner（迭代优化），自动将开放式问题转为多选题，保证正确性（仅 3% 高评分题错误）且提升难度（VLMs 准确率与人类题相当或更低）；基于此构建VMCBench 基准（20 个数据集转化 / 优化，共 9,018 题，覆盖通用理解、推理、OCR、文档图表 4 类任务），对 33 个 SOTA VLMs 评估发现，Qwen2-VL-72B（85.0%）表现最优，超越 GPT-4o（80.3%），且公私模型差距缩小、VLM 性能随规模提升，为 VLMs 提供可扩展、一致的评估新标准。
  
- [x] **VideoChat-Online**: Towards Online Spatial-Temporal Video Understanding via Large Video Language Models [Paper](https://arxiv.org/abs/2501.00584v1) [Page](https://videochat-online.github.io/)
- [x] **ODE**: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models [Paper](https://arxiv.org/abs/2409.09318)
- [x] **VideoEspresso**: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection [Paper](https://arxiv.org/abs/2411.14794v1) [Code](https://github.com/hshjerry/VideoEspresso)
