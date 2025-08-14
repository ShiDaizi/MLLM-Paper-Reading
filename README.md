# MLLM-Paper-Reading
This is a paper reading repository for recording my list of read papers.

## 📖 Table of Contents
- [MLLM-Paper-Reading](#mllm-paper-reading)
  - [📖 Table of Contents](#-table-of-contents)
  - [Image LLMs](#image-llms)

## Image LLMs
- [x] **LLaVA-Critic**: Learning to Evaluate Multimodal Models. CVPR2025 [Paper](https://arxiv.org/abs/2410.02712) [Page](https://llava-vl.github.io/blog/2024-10-03-llava-critic/)
  
  This paper introduces LLaVA-Critic, the first open-source large multimodal model (LMM) designed as a generalist evaluator for assessing performance across various multimodal tasks. Trained on a high-quality critic instruction-following dataset comprising 46k images and 113k evaluation samples (including both pointwise and pairwise settings), LLaVA-Critic demonstrates effectiveness in two key areas. Firstly, as an LMM-as-a-Judge, it provides reliable evaluation scores, achieving performance on par with or surpassing GPT models across multiple benchmarks, with high correlation with GPT-4o in instance-level scoring and model-level ranking. Secondly, in preference learning, it generates effective reward signals for iterative Direct Preference Optimization (DPO), outperforming human feedback-based reward models in enhancing model alignment capabilities. The model, built by fine-tuning LLaVA-OneVision, preserves original visual capabilities while offering a cost-effective, open-source alternative to commercial evaluators, supporting tasks like visual chat, detailed description, and hallucination detection. This work highlights the potential of open-source LMMs in self-critique and evaluation, paving the way for scalable alignment feedback mechanisms.

- [x] **Img-Diff**: Contrastive Data Synthesis for Multimodal Large Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2408.04594) [Code](https://github.com/modelscope/data-juicer/tree/ImgDiff)

  This paper introduces Img-Diff, a novel contrastive data synthesis method designed to enhance fine-grained image recognition capabilities in Multimodal Large Language Models (MLLMs). The approach generates high-quality datasets of "object replacement" samples by creating pairs of similar images with subtle object variations, identifying difference regions via a Difference Area Generator, and producing precise difference descriptions using a Difference Captions Generator. The resulting Img-Diff dataset, comprising 12,688 instances, effectively improves MLLMs' performance when used for fine-tuning. Experimental results show that fine-tuned models (e.g., LLaVA-1.5-7B, MGM-7B, InternVL2-8B) achieve significant gains on image difference benchmarks, with MGM-7B surpassing state-of-the-art models like GPT-4V and Gemini by up to 12 points on the MMVP benchmark. Additionally, the models demonstrate an average improvement of 3.06% across eight MLLM benchmarks, validating the dataset's ability to enhance both image difference recognition and overall visual understanding. The dataset exhibits high quality (over 70% accurate difference descriptions) and diversity (covering 1,203 object categories), offering valuable insights for multimodal data synthesis.

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

- [ ] Devils in Middle Layers of Large Vision-Language Models: Interpreting, Detecting and Mitigating Object Hallucinations via Attention Lens. CVPR2025 [Paper](https://arxiv.org/abs/2411.16724)
- [ ] **HoVLE**: Unleashing the Power of Monolithic Vision-Language Models with Holistic Vision-Language Embedding. CVPR2025 [Paper](https://arxiv.org/abs/2412.16158)
- [ ] **VoCo-LLaMA**: Towards Vision Compression with Large Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2406.12275v2) [Code](https://github.com/Yxxxb/VoCo-LLaMA?tab=readme-ov-file)
- [ ] Perception Tokens Enhance Visual Reasoning in Multimodal Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2412.03548) [Page](https://aurora-perception.github.io/)
- [ ] **Florence-VL**: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion. CVPR2025 [Paper](https://arxiv.org/abs/2412.04424) [Page](https://jiuhaichen.github.io/florence-vl.github.io/)
- [ ] **ATP-LLaVA**: Adaptive Token Pruning for Large Vision Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2412.00447) [Page](https://yxxxb.github.io/ATP-LLaVA-page/)
- [ ] Accelerating Multimodel Large Language Models by Searching Optimal Vision Token Reduction. CVPR2025 [Paper](https://arxiv.org/abs/2412.00556)
- [ ] **VisionZip**: Longer is Better but Not Necessary in Vision Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2412.04467) [Code](https://github.com/dvlab-research/VisionZip)
