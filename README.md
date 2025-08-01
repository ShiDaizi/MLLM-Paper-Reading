# MLLM-Paper-Reading
This is a paper reading repository for recording my list of read papers.

## Image LLMs
- [x] **LLaVA-Critic**: Learning to Evaluate Multimodal Models. CVPR2025 [Paper](https://arxiv.org/abs/2410.02712) [Page](https://llava-vl.github.io/blog/2024-10-03-llava-critic/)

This paper introduces LLaVA-Critic, the first open-source large multimodal model (LMM) designed as a generalist evaluator for assessing performance across various multimodal tasks. Trained on a high-quality critic instruction-following dataset comprising 46k images and 113k evaluation samples (including both pointwise and pairwise settings), LLaVA-Critic demonstrates effectiveness in two key areas. Firstly, as an LMM-as-a-Judge, it provides reliable evaluation scores, achieving performance on par with or surpassing GPT models across multiple benchmarks, with high correlation with GPT-4o in instance-level scoring and model-level ranking. Secondly, in preference learning, it generates effective reward signals for iterative Direct Preference Optimization (DPO), outperforming human feedback-based reward models in enhancing model alignment capabilities. The model, built by fine-tuning LLaVA-OneVision, preserves original visual capabilities while offering a cost-effective, open-source alternative to commercial evaluators, supporting tasks like visual chat, detailed description, and hallucination detection. This work highlights the potential of open-source LMMs in self-critique and evaluation, paving the way for scalable alignment feedback mechanisms.

- [ ] **Img-Diff**: Contrastive Data Synthesis for Multimodal Large Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2408.04594) [Code](https://github.com/modelscope/data-juicer/tree/ImgDiff)
- [ ] **FlashSloth**: Lightning Multimodal Large Language Models via Embedded Visual Compression. CVPR2025 [Paper](https://arxiv.org/abs/2412.04317) [Code](https://github.com/codefanw/FlashSloth)
- [ ] **BlueLM-V-3B**: Algorithm and System Co-Design for Multimodal Large Language Models on Mobile Devices. CVPR2025 [Paper](https://arxiv.org/abs/2411.10640v1)
- [ ] **Insight-V**: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2411.14432) [Code](https://github.com/dongyh20/Insight-V)
- [ ] **Critic-V**: VLM Critics Help Catch VLM Errors in Multimodal Reasoning. CVPR2025 [Paper](https://arxiv.org/abs/2411.18203)
- [ ] **Mono-InternVL**: Pushing the Boundaries of Monolithic Multimodal Large Language Models with Endogenous Visual Pre-training. CVPR2025 [Paper](https://arxiv.org/abs/2410.08202) [Code](https://internvl.github.io/blog/2024-10-10-Mono-InternVL/)
- [ ] **DivPrune**: Diversity-based Visual Token Pruning for Large Multimodal Models. CVPR2025 [Paper](https://arxiv.org/abs/2503.02175) [Code](https://github.com/vbdi/divprune)
- [ ] **ODE**: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2409.09318)
- [ ] Augmenting Multimodal LLMs with Self-Reflective Tokens for Knowledge-based Visual Question Answering. CVPR2025 [Paper](https://arxiv.org/abs/2411.16863) [Code](https://github.com/aimagelab/ReflectiVA)
- [ ] **AGLA**: Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention. CVPR2025 [Paper](https://arxiv.org/abs/2406.12718) [Code](https://github.com/Lackel/AGLA)
- [ ] **ICT**: Image-Object Cross-Level Trusted Intervention for Mitigating Object Hallucination in Large Vision-Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2411.15268v1)
- [ ] Can Large Vision-Language Models Correct Grounding Errors By Themselves? CVPR2025 [Paper](https://openreview.net/pdf?id=fO1xnmW8T6)
- [ ] **Molmo and PixMo**: Open Weights and Open Data for State-of-the-Art Vision-Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2409.17146) [Page](https://molmo.allenai.org/blog)
- [ ] **Nullu**: Mitigating Object Hallucinations in Large Vision-Language Models via HalluSpace Projection. CVPR2025 [Paper](https://arxiv.org/abs/2412.13817) [Code](https://github.com/Ziwei-Zheng/Nullu)
- [ ] **HiRes-LLaVA**: Restoring Fragmentation Input in High-Resolution Large Vision-Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2407.08706)
- [ ] Devils in Middle Layers of Large Vision-Language Models: Interpreting, Detecting and Mitigating Object Hallucinations via Attention Lens. CVPR2025 [Paper](https://arxiv.org/abs/2411.16724)
- [ ] **HoVLE**: Unleashing the Power of Monolithic Vision-Language Models with Holistic Vision-Language Embedding. CVPR2025 [Paper](https://arxiv.org/abs/2412.16158)
- [ ] **VoCo-LLaMA**: Towards Vision Compression with Large Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2406.12275v2) [Code](https://github.com/Yxxxb/VoCo-LLaMA?tab=readme-ov-file)
- [ ] Perception Tokens Enhance Visual Reasoning in Multimodal Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2412.03548) [Page](https://aurora-perception.github.io/)
- [ ] **Florence-VL**: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion. CVPR2025 [Paper](https://arxiv.org/abs/2412.04424) [Page](https://jiuhaichen.github.io/florence-vl.github.io/)
- [ ] **ATP-LLaVA**: Adaptive Token Pruning for Large Vision Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2412.00447) [Page](https://yxxxb.github.io/ATP-LLaVA-page/)
- [ ] Accelerating Multimodel Large Language Models by Searching Optimal Vision Token Reduction. CVPR2025 [Paper](https://arxiv.org/abs/2412.00556)
- [ ] **VisionZip**: Longer is Better but Not Necessary in Vision Language Models. CVPR2025 [Paper](https://arxiv.org/abs/2412.04467) [Code](https://github.com/dvlab-research/VisionZip)
