
## Abstract


# InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation

Jinlai Liu*, Jian Han*, Bin Yan*, Hui Wu, Fengda Zhu, Xing Wang
Yi Jiang, Bingyue Peng, Zehuan Yuan†

ByteDance

{liujinlai.licio, hanjian.thu123, bin.yan, wuhui.321, fengdazhu}@bytedance.com,
{xing.wang, jiangyi.enjoy, bingyue.peng, yuanzehuan}@bytedance.com,

Codes and models: https://github.com/FoundationVision/InfinityStar

*Equal contribution. †Corresponding author: yuanzehuan@bytedance.com*

## Abstract

We introduce InfinityStar, a unified spacetime autoregressive framework for high-resolution image and dynamic video synthesis. Building on the recent success of autoregressive modeling in both vision and language, our purely discrete approach jointly captures spatial and temporal dependencies within a single architecture. This unified design naturally supports a variety of generation tasks such as text-to-image, text-to-video, image-to-video, and long-duration video synthesis via straightforward temporal autoregression. Through experiments, InfinityStar achieves a competitive score of 83.74 on VBench, marginally outperforming selected autoregressive models, and comparable to leading diffusion competitors such as HunyuanVideo. Without extra optimizations, our model generates a 5s, 720p video approximately 10× faster than leading diffusion-based methods. InfinityStar is among the first discrete autoregressive video generators capable of producing high-quality 720p videos suitable for industrial applications. We release all code and models to foster further research in efficient, high-quality video generation.


## Introduction


## 1 Introduction

Visual synthesis has witnessed remarkable progress in recent years, largely propelled by the scaling of Transformer architectures. In particular, video generation has attracted growing interest from both academia and industry, owing to its wide-ranging applications in content creation, world simulation, etc. At present, diffusion models [3, 18, 17, 28, 8, 39] lead the field by iteratively denoising latent representations to produce high-fidelity clips. Concurrently, autoregressive models [16, 30, 9] have been explored for their potential to unify image and video generation and to generalize over longer time horizons.

Despite their successes, each paradigm exhibits critical shortcomings. Video diffusion models excel at synthesizing fixed-length frame sequences by exploiting bidirectional attention, yet they incur substantial computational cost due to tens or even hundreds of sequential denoising steps, and they struggle to extend seamlessly to video extrapolation. Autoregressive methods based on next-token prediction, while inherently capable of streaming generation, often fall short in visual fidelity and suffer from prohibitive latency due to tens of thousands of inference steps.

These observations motivate the need for a generation framework that simultaneously possess high visual quality, efficiency and temporal generalization. Recently, Visual AutoRegressive modeling (VAR) [25] redefined image generation as a coarse-to-fine next-scale prediction. Its follow-up work, Infinity [13] further introduces bitwise modeling and scales up the vocabulary size, achieving

comparable performance to diffusion models while offering significant advantages in inference speed. Inspired by prior work [25, 13], we propose InfinityStar, a spacetime pyramid modeling approach aimed at unifying diverse visual generation tasks including text-to-image and video extrapolation. The model decomposes a video into an image pyramid and clip pyramids, separating static appearance from motion content. Key techniques include knowledge transfer from continuous video tokenizers, stochastic quantizer depth to balance scale information, and semantic scales repetition to refine earlier pyramid scales, collectively improving generation quality.

We train InfinityStar on large-scale video corpora to support up to 720p resolution and variable durations. On the VBench benchmark [38], InfinityStar establishes a new state-of-the-art among autoregressive video models, even surpassing industry-leading HunyuanVideo [17] (83.74 vs 83.24). Besides, InfinityStar shows a great advantage in terms of speed. Using visual tokenizers of the same compression rate, InfinityStar achieves a 10× reduction in inference latency relative to leading diffusion models.

In summary, the main contributions of our work are as follows:

1. We propose InfinityStar, a novel spacetime pyramid modeling framework that unifies diverse visual generation tasks, demonstrating superior flexibility and versatility.

2. InfinityStar is the first discrete autoregressive model capable of generating high-quality videos, outperforming existing autoregressive text-to-video models and matching the performance of leading diffusion models.

3. Compared to the inefficiency of existing autoregressive models and diffusion models, InfinityStar significantly accelerates high-quality video generation.


## Related Work


## 2 Related Work

### 2.1 Video Diffusion Models

Diffusion models excel at generating high-fidelity data by gradually denoising random noise and can naturally extend to video generation. Early attempts [2, 5, 36] are built on U-Net architectures, demonstrating the feasibility of this approach but falling short in producing sharp, temporally coherent frames due to limited model capacity. The advent of Diffusion Transformers (DiT [21]) marked a turning point. SORA [3] harnessed DiT’s scaling ability to process spatio-temporal patches at scale, dramatically improving both video consistency and generation quality. Inspired by SORA’s success, industry efforts [34, 17, 28] have further advanced the field, pushing video generation to new heights. Although video diffusion models deliver outstanding quality, their slow generation speed hinders the production of high-resolution, long-duration videos.

### 2.2 Video AutoRegressive Models

Another class of methods [30, 9, 16] employs autoregressive models for video generation. Inspired by the success of LLMs, these works predict video tokens in specific orders using an autoregressive Transformer. For example, Emu3 [30] performs next-token prediction along both spatial and temporal axes, while NOVA [9] first predicts spatial tokens set-by-set and subsequently proceeds frame-by-frame in the temporal dimension. Although achieving preliminary progress, they require hundreds to thousands of inference steps, resulting in prohibitively low generation efficiency. In contrast, recent advances in next-scale prediction [25, 13] have demonstrated state-of-the-art performance in image synthesis, offering both improved quality and markedly faster inference. In this work, we extend the next-scale prediction paradigm to the unified tasks of image and video generation.

### 2.3 Discrete Video Tokenizers

For a long time, discrete [33, 35] and continuous [17, 34, 28] video tokenizers have been developed independently. Although some works [1, 29] provide both discrete and continuous tokenizers, the

Infinity for Image Generation. Infinity [13] decomposes an image into a sequence of hierarchical token blocks using a visual tokenizer and models the relationship between tokens by a visual autoregressive Transformer (VAR Transformer). To cover images of various sizes, Infinity pre-defines a list of token block sizes {(h1, w1), ...(hK, wK)}, called scale schedule. The size (hi, wi) in scale schedule grows as i increases, forming a pyramid-like structure, which we refer as image pyramid in later discussion. Next we introduce the training and inference procedure of Infinity.

In the first training stage, a visual tokenizer learns to reconstruct the raw image and compress it into a sequence of discrete tokens, which can be modeled by the VAR Transformer in the next stage. Specifically, the tokenizer first encodes the raw images into compact latents, then transforms latents into K discrete residual token blocks (r1, r2, ..., rK) using a bitwise multi-scale residual quantizer [13]. Each token block ri consists of hi × wi discrete tokens of d-dim with vocabulary size of 2^d. Then in the second stage, a VAR Transformer is trained to predict next residual token block rk conditioned on text embedding ψ(t) and former token blocks r<k. Formally, in each step, VAR Transformer predicts a conditional probability p(rk|r<k, ψ(t)). During the inference, Infinity generates an image by running the VAR Transformer K times autoregressively, merging the predicted tokens and running the tokenizer decoder once.

Network configurations are usually not aligned. For example, Cosmos [1] chooses 6 and 16 as latent dimensions in its discrete and continuous variants respectively. This misalignment hinders the knowledge reuse between two types of tokenizers. As a result, most mainstream discrete video tokenizers are either trained from scratch [1] or starting from a pretrained discrete image tokenizer [35, 29]. However, these training strategies have the following drawbacks. First, training from scratch is inefficient and converges slowly. Second, weights pretrained on static images are not optimal for video reconstruction. To alleviate these deficiencies, we propose a new training strategy, which inherits the architecture and knowledge of a trained continuous video tokenizer. Experiments show that this strategy significantly boosts the convergence of discrete video tokenizers.


## Method



![Figure 11](image_11.jpg)
<!-- Caption: Method, index 11 -->


## 3.2 Spacetime Pyramid Modeling for Unified Generation

Extending the spatial-only next-scale prediction paradigm of Infinity [13] to video generation presents a primary challenge: how to incorporate the temporal dimension. The straightforward strategies are either letting time grow uniformly, i.e., from $(1, 1, 1)$ to $(T, H, W)$, or keeping time constant, i.e., from $(T, 1, 1)$ to $(T, H, W)$. We empirically found that letting time grow uniformly produces flickering videos. As for the constant time pyramid, we refer to it as the pseudo-spacetime pyramid. Despite its conceptual simplicity, it suffers from two fundamental limitations. First, the treatment of videos differs markedly from that of images, preventing a text-to-video (T2V) model from effectively leveraging the knowledge learned by a text-to-image (T2I) model and complicating its extension to tasks such as image-to-video (I2V). Second, because appearance and motion in videos are coupled in this design, the model faces significant difficulty in accurately fitting both aspects.

To overcome these challenges, we propose a novel spacetime pyramid modeling framework as shown in Fig. 1. Each video is decomposed into sequential clips $\{c_1, c_2, \cdots, c_N\}$. We regard the first frame as $c_1$ (i.e., $T = 1$) to encode video main static appearance cues specifically and other clips share an equal duration $T > 1$. Each clip is represented as a 3D volume pyramid with $K$ scales, each denoted by residual token blocks $r_k$ of dimensions $(T, h_k, w_k)$. Contrary to prior approaches, temporal extension varies across clips but not within pyramid scales, effectively segmenting temporal modeling to inter-clip autoregression while maintaining spatial scaling within each clip. Mathematically, the clip tokens are generated auto-regressively across scales as:

$$
p(r_1^1, \ldots, r_K^1) = \prod_{k=1}^{K} p(r_k^1 \mid r_1^1, \ldots, r_{k-1}^1, \psi(t)),
\tag{1}
$$

For inter-clip predictions, clips are generated sequentially conditioned on prior clip predictions and the text input in an autoregressive manner. In this way, we could generate infinitely long videos theoretically. Formally, the autoregressive likelihood of the whole video can be expressed as:

$$
p(r_1^1, \ldots, r_K^N) = \prod_{c=1}^{N} \prod_{k=1}^{K} p(r_k^c \mid r_1^c, \ldots, r_{k-1}^c, \psi(t)),
\tag{2}
$$

## 3.3 Visual Tokenizer

Training video tokenizers faces greater challenges than training image tokenizers. First, training tokenizers on videos of tens of frames is much computationally heavier than training on static images. Therefore, training a video tokenizer from scratch is extremely time-consuming and suffers from slow convergence. Second, the scale schedule in videos leads to more imbalanced information distribution, where most information is concentrated in the last few scales. This brings great difficulties to the optimization of VAR Transformer. To solve these challenges, we introduce two techniques, knowledge inheritance from continuous video tokenizer and stochastic quantizer depth.

Knowledge Inheritance from Continuous Video Tokenizer. Instead of designing and training a discrete video tokenizer from scratch, we inherit the architecture and weights of a trained continuous video tokenizer, i.e., video VAE. Specifically, we first insert a parameter-free quantizer between the pre-trained VAE encoder and the decoder. The quantizer is based on binary spherical quantization [40], being similar to that of Infinity [13] but with new spacetime pyramid scale schedule. This does not introduce any new parameter like codebook in VQ [26] and well retains knowledge of the original VAE. As shown in Fig. 5, the discrete video tokenizer reconstructs videos decently, even without any fine-tuning. To further improve the reconstruction quality, we fine-tune the new tokenizer jointly on images and videos like previous works [29, 1]. During the fine-tuning, the KL loss of the original VAE is replaced with the commitment loss plus the entropy penalty [40]. As shown in Fig. 5, with the help of knowledge of continuous video VAE, the convergence accelerates dramatically.

Stochastic Quantizer Depth. When tokenizing videos using the spacetime pyramid schedule, the information distribution on different scales gets extremely imbalanced. Specifically, there are only a few tokens in the early scales, while there are tens of thousands of tokens in the last scales. Thus the tokenizer tends to reconstruct videos solely relying on tokens from the last few scales and not to learn useful representation in early scales as shown in Fig. 6 (left). However, this imbalanced distribution is difficult to model using VAR Transformer because the dependence between the latter token blocks and the former ones is weak. To alleviate this problem, we propose a regularization called stochastic quantizer depth. During training, each one of the last $N$ scales has a probability $p$ of being discarded.

In this way, there are $2^N$ possible scale schedules during training. This requires the tokenizer to reduce the reliance on last scales and store more information in tokens of early scales. As in Fig. 6 (right), with the help of this regularization, the reconstruction results of early scales become much clearer. This balanced information distribution makes the training of VAR Transformer easier.

## 3.4 Spacetime Autoregressive Transformer

To accommodate the newly introduced temporal dimension, enhance the quality of generated videos, and alleviate the substantial computational overhead associated with a large number of tokens, we propose the following modifications to the VAR Transformer: Spacetime RoPE, Semantic Scale Repetition, and Spacetime Sparse Attention. We put Spacetime RoPE in the appendix.

Semantic Scale Repetition. With carefully crafted positional encodings, InfinityStar can already generate videos of acceptable quality. However, we observe that the structural coherence and motion dynamics in these outputs remain suboptimal. As shown in Figure 6, the overall layout and the placement of foreground objects are determined by the early scales of the clip pyramid—what we term the “semantic scales.” This observation motivates us to enhance generation fidelity at these semantic scales. To this end, we introduce a simple yet effective technique called semantic scale repetition. Concretely, if a clip pyramid comprises $K$ scale tuples, we repeat the first $K_s$ tuples $N$ times, thereby reinforcing the semantic-level representations. In this way, every earlier residual $r_k$ undergoes multiple rounds of refinement, improving the generation quality of semantics and the performance in complex scenarios with large motion. Given that the tokens at these early scales account for only a small fraction of the total token count, the additional computational overhead incurred by repeating them is negligible.

| VAR-like Attention mask | Switti-like Attention mask | Spacetime Attention mask |
| :---------------------: | :------------------------: | :----------------------: |

Figure 2: Illustration of three causal attention variants. We plot three pyramids on the scale sizes = (1, 2, 3) for visualization simplicity. From left to right, VAR block-wise causal mask with full history, Switti block-wise non-causal mask with full history, and spacetime sparse attention.

Spacetime Sparse Attention. Autoregressive video generation faces significant challenges due to the high computational costs of long context. As shown on the left of Fig. 2, Infinity [13] employs a block-wise causal mask for single-pyramid modeling. Switti [27] verifies that conditioning next-scale predictions solely on inputs from preceding scales is sufficient, resulting in a sparser attention mask as in the middle of Fig. 2. For long video generation, it’s necessary to attend to history tokens to achieve temporal consistency. However, attending full history leads to an explosively long sequence. Considering each clip corresponds to 5s, which is sufficient to maintain temporal consistency, here we only attend to the last scale of the preceding clip. Finally, we obtain a highly sparse attention as shown in Fig. 2 (right). Our spacetime sparse attention reduces attention computational overhead during training and inference, maintaining comparable or improved performance relative to full attention, as discussed in the experimental section.


## Experiments


# 4 Experiments

## 4.1 Implementation

Datasets. The training data comprises text-to-image and text-to-video sets totaling approximately 221M samples, sourced from COYO [4], OpenImages [19], vendor suppliers, and internal collections. However, the exact splits between training and validation are unspecified. The 16M video dataset includes diverse resolutions, but detailed distribution statistics and balancing criteria are not provided, complicating reproducibility and fair comparison.


![Figure 17](image_17.jpg)
<!-- Caption: Experiments, index 17 -->



![Figure 18](image_18.jpg)
<!-- Caption: Experiments, index 18 -->



![Figure 19](image_19.jpg)
<!-- Caption: Experiments, index 19 -->


Figure 3: Text-to-image and text-to-video examples.

Model and Training. After inserting the patchify and unpatchify layers between Wan 2.1 VAE’s encoder and decoder, we obtain a video tokenizer with a compression rate of 4 × 16 × 16 and a latent dimension of 64. BSQ quantization is adopted with a vocabulary of size 2^64. In contrast to using a vocabulary of size 2^64 for all scales, we use a vocabulary of size 2^16 for the former small scales and 2^64 for the latter large scales. We empirically find that it boosts convergence and has a negligible impact on reconstruction quality. Starting from Wan 2.1 VAE pretrained weights, the discrete tokenizer was fine-tuned on multi-resolution images and videos for 30K iterations with a constant learning rate of 1e-4, using the Adam optimizer with unspecified momentum and weight decay settings. InfinityStar training proceeded through four progressive stages from T2I to T2V with videos at increasing resolutions: 192p, 360p, 720p. The global batch sizes and learning rate schedules are as detailed, but exact optimizer variants, warm-up, or gradient clipping procedures are not reported. Each clip pyramid has 80 frames at 16 fps; semantic scale repetition repeats the first 7 scales 6 times. Details about infrastructure optimizations are presented in the appendix.

4.2 Text-to-Image Generation

The upper part of Fig. 3 shows generated images from our InfinityStar-T2I model, showcasing InfinityStar’s strength in generating high-fidelity and photorealistic images across various categories and image styles. We also carry out quantitative evaluation on the GenEval [12] and DPG [14] benchmarks. As shown in Tab. 1, InfinityStar attains leading scores on GenEval and DPG benchmarks. However, no variance or significance metrics are presented, which limits assessment of the robustness of these improvements beyond the margin of experimental variability.


![Figure 23](image_23.jpg)
<!-- Caption: Experiments, index 23 -->


Table 1: Evaluation on the GenEval [12] and DPG [14] benchmark. † result is with prompt rewriting.

| Methods                   | # Params | GenEval↑<br/>Two Obj. | GenEval↑<br/>Position | GenEval↑<br/>Color Attri. | GenEval↑<br/>Overall | DPG↑<br/>Global | DPG↑<br/>Relation | DPG↑<br/>Overall |
| ------------------------- | -------- | --------------------- | --------------------- | ------------------------- | -------------------- | --------------- | ----------------- | ---------------- |
| Diffusion Models          |          |                       |                       |                           |                      |                 |                   |                  |
| SDXL [22]                 | 2.6B     | 0.74                  | 0.15                  | 0.23                      | 0.55                 | 83.27           | 86.76             | 74.7             |
| PixArt-Sigma [6]          | 0.6B     | 0.62                  | 0.14                  | 0.27                      | 0.55                 | 86.89           | 86.59             | 80.5             |
| SD3 (d=38) [10]           | 8B       | 0.89                  | 0.34                  | 0.47                      | 0.71                 | -               | -                 | -                |
| SANA-1.0 [31]             | 1.6B     | -                     | -                     | -                         | 0.66                 | -               | -                 | 84.8             |
| FLUX-dev [20]             | 12B      | -                     | -                     | -                         | 0.67                 | -               | -                 | 84.0             |
| FLUX-schnell [20]         | 12B      | -                     | -                     | -                         | 0.71                 | -               | -                 | 84.8             |
| AutoRegressive Models     |          |                       |                       |                           |                      |                 |                   |                  |
| LlamaGen [23]             | 0.8B     | 0.34                  | 0.07                  | 0.04                      | 0.32                 |                 |                   | 65.2             |
| Chameleon [24]            | 7B       | -                     | -                     | -                         | 0.39                 | -               | -                 | -                |
| Show-o [32]               | 1.3B     | 0.80                  | 0.31                  | 0.50                      | 0.68                 | -               | -                 | 67.5             |
| Emu3 [30]                 | 8B       | 0.81†                 | 0.49†                 | 0.45†                     | 0.66†                | -               | -                 | 81.6             |
| Infinity [13]             | 2B       | 0.85†                 | 0.49†                 | 0.57†                     | 0.73†                | 93.11           | 90.76             | 83.46            |
| InfinityStar-T2I          | 8B       | 0.90†                 | 0.62†                 | 0.67†                     | 0.79†                | 91.68           | 91.87             | 86.55            |

Quantitative results demonstrate InfinityStar’s strong capabilities in image generation following users’ prompts.

## 4.3 Text-to-Video Generation

In the lower part of Fig. 3, we present the generated videos of InfinityStar regarding user prompts. The generated videos successfully capture the semantic information in user prompts while maintaining high aesthetics and visual quality. Especially for the second example in Fig. 3, the generated video accurately restores the delicate movements of the characters flipping through sketchbooks, talking while pointing to different parts of the drawings. In Tab. 2, we compare InfinityStar with leading diffusion and autoregressive models on VBench. Our model achieves a score of 83.74 with prompt rewriting applied, while other baselines have variable usage of prompt enhancements. Statistical variance was not reported, limiting the interpretability of performance differences. These results demonstrate that, through its novel spacetime autoregressive design, InfinityStar not only pushes the capabilities of discrete autoregressive video models but also attains performance on par with—and in some cases superior to—state-of-the-art diffusion methods.

| 0s | Reference Video | 5s | Extrapolated Video | Extrapolated Video | Extrapolated Video | 10s |
| --- | --------------- | -- | ------------------ | ------------------ | ------------------ | --- |
| In a realistic medium shot, a camera steadily moves to follow a middle-aged Chinese woman. She wears a light brown top paired with blue jeans and holds a smartphone in her hand. She walks casually inside a mall. In the background, there is a clothing store and a wall adorned with cartoon patterns. | | | | | | |
| The camera slowly zooms in on the lips of a 30-year-old Chinese woman who is applying moisturizing lipstick. She has a natural smile adorning her face, set against a bright indoor background. | | | | | | |

Figure 4: Zero-shot video extrapolation examples.

Zero-shot Generation. Trained on text-to-video data, InfinityStar can conditionally generate videos from images or video clips without fine-tuning. Preliminary qualitative results in Fig. 4 suggest temporal coherence and semantic consistency, but comprehensive quantitative evaluation and robustness tests remain future work. Zero-shot I2V samples are presented in the appendix.

## 4.4 Ablation Study

Visual Tokenizer. As shown in Fig. 5 and Tab. 3, loading weights of a continuous video tokenizer significantly speeds up convergence and achieves the best reconstruction results. As shown in Fig. 6, stochastic quantizer depth largely improves the reconstruction quality of early scales. In terms of generation, using VAE with SQD leads to a notable improvement in VBench scores (81.28 vs. 81.07 as shown in Tab. 4). Moreover, we observe that SQD contributes to faster convergence during video generation training.


![Figure 27](image_27.jpg)
<!-- Caption: Experiments, index 27 -->



![Figure 28](image_28.jpg)
<!-- Caption: Experiments, index 28 -->


Table 2: Evaluation on the VBench benchmark. † result is with prompt rewriting.

| Models                    | # Params | Human Action | Scene | Multiple Objects | Appearance Style | Quality Score | Semantic Score | Overall |
| ------------------------- | -------- | ------------ | ----- | ---------------- | ---------------- | ------------- | -------------- | ------- |
| Diffusion Models          |          |              |       |                  |                  |               |                |         |
| AnimateDiff-V2            | 1.5B     | 92.60        | 50.19 | 36.88            | 22.42            | 82.90         | 69.75          | 80.27   |
| VideoCrafter-2.0 [5]      | 1.5B     | 95.00        | 55.29 | 40.66            | 25.13            | 82.20         | 73.42          | 80.44   |
| OpenSora V1.2 [41]        | 1.1B     | 85.80        | 42.47 | 58.41            | 23.89            | 80.71         | 73.30          | 79.23   |
| Show-1 [37]               | 6B       | 95.60        | 47.03 | 45.47            | 23.06            | 80.42         | 72.98          | 78.93   |
| Gen-3 [11]                | -        | 96.40        | 54.57 | 53.64            | 24.31            | 84.11         | 75.17          | 82.32   |
| CogVideoX-5B [34]         | 5B       | 99.40        | 53.20 | 62.11            | 24.91            | 82.75         | 77.04          | 81.61   |
| HunyuanVideo [17]         | 13B      | 94.40        | 53.88 | 68.55            | 19.80            | 85.09         | 75.82          | 83.24   |
| Wan 2.1 [28]              | 14B      | 98.80        | 53.67 | 81.44            | 21.13            | 85.64         | 80.95          | 84.70   |
| Autoregressive Models     |          |              |       |                  |                  |               |                |         |
| Nova [9] †                | 0.6B     | 95.20        | 54.06 | 77.52            | 20.92            | 80.39         | 79.05          | 80.12   |
| Emu3 [30]                 | 8B       | 77.71        | 37.11 | 44.64            | 20.92            | 84.09         | 68.43          | 80.96   |
| InfinityStar †            | 8B       | 96.43        | 52.08 | 78.66            | 21.81            | 84.73         | 79.78          | 83.74   |

Figure 5: Influence of pretrained weights on reconstruction and convergence. The left sub-figure shows the reconstructed frames using different pretrained weights without fine-tuning. Loading the weights of the continuous video tokenizer achieves the best results. The right sub-figure shows that training with the pretrained video tokenizer converges significantly faster than the other two strategies.

Table 3: Reconstruction metrics on an internal high-motion video benchmark (480p, 81 frames).

| Pretrained Weights   | PSNR (↑) | SSIM (↑) | LPIPS (↓) |
| -------------------- | -------- | -------- | --------- |
| Continuous Video VAE | 33.37    | 0.94     | 0.065     |
| Image VAE            | 29.10    | 0.90     | 0.123     |
| None                 | 30.04    | 0.90     | 0.124     |

Figure 6: The influence of stochastic quantizer depth. Sub-figure (s_i, n_t) represents the reconstructed frame n_t using all tokens from the image pyramid plus tokens of the first i scales in the clip pyramid. SQD significantly improves the reconstruction quality of early scales. Besides, the earlier scales correspond to global semantics, while the later ones are responsible for local visual details.

Pseudo-Spacetime Pyramid vs. Spacetime Pyramid. As illustrated in Fig. 7, videos generated by the pseudo-spacetime pyramid lack visual details and deliver simpler motion. In contrast, the spacetime pyramid generates videos with richer details and higher motion. Besides, the spacetime pyramid improves VBench’s overall score from 80.30 to 81.28 as illustrated in Tab. 4. These experiments support the


![Figure 34](image_34.jpg)
<!-- Caption: Experiments, index 34 -->



![Figure 35](image_35.jpg)
<!-- Caption: Experiments, index 35 -->



![Figure 36](image_36.jpg)
<!-- Caption: Experiments, index 36 -->


Single Pyramid | Kitchen scene 1 | Kitchen scene 2 | Kitchen scene 3 | Kitchen scene 4 | Kitchen scene 5
Image Clips | Kitchen scene 6 | Kitchen scene 7 | Kitchen scene 8 | Kitchen scene 9 | Kitchen scene 10

A man stands in a well-lit kitchen with white cabinets, a large window, and various kitchen items on the counter and shelves. He gestures with his hand while speaking, then turns and walks toward a pot on the stove. The man reaches out to lift the lid of the pot.

Single Pyramid | Cable car scene 1 | Cable car scene 2 | Cable car scene 3 | Cable car scene 4 | Cable car scene 5
Image Clips | Cable car scene 6 | Cable car scene 7 | Cable car scene 8 | Cable car scene 9 | Cable car scene 10

The video shows a cable car system with a tower in the foreground, situated on a mountainous area with lush greenery. In the background, there is a cityscape with buildings and a river, partially obscured by fog. A cable car moves from the right toward the left.

Figure 7: Comparison between Pseudo-Spacetime Pyramid and Spacetime Pyramid. Spacetime Pyramid can generate videos with richer details and higher motion.

We hypothesize that the spacetime pyramid can decouple appearance and temporal information. The image pyramid corresponds to the appearance information and clip pyramids focus on subsequent motions. This decoupling makes it easier to learn video motions. In addition to advances in performance, the spacetime pyramid can unify T2I, T2V, and I2V tasks into one framework.

Semantic Scale Repetition. In Fig. 6, we can observe that the earlier scales correspond to semantic information, while the later ones are responsible for high-frequency details. Here we compare the generation results with and without semantic scale repetition. As shown in Fig. 8, semantic scale repetition is highly effective in improving the structure stability and motion quality. The quantitative results further confirm the significant gains. As shown in Tab. 4, semantic scale repetition improves VBench’s overall score from 75.72 to 81.28.

Spacetime Sparse Attention. In Tab. 4 and Tab. 5, we compare different attention mechanisms. Spacetime sparse attention shows superior performance to full attention in the VBench total score (81.28 vs. 80.77), while showing a significant advantage in saving computation and GPU VRAM. SSA reaches 1.5× speedup when generating 192p 16f. The efficiency advantage becomes larger as the resolution and duration grow. For 480p 16f, full attention fails due to OOM while SSA completes it within 44.7 s using 63 GB VRAM. We hypothesize that SSA produces better results than full attention because it reduces exposure bias. Full attention is more susceptible to accumulated errors. The reason we do not condition on smaller scales of the preceding clip is that it misses the former clips’ visual details and brings visual inconsistency between clips. Although it reaches 1.1× and 1.5× speedup for 192p and 480p 16f, we observe a significant performance drop in VBench from 81.28 to 80.75 as shown in Tab. 4. Therefore, the proposed spacetime sparse attention strikes a better balance between computational efficiency and visual quality.

4.5 Inference Latency

As shown in Tab. 6, we report the end-to-end inference latency measured on a single GPU, including both the text encoder and VAE decoder. Wan-2.1 [28] and Nova [9] were evaluated using their default GitHub configurations. Even without employing stronger compression, InfinityStar achieves a 32× speedup over Wan-2.1. Furthermore, despite its larger model size, InfinityStar delivers a 6× speedup compared to Nova. These results highlight our model’s significant efficiency advantage over both diffusion and autoregressive approaches.


## Conclusion


5 Conclusion and Limitation

We introduce InfinityStar, a unified spacetime autoregressive framework capable of synthesizing high-resolution images and dynamic, high-motion videos. By seamlessly integrating spatial and


## Experiments



![Figure 42](image_42.jpg)
<!-- Caption: Experiments, index 42 -->



![Figure 43](image_43.jpg)
<!-- Caption: Experiments, index 43 -->


Figure 8: Semantic Scale Repetition (SSR) greatly improves structural stability and motion quality.

Table 4: Comprehensive ablation studies. Experiments with 1M 192p training data, batch size = 40, and 30K iterations. We evaluate the results on the VBench benchmark.

| VBench                                                                  | total score | quality score | semantic score |
| ----------------------------------------------------------------------- | ----------- | ------------- | -------------- |
| InfinityStar (our model) — Attend to former clip's largest scale        | 81.28       | 81.56         | 80.16          |
| Ablation by removing/replacing core components                          |             |               |                |
| w/o Semantic Scale Repetition (SSR)                                     | 75.72       | 76.73         | 71.68          |
| w/o Spacetime Pyramid (using Pseudo-Spacetime)                          | 80.30       | 80.81         | 78.28          |
| w/o Stochastic Quantizer Depth (SQD)                                    | 81.07       | 81.21         | 80.54          |
| Comparison of different attention mechanism variants                    |             |               |                |
| Full Attention                                                          | 80.77       | 81.15         | 79.23          |
| Attend to former clip's 3rd largest scale                               | 80.86       | 81.26         | 79.26          |
| Attend to former clip's 6th largest scale                               | 80.75       | 80.98         | 79.80          |

Table 5: Computational efficiency comparison of attention mechanisms on a single GPU.

|                                         | (192p 65 frames) | (192p 161 frames) | (480p 161 frames) |
| --------------------------------------- | ---------------- | ----------------- | ----------------- |
| Full Attention                          | 8.6s / 40.8GB    | 24.3s / 57GB      | OOM               |
| Attend to former clip's largest scale   | 7.7s / 38.5GB    | 16.7s / 40GB      | 44.7s / 63GB      |
| Attend to former clip's 3rd largest     | 7.4s / 38.2GB    | 15.8s / 39GB      | 34.5s / 58GB      |
| Attend to former clip's 6th largest     | 7.3s / 37.9GB    | 15.2s / 38GB      | 30.5s / 55GB      |

Table 6: Computational efficiency comparison.

| Method    | Model        | # Parameters | Duration(s) | Frames | Resolution | Time(s) | Speedup |
| --------- | ------------ | ------------ | ----------- | ------ | ---------- | ------- | ------- |
| Diffusion | Wan 2.1 [28] | 14B          | 5           | 81     | 720p       | 1864    | 1       |
| AR        | Nova [9]     | 0.6B         | 5           | 81     | 720p       | 354     | 5       |
| AR        | InfinityStar | 8B           | 5           | 81     | 720p       | 58      | 32      |

Note: Baselines evaluated at different resolutions and model sizes; direct speed comparisons should be interpreted cautiously.


## Conclusion


By enabling temporal prediction within a purely discrete architecture, InfinityStar supports diverse generation tasks while maintaining both state-of-the-art quality and exceptional efficiency. Our evaluations indicate InfinityStar achieves competitive performance with leading diffusion-based approaches while significantly improving inference efficiency. Though not fully optimized or scaled to the largest models, InfinityStar makes important progress toward practical high-resolution video generation and provides a foundation for future work in this direction.

However, due to limited computational resources, we did not scale our model training and parameter size to the level of leading diffusion models, which bottlenecks current performance. Furthermore, our inference pipeline has not yet been fully optimized, indicating room for future improvement.


![Figure 49](image_49.jpg)
<!-- Caption: Conclusion, index 49 -->



## References


## References

[1] N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding, et al. Cosmos World: Foundation model platform for physical AI. arXiv preprint arXiv:2501.03575, 2025. 2, 3, 4

[2] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2

[3] T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman, C. Ng, R. Wang, and A. Ramesh. Video generation models as world simulators. OpenAI, 2024. 1, 2

[4] M. Byeon, B. Park, H. Kim, S. Lee, W. Baek, and S. Kim. COYO-700M: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022. 5

[5] H. Chen, M. Xia, Y. He, Y. Zhang, X. Cun, S. Yang, J. Xing, Y. Liu, Q. Chen, X. Wang, et al. VideoCrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 2, 8

[6] J. Chen, C. Ge, E. Xie, Y. Wu, L. Yao, X. Ren, Z. Wang, P. Luo, H. Lu, and Z. Li. PixArt-Sigma: Weak-to-strong training of diffusion transformer for 4K text-to-image generation. arXiv preprint arXiv:2403.04692, 2024. 7

[7] T.-S. Chen, A. Siarohin, W. Menapace, E. Deyneka, H.-w. Chao, B. E. Jeon, Y. Fang, H.-Y. Lee, J. Ren, M.-H. Yang, et al. Panda-70M: Captioning 70M videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024. 5

[8] Google DeepMind. Veo 3. https://deepmind.google/technologies/veo/veo-3/, 2025.05. 1

[9] H. Deng, T. Pan, H. Diao, Z. Luo, Y. Cui, H. Lu, S. Shan, Y. Qi, and X. Wang. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169, 2024. 1, 2, 8, 9, 10

[10] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Müller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 7

[11] A. Germanidis. Introducing Gen-3 Alpha: A new frontier for video generation. Runway, 2024. 8

[12] D. Ghosh, H. Hajishirzi, and L. Schmidt. GenEval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36, 2024. 6, 7

[13] J. Han, J. Liu, Y. Jiang, B. Yan, Y. Zhang, Z. Yuan, B. Peng, and X. Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. arXiv preprint arXiv:2412.04431, 2024. 1, 2, 3, 4, 5, 7

[14] X. Hu, R. Wang, Y. Fang, B. Fu, P. Cheng, and G. Yu. ELLA: Equip diffusion models with LLM for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024. 6, 7

[15] X. Ju, Y. Gao, Z. Zhang, Z. Yuan, X. Wang, A. Zeng, Y. Xiong, Q. Xu, and Y. Shan. MiraData: A large-scale video dataset with long durations and structured captions, 2024. 5

[16] D. Kondratyuk, L. Yu, X. Gu, J. Lezama, J. Huang, G. Schindler, R. Hornung, V. Birodkar, J. Yan, M.-C. Chiu, K. Somandepalli, H. Akbari, Y. Alon, Y. Cheng, J. Dillon, A. Gupta, M. Hahn, A. Hauth, D. Hendon, A. Martinez, D. Minnen, M. Sirotenko, K. Sohn, X. Yang, H. Adam, M.-H. Yang, I. Essa, H. Wang, D. A. Ross, B. Seybold, and L. Jiang. VideoPoet: A large language model for zero-shot video generation, 2024. 1, 2

[17] W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong, X. Li, B. Wu, J. Zhang, et al. HunyuanVideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 1, 2, 7, 8

[18] Kuaishou. Kling AI. https://klingai.kuaishou.com/, 2024.06. 1

[19] A. Kuznetsova, H. Rom, N. Alldrin, J. Uijlings, I. Krasin, J. Pont-Tuset, S. Kamali, S. Popov, M. Malloci, A. Kolesnikov, et al. The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale. International Journal of Computer Vision, 128(7):1956–1981, 2020. 5

[20] Black Forest Labs. FLUX. https://blackforestlabs.ai/announcing-black-forest-labs/, 2024. 7

[21] W. Peebles and S. Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023. 2

[22] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 7

[23] P. Sun, Y. Jiang, S. Chen, S. Zhang, B. Peng, P. Luo, and Z. Yuan. Autoregressive model beats diffusion: LLaMA for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 7

[24] Chameleon Team. Chameleon: Mixed-Modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 7

[25] K. Tian, Y. Jiang, Z. Yuan, B. Peng, and L. Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905, 2024. 1, 2

[26] A. van den Oord, O. Vinyals, et al. Neural discrete representation learning. Advances in Neural Information Processing Systems, 30, 2017. 4

[27] A. Voronov, D. Kuznedelev, M. Khoroshikh, V. Khrulkov, and D. Baranchuk. SWITTI: Designing scale-wise transformers for text-to-image synthesis. arXiv preprint arXiv:2412.01819, 2024. 5

[28] A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, J. Zeng, et al. WAN: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 2, 8, 9, 10

[29] J. Wang, Y. Jiang, Z. Yuan, B. Peng, Z. Wu, and Y.-G. Jiang. OmniTokenizer: A joint image-video tokenizer for visual generation. Advances in Neural Information Processing Systems, 37:28281–28295, 2024. 2, 3, 4

[30] X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu, et al. Emu3: Next-Token Prediction Is All You Need. *arXiv preprint arXiv:2409.18869*, 2024. 1, 2, 7, 8
[31] E. Xie, J. Chen, J. Chen, H. Cai, H. Tang, Y. Lin, Z. Zhang, M. Li, L. Zhu, Y. Lu, et al. Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers. *arXiv preprint arXiv:2410.10629*, 2024. 7
[32] J. Xie, W. Mao, Z. Bai, D. J. Zhang, W. Wang, K. Q. Lin, Y. Gu, Z. Chen, Z. Yang, and M. Z. Shou. Show-O: One Single Transformer to Unify Multimodal Understanding and Generation. *arXiv preprint arXiv:2408.12528*, 2024. 7
[33] W. Yan, Y. Zhang, P. Abbeel, and A. Srinivas. VideoGPT: Video Generation Using VQ-VAE and Transformers. *arXiv preprint arXiv:2104.10157*, 2021. 2
[34] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng, et al. CogVideoX: Text-to-Video Diffusion Models with an Expert Transformer. *arXiv preprint arXiv:2408.06072*, 2024. 2, 7, 8
[35] L. Yu, J. Lezama, N. B. Gundavarapu, L. Versari, K. Sohn, D. Minnen, Y. Cheng, A. Gupta, X. Gu, A. G. Hauptmann, et al. Language Model Beats Diffusion — Tokenizer Is Key to Visual Generation. *arXiv preprint arXiv:2310.05737*, 2023. 2, 3
[36] Y. Zeng, G. Wei, J. Zheng, J. Zou, Y. Wei, Y. Zhang, and H. Li. Make Pixels Dance: High-Dynamic Video Generation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 8850–8860, 2024. 2
[37] D. J. Zhang, J. Z. Wu, J.-W. Liu, R. Zhao, L. Ran, Y. Gu, D. Gao, and M. Z. Shou. Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation. *International Journal of Computer Vision*, pages 1–15, 2024. 8
[38] F. Zhang, S. Tian, Z. Huang, Y. Qiao, and Z. Liu. Evaluation Agent: Efficient and Promptable Evaluation Framework for Visual Generative Models. *arXiv preprint arXiv:2412.09645*, 2024. 2
[39] Y. Zhang, H. Yang, Y. Zhang, Y. Hu, F. Zhu, C. Lin, X. Mei, Y. Jiang, B. Peng, and Z. Yuan. Waver: Wave Your Way to Lifelike Video Generation. *arXiv preprint arXiv:2508.15761*, 2025. 1
[40] Y. Zhao, Y. Xiong, and P. Krähenbühl. Image and Video Tokenization with Binary Spherical Quantization. *arXiv preprint arXiv:2406.07548*, 2024. 4
[41] Z. Zheng, X. Peng, T. Yang, C. Shen, S. Li, H. Liu, Y. Zhou, T. Li, and Y. You. Open-Sora: Democratizing Efficient Video Production for All. *arXiv preprint arXiv:2412.20404*, 2024. 7, 8