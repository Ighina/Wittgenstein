
## Abstract


HyperET: Efficient Training in Hyperbolic Space for Multi-modal Large Language Models

Zelin Peng¹, Zhengqin Xu², Qingyang Liu¹, Xiaokang Yang¹, Wei Shen¹ ⁽ᴮ⁾

¹ MoE Key Lab of Artificial Intelligence, AI Institute, School of Computer Science, SJTU
² State Key Laboratory of Infrared Physics, Shanghai Institute of Technical Physics, CAS

{zelin.peng, fate311, narumimaria, xkyang, wei.shen}@sjtu.edu.cn

Abstract

Multi-modal large language models (MLLMs) have emerged as a transformative approach for aligning visual and textual understanding. They typically require extremely high computational resources (e.g., thousands of GPUs) for training to achieve cross-modal alignment at multi-granularity levels. We argue that a key source of this inefficiency lies in the vision encoders they widely equip with, e.g., CLIP and SAM, which lack the alignment with language at multi-granularity levels. To address this issue, in this paper, we leverage hyperbolic space, which inherently models hierarchical levels and thus provides a principled framework for bridging the granularity gap between visual and textual modalities at an arbitrary granularity level. Concretely, we propose an efficient training paradigm for MLLMs, dubbed as HyperET, which can optimize visual representations to align with their textual counterparts at an arbitrary granularity level through dynamic hyperbolic radius adjustment in hyperbolic space. HyperET employs learnable matrices with Möbius multiplication operations, implemented via three effective configurations: diagonal scaling matrices, block-diagonal matrices, and banded matrices, providing a flexible yet efficient parametrization strategy. Comprehensive experiments across select MLLM benchmarks show that HyperET can often improve pre-training and fine-tuning performance with roughly 1% additional parameters, though gains may vary across tasks and models.


## Introduction


1 Introduction

Thanks to advancements in pre-trained foundation models in both computer vision and natural language processing [60, 61, 58, 1, 29, 50, 14, 18, 50, 15], researchers have been inspired to explore the alignment of visual and language models, leading to the development of multi-modal large language models (MLLMs). This alignment is often achieved through adapters, e.g., Q-former [16]. As a result, MLLMs [2, 10, 16, 40, 49, 4, 9, 65] rapidly develop in recent years, demonstrating strong performance in tasks that require both visual and textual understanding, e.g., image captioning and visual question answering (VQA).

Although modern multimodal large language models (MLLMs), e.g., Qwen-VL series [5, 64, 4] and Intern-VL series [13, 12, 11], achieve cross-modal alignment across a wide range of MLLM tasks that inherently involve multi-granularity levels, their success heavily depends on extensive data scaling and massive computational resources. For example, InternVL [13] requires training on hundreds of millions of image-text pairs using up to 640 GPUs. Such a resource-intensive training scheme raises serious concerns about efficiency, reproducibility, and long-term sustainability in the MLLM community.

community, especially for researchers and institutions constrained by limited computational resources. To address these concerns, it is essential to identify the underlying cause of this inefficiency. We argue that a key factor lies in the vision encoders that are commonly used, e.g., CLIP [54], SAM [29], and DINOv2 [46]. These encoders are typically aligned with language at a single granularity level, e.g., either pixel-level or object-level, and are thus insufficient to deal with tasks required for alignment at different granularity levels. This mismatch in granularity during training significantly impedes the optimization process, leading to inefficient cross-modal alignment and increased reliance on large-scale computational resources.

To solve this issue, we propose to directly quantify the granularity levels by leveraging hyperbolic space [7]. Empirical observations in prior works demonstrate that visual representations at different hierarchical levels (e.g., image-level and object-level) are naturally stratified in hyperbolic space [20, 52, 47]. This property enables the use of hyperbolic radius [57]—defined as the distance from a point to the origin in hyperbolic space—to quantify granularity levels [47]. Specifically, points in hyperbolic space closer to the origin (smaller radius) encode low-level visual features (e.g., pixel-level information), while points near the boundary (larger radius) represent high-level visual semantics (e.g., image-level concepts). This hierarchical property suggests that adjusting the hyperbolic radius of visual representations can effectively align them with language models at arbitrary granularity levels.

Building on this insight, we propose an efficient training paradigm (HyperET) for MLLMs that is capable of optimizing visual representations to align with their textual counterparts at arbitrary granularity levels. This is achieved through learnable matrices equipped with Möbius multiplication operations [62], which enable direct and continuous adjustment of the hyperbolic radius of visual representations, thereby facilitating cross-modal alignment. In practice, we introduce three parameter-efficient forms of the learnable matrices for adjusting the hyperbolic radius: (1) diagonal scaling matrices, (2) block-diagonal scaling matrices, and (3) banded scaling matrices. These designs significantly reduce the number of trainable parameters while retaining the capacity to address granularity mismatch. To further enhance parameter flexibility, the matrices can be extended to a dense version with fully populated learnable elements. This expansion increases the expressive capacity of our proposed training paradigm, facilitating more effective alignment across diverse cross-modal scenarios.

To evaluate the generalization capability and effectiveness of HyperET, we conduct extensive experiments across multiple pre-trained MLLM benchmarks and various downstream MLLM tasks, e.g., ScienceQA [42]. Results demonstrate that the proposed HyperET can be easily plug-and-play and consistently improve various MLLMs, including LLaVA-1.5 [39] and LLaVA-Next [32] for pre-training, as well as MemVP [26] and LaVIN [43] for fine-tuning on downstream tasks. Notably, HyperET introduces less than 1% of the total trainable parameters, ensuring high parameter efficiency.


## Related Work


## 2 Related Work

### 2.1 Multi-modal Large Language Models

Multi-modal large language models (MLLMs) [31, 10, 33, 38, 2, 55, 36, 70] have made significant breakthroughs in recent advancements, aiming to equip large language models [73, 1, 60, 61, 3] with the capability to process and interpret visual information. Most MLLMs achieve this goal by integrating CLIP’s vision encoder [69, 54] into pre-trained large language models through adapters, e.g., MLPs [40, 39], Q-Former [16, 35], and attention mechanisms [2]. Despite its effectiveness, this straightforward connection between the visual and language modalities still fails to align pre-trained models from different modalities, thereby resulting in inferior performance, e.g., hallucinations, across various downstream tasks.

### 2.2 Towards Alignment in MLLM

This failure mainly stems from the fact that CLIP’s vision encoder [50] is designed for standard classification tasks and is not equipped to handle the more fine-grained visual understanding tasks required by the language modality [59, 51, 27]. To better align the vision and language modalities in MLLMs, recent works explore parameter-efficient fine-tuning methods [43, 56, 71, 26, 44]. For example, LaVIN [43] introduces adapters in both the vision encoder and LLaMA [60] to achieve better alignment of the modalities. Another line of research [66, 59, 27] seeks to overcome this

bottleneck by incorporating additional vision models, such as DINOv2 [46], to construct a more powerful and capable vision branch. Despite the advancements in the field, few studies focus on understanding the changes in the representation of vision encoders that enable MLLMs to tackle more complex vision tasks. In this work, we aim to bridge this gap by leveraging hyperbolic space to directly model the granularity levels and adapt the granularity of visual representations to an appropriate level.

## 2.3 Learning in Hyperbolic Space

Unlike Euclidean space, hyperbolic space can be viewed as the continuous analog of a tree [6], making it inherently suitable for capturing hierarchical levels among various data types. Since visual and textual concepts are inherently hierarchical, recent works [17, 30, 52, 47, 48] show that hyperbolic space serves as a promising manifold for preserving granularity levels in vision-language model representations, leading to strong performance across downstream tasks. Existing hyperbolic space methods often manipulate hierarchical embeddings or impose constraints to improve modality alignment [incorrect ref]. However, these approaches typically discard some natural structural cues or rely heavily on task-specific tuning. Our method builds on these by preserving global hierarchy and applying hyperbolic radius adjustment, although the distinction from existing approaches remains subtle. This approach provides a complementary perspective to existing efforts in the MLLM community.


## Preliminaries


## 3 Preliminary Concepts

Hyperbolic Geometry. In contrast to Euclidean or spherical geometries, hyperbolic geometry is characterized by a constant negative curvature, which fundamentally distinguishes its geometric properties and computational behaviors. Following prior studies [8, 57], we utilize the classical Poincaré ball model—one of five principal analytic models for constructing hyperbolic space [7]—due to its demonstrated efficacy in representing hierarchical levels [23, 19, 45]. The Poincaré ball model $$(\mathbb{D}_c^n, g^{D_c})$$, characterized by a radius of $$1/\sqrt{c}$$ and constant negative curvature $$-c$$ $$(c > 0)$$, is formally defined as follows:
$$\begin{cases}
\mathbb{D}_c^n := \{\mathbf{X} \in \mathbb{R}^n : c\|\mathbf{X}\|^2 < 1\} \\
g^{D_c} := \lambda_{c,\mathbf{X}}^2 g^E
\end{cases} \quad (1)$$
where $$\lambda_{c,\mathbf{X}} = \frac{2}{1 - c\|\mathbf{X}\|^2}$$ and $$g^E = \mathbf{I}_n$$ denotes the Euclidean metric tensor, serving as the foundation for the hyperbolic space construction.

Hyperbolicity. Building upon the theoretical framework of gyrovector spaces [62, 63], we incorporate Möbius operations into hyperbolic space, specifically the Möbius addition operation "$$\oplus_c$$" and Möbius multiplication operation "$$\otimes_c$$". In hyperbolic geometry, the tangent space $$T_{\mathbf{X}}\mathbb{D}_c^n$$ at any point $$\mathbf{X} \in \mathbb{D}_c^n$$ serves as a first-order approximation of $$\mathbb{D}_c^n$$, representing an $$n$$-dimensional Euclidean space that locally approximates the hyperbolic structure. The tangent space $$T_{\mathbf{X}}\mathbb{D}_c^n$$ and $$\mathbb{D}_c^n$$ are mapped to each other by exponential $$(T_{\mathbf{X}}\mathbb{D}_c^n \mapsto \mathbb{D}_c^n : \exp_{\mathbf{X}}^{D_c}(\cdot))$$ and logarithmic $$(\mathbb{D}_c^n \mapsto T_{\mathbf{X}}\mathbb{D}_c^n : \log_{\mathbf{X}}^{D_c}(\cdot))$$ maps, respectively. Detailed mathematical definitions and derivations are provided in the supplementary material.


## Method


## 4 Methodology

This section first provides the necessary background on the conventional training paradigm from the perspective of parameter space tuning (Sec. 4.1) to contextualize our approach, followed by the detailed presentation of HyperET in Sec. 4.2. Then, Sec. 4.3 provides a theoretical analysis on adjusting the hyperbolic radius of visual representations.

### 4.1 Parameter Space Tuning

Parameter space tuning in multi-modal large language models (MLLMs), whether through full fine-tuning or parameter-efficient fine-tuning, aims to adapt pre-trained visual and language models to target multi-modal scenarios. However, these tuning methods, which rely solely on gradient updates, operate as constraint-free adjustments in Euclidean space. They often implicitly assume that visual representations can sufficiently adapt to the required granularity level (e.g., transitioning

Figure 1: The schematic representation of HyperET. In HyperET, we efficiently train MLLMs in hyperbolic space by adjusting the hyperbolic radius using a tunable scaling matrix $\mathbf{W}_s$. Here, $\mathbf{W}_s$ can be configured into three parameter-efficient variants, i.e., Diagonal, Block-Diagonal and Banded.

From image-level to pixel-level, which may lead to inefficiency in alignment at a certain granularity level. In contrast, the core technique of our proposed HyperET involves hyperbolic radius adjustment, which explicitly adjusts the granularity level of visual representations in MLLMs. HyperET provides a simple yet effective solution to granularity mismatch challenges.

## 4.2 General Hyperbolic Radius Adjustment

As previously discussed, hyperbolic radius adjustment provides a direct mechanism for optimizing the granularity level of visual representations, effectively bridging the granularity gap between visual and language modalities. In practice, we introduce a radius adjustment constraint into the weight update process, generally defined as follows:

$$\text{Rad}_{\mathbf{W}^{\mathbb{D}}}/\text{Rad}_{\mathbf{W}^{\mathbb{D}}_0} = s \quad \Leftrightarrow \quad \text{Rad}_{\mathbf{W}^{\mathbb{D}}} = s \cdot \text{Rad}_{\mathbf{W}^{\mathbb{D}}_0},$$
(2)

where $\text{Rad}_{\mathbf{W}^{\mathbb{D}}}$ and $\text{Rad}_{\mathbf{W}^{\mathbb{D}}_0}$ represent the hyperbolic radii of $\mathbf{W}^{\mathbb{D}}$ and $\mathbf{W}^{\mathbb{D}}_0$, respectively, quantifying their granularity levels. The hyperbolic weight matrices $\mathbf{W}^{\mathbb{D}}$ and $\mathbf{W}^{\mathbb{D}}_0$ are derived by projecting their Euclidean counterparts $\mathbf{W}$ and $\mathbf{W}_0$ into hyperbolic space through exponential mapping operations defined in Sec. 3. The scaling coefficient $s$ is task-adaptive, dynamically adjusting to optimize performance. However, this modification requires a two-step procedure: (1) computing the hyperbolic radius of $\mathbf{W}^{\mathbb{D}}$ and (2) subsequently adjusting it. To streamline this process, we propose a more efficient approach that directly optimizes $\mathbf{W}^{\mathbb{D}}$ without intermediate radius computations.

Hyperbolic Radius. Without loss of generality, for any point $\mathbf{X} \in \mathbb{D}^n_c$ in hyperbolic space, its hyperbolic radius is formally defined as follows:

$$\text{Rad}_{\mathbf{X}} := d^{\mathbb{D}}_c(\mathbf{X}, \mathbf{0}) = \left(\frac{2}{\sqrt{c}}\right)\tanh^{-1}(\sqrt{c}\|\mathbf{X}\|),$$
(3)

where $\mathbf{0}$ denotes the origin point in hyperbolic space. In the subsequent analysis, we demonstrate that Möbius multiplication operations $\otimes_c$ defined in Sec. 3 can enable precise control over the hyperbolic radius. This critical property is formally stated in the following theorems:

Theorem 1 (Hyperbolic Radius Scaling) For a point $\mathbf{X} \in \mathbb{D}^n_c$ in hyperbolic space, the hyperbolic radius adjustment function is expanded as follows:

$$s \cdot \text{Rad}_{\mathbf{X}} = \frac{2}{\sqrt{c}}\left(s\frac{\sqrt{c}}{2}\text{Rad}_{\mathbf{X}}\right)$$
$$= \frac{2}{\sqrt{c}}\tanh^{-1}\!\left(\sqrt{c}\,\|s \otimes_c \mathbf{X}\|\right)$$
$$= \text{Rad}_{s \otimes_c \mathbf{X}},$$
(4)

where $\otimes_c$ here is instantiated as a Möbius scalar multiplication operation. Therefore, hyperbolic radius adjustment can be precisely controlled through $\otimes_c$ between $s$ and $\mathbf{X}$, with the scaling coefficient $s$ serving as a primary learnable parameter.


![Figure 13](image_13.jpg)
<!-- Caption: Method, index 13 -->


According to Theorem 1, ⊗_c can provide an equivalent mechanism during parameter space tuning. Consequently, hyperbolic radius adjustment in Eq. (2) can be easily achieved as follows:
$$\mathbf{W}^{D} = s \otimes_c \mathbf{W}_0^{D}.$$ (5)
Then, building upon the exponential mapping $$\exp_0^{D^n,c}(\cdot)$$ and the logarithmic mapping $$\log_0^{D^n,c}(\cdot)$$ defined in Sec. 3, we achieve general hyperbolic radius adjustment via the following reformulation of Eq. (5):
$$\mathbf{W} = \log_0^{D^n,c}(s \otimes_c \exp_0^{D^n,c}(\mathbf{W}_0)),$$ (6)
where s is a learnable parameter for adjustment. Given that a constrained parameter set—particularly when limited to a single learnable parameter—may ineffectively adjust hyperbolic radius during restricted training iterations, we propose a more flexible parameterization strategy through matrix-based formulations, termed flexible adjustment, which further enables precise control over hyperbolic radius optimization.

Flexible Adjustment for Hyperbolic Radius. To achieve this, we adopt a scaling matrix $$\mathbf{W}_s$$ to replace the scaling coefficient s, which is satisfied: $$s = \|\mathbf{W}_s \mathbf{W}_0^{D}\|/\|\mathbf{W}_0^{D}\|$$. Consequently, the right-hand side of Eq. (2) can be reformulated as follows:
$$\text{Rad}_{\mathbf{W}^{D}} = s \cdot \text{Rad}_{\mathbf{W}_0^{D}}$$
$$= \frac{\|\mathbf{W}_s \mathbf{W}_0^{D}\|}{\|\mathbf{W}_0^{D}\|} \cdot \text{Rad}_{\mathbf{W}_0^{D}}.$$ (7)
In this framework, the learnable parameters transition from scalar values to matrix-based formulations, significantly enhancing flexibility within constrained training iterations. The following theorem demonstrates that hyperbolic radius can be dynamically scaled through Möbius matrix multiplication operations, a generalized instantiation of $$\otimes_c$$.
Theorem 2 (Hyperbolic Radius Flexibility Scaling) For a point $$\mathbf{X} \in \mathbb{D}_c^n$$ in hyperbolic space and a scaling matrix $$\mathbf{X}_s \in \mathbb{R}^{n \times n}$$, the flexible radius adjustment function is formally defined through Eq. (7) and Möbius matrix multiplication operations as follows:
$$\frac{\|\mathbf{X}_s \mathbf{X}\|}{\|\mathbf{X}\|} \cdot \text{Rad}_{\mathbf{X}}
= \frac{2}{\sqrt{c}}\!\left(\frac{\|\mathbf{X}_s \mathbf{X}\|}{\|\mathbf{X}\|}\frac{\sqrt{c}}{2}\text{Rad}_{\mathbf{X}}\right)$$
$$= \frac{2}{\sqrt{c}} \tanh^{-1}\!\big(\sqrt{c}\,\|\mathbf{X}_s \otimes_c \mathbf{X}\|\big)$$
$$= \text{Rad}_{\mathbf{X}_s \otimes_c \mathbf{X}}.$$ (8)
Analogically, the scaling matrix $$\mathbf{W}_s$$ is able to directly adjust the hyperbolic radius $$\text{Rad}_{\mathbf{X}}$$ through Möbius matrix multiplication operations, providing precise control over representation granularity. ■
Building upon Theorem 2, we introduce a matrix-based formulation by replacing the scaling coefficient s with $$\mathbf{W}_s$$, resulting in the following reformulation of Eq. (6):
$$\mathbf{W} = \log_0^{D^n,c}\!\big(\mathbf{W}_s \otimes_c \exp_0^{D^n,c}(\mathbf{W}_0)\big).$$ (9)
With up to $$O(n^2)$$ learnable elements, $$\mathbf{W}_s$$ in Eq. (9) offers significantly enhanced flexibility for hyperbolic radius adjustment. Notably, Eqs. (5) and (9) become equivalent when $$\mathbf{W}_s$$ is constrained to a diagonal matrix with uniform scaling factors, i.e., $$\mathbf{W}_s = \text{diag}(\omega_1, \omega_2, \cdots, \omega_n) \in \mathbb{R}^{n \times n}$$ and $$\omega_i = \omega_j, i \neq j$$. Moreover, in alignment with the current paradigm of parameter-efficient fine-tuning, we also introduce a parameter-efficient variant of $$\mathbf{W}_s$$, designed to maintain flexibility while reducing computational overhead.

Efficient Adjustment for Hyperbolic Radius. To realize this strategy, we define a diagonal scaling matrix $$\mathbf{W}_s^D = \text{diag}(\omega_1, \omega_2, \ldots, \omega_n) \in \mathbb{R}^{n \times n}$$, where each $$\omega_i$$ is a learnable scalar and $$\omega_i \neq \omega_j$$ when $$i \neq j$$, significantly reducing the number of learnable parameters. In this form, only the diagonal elements are learnable parameters, making the diagonal scaling matrix $$\mathbf{W}_s^D$$ the most parameter-efficient configuration for hyperbolic radius adjustment. Building upon this foundation, we extend $$\mathbf{W}_s^D$$ into two more flexible variants by incrementally introducing learnable off-diagonal elements, balancing enhanced adjustment capability with parameter efficiency: (1) Block-diagonal scaling matrix $$\mathbf{W}_s^{B\text{-}D} = \text{diag}(\mathbf{R}_1, \mathbf{R}_2, \ldots, \mathbf{R}_r)$$, where $$\mathbf{R}_i \in \mathbb{R}^{\frac{n}{r} \times \frac{n}{r}}$$. Here, r is the block size (we assume

n is divisible by r). This formulation allows intra-block interactions while maintaining sparsity across blocks; (2) Banded scaling matrix $\mathbf{W}_s^B = \begin{bmatrix} \omega_{11} & \cdots & \omega_{1n} \\ \vdots & \ddots & \vdots \\ \omega_{n1} & \cdots & \omega_{nn} \end{bmatrix} \in \mathbb{R}^{n \times n}$, where $\omega_{ij} = 0$ for all $i, j$ such that $|i - j| > d$. Here, $d$ denotes the bandwidth, indicating that nonzero elements are allowed within $d$ entries above and below the main diagonal. These two variants provide mechanisms to capture localized interactions while preserving parameter efficiency. Notably, both $\mathbf{W}_s^{B\text{-}D}$ and $\mathbf{W}_s^B$ degenerate to $\mathbf{W}_s^D$ under specific configurations: when $r = n$ for the block-diagonal matrix and $d = 0$ for the banded matrix. This highlights the inherent hierarchical flexibility enabled by our parametrization strategy and reflects an insightful way of designing fine-tuning methods.

To summarize these fine-tuning strategies, Fig. 1 provides a unified illustration of different variants of the scaling matrix $\mathbf{W}_s$. Our introduced hyperbolic radius adjustment, i.e., HRA, adjusts visual representations by applying learnable scaling matrices to the frozen pre-trained weights $\mathbf{W}_0$ in hyperbolic space, enabling precise adjustment over arbitrary granularity levels.

## 4.3 Theoretical Analysis

To demonstrate the impact of hyperbolic radius adjustment, we provide a theoretical analysis of the Möbius multiplication operation. The following deduction shows that the proposed Möbius multiplication operation can directly adjust the granularity level of visual representations.

Deduction. For $\mathbf{Y}_0 = \mathbf{W}_0 \mathbf{X}$, where $\mathbf{X} \in \mathbb{R}^{d \times k}$ is the input embedding, $\mathbf{W}_0$ is the pre-trained weight and $\mathbf{Y}_0$ is the pre-trained visual representation, the forward pass of HyperET is as follows:
$$\mathbf{Y} = \mathbf{W}\mathbf{X} = \log_0^{\mathbb{D}, c}\!\big(\mathbf{W}_s \otimes_c \exp_0^{\mathbb{D}, c}(\mathbf{W}_0)\big)\mathbf{X}. \tag{10}$$
Then, according to the definition of hyperbolic space, $\mathbf{Y}_0$ is firstly projected into the hyperbolic space $\mathbb{D}_c^n$ and our HyperET is applied to adjust the hyperbolic radius, resulting in the new visual representation $\mathbf{Y}$, expressed as:
$$\mathbf{Y}_0^{\mathbb{D}_c^n} = \exp_0^{\mathbb{D}, c}(\mathbf{Y}_0) = \mathbf{W}_0^{\mathbb{D}_c^n}\mathbf{X}^{\mathbb{D}_c^n}, \tag{11}$$
$$\mathbf{Y}^{\mathbb{D}_c^n} = \exp_0^{\mathbb{D}, c}(\mathbf{Y}) = \mathbf{W}_s \otimes_c \mathbf{W}_0^{\mathbb{D}_c^n}\mathbf{X}^{\mathbb{D}_c^n}, \tag{12}$$
where $\mathbf{W}_0^{\mathbb{D}_c^n} = \exp_0^{\mathbb{D}, c}(\mathbf{W}_0)$ and $\mathbf{X}^{\mathbb{D}_c^n} = \exp_0^{\mathbb{D}, c}(\mathbf{X})$. According to Theorem 2, the hyperbolic radius of $\mathbf{Y}^{\mathbb{D}_c^n}$ is obtained as:
$$\mathrm{Rad}_{\mathbf{Y}} = \mathrm{Rad}_{\mathbf{W}_s \otimes_c \mathbf{W}_0^{\mathbb{D}_c^n}\mathbf{X}^{\mathbb{D}_c^n}}$$
$$= \frac{\|\mathbf{W}_s \mathbf{W}_0^{\mathbb{D}_c^n}\mathbf{X}^{\mathbb{D}_c^n}\|}{\|\mathbf{W}_0^{\mathbb{D}_c^n}\mathbf{X}^{\mathbb{D}_c^n}\|} \cdot \mathrm{Rad}_{\mathbf{W}_0^{\mathbb{D}_c^n}\mathbf{X}^{\mathbb{D}_c^n}}$$
$$= \frac{\|\mathbf{W}_s \mathbf{W}_0^{\mathbb{D}_c^n}\mathbf{X}^{\mathbb{D}_c^n}\|}{\|\mathbf{W}_0^{\mathbb{D}_c^n}\mathbf{X}^{\mathbb{D}_c^n}\|} \mathrm{Rad}_{\mathbf{Y}_0}$$
$$= s \cdot \mathrm{Rad}_{\mathbf{Y}_0}, \tag{13}$$
where $s$ is a scaling coefficient. Therefore, our hyperbolic radius adjustment method can directly adjust the hyperbolic radius of visual representations, thus is able to bridge the granularity gap between visual and textual modalities at an arbitrary granularity level.


## Experiments


# 5 Experiments

Our experimental evaluation encompasses two MLLM scenarios, (1) MLLM’s fine-tuning (Sec. 5.1), and (2) MLLM’s pre-training (Sec. 5.2). A detailed ablation study is presented in Sec. 5.3 and 5.4.

## 5.1 MLLM’s Fine-tuning

Experimental Setting. We evaluate our method on ScienceQA [42], a challenging large-scale VQA benchmark encompassing diverse scientific domains. Our comparative analysis includes MemVP and other LLaMA-based models with input-space visual prompting: LLaVA [40], and LaVIN [43]. We

Table 1: Comparison with SOTA fine-tuning methods on the ScienceQA test set [42]. Question categories: NAT = natural science, SOC = social science, LAN = language science, TXT = w/ text context, IMG = w/ image context, NO = no context, G1-6 = grades 1–6, G7-12 = grades 7–12. “Ours”: we realize the extra learnable parameters as diagonal matrices, i.e., $\mathbf{W}_s^D$. Vision encoder: CLIP.

| Method                            | #TrainableParams | LanguageModel | Subject<br/>NAT | Subject<br/>SOC | Subject<br/>LAN | Context Modality<br/>TXT | Context Modality<br/>IMG | Context Modality<br/>NO | Grade<br/>G1-6 | Grade<br/>G7-12 | Average |
| --------------------------------- | ---------------- | ------------- | --------------- | --------------- | --------------- | ------------------------ | ------------------------ | ----------------------- | -------------- | --------------- | ------- |
| Human                             | -                | -             | 90.23           | 84.97           | 87.48           | 89.60                    | 87.50                    | 88.10                   | 91.59          | 82.42           | 88.40  |
| *Fully Fine-Tuning*               |                  |               |                 |                 |                 |                          |                          |                         |                |                 |         |
| LLaVA                             | 13B              | Vicuna-13B    | 90.36           | **95.95**       | 88.00           | 89.49                    | 88.00                    | 90.66                   | 90.93          | **90.90**       | 90.92  |
| *Parameter-efficient Fine-Tuning* |                  |               |                 |                 |                 |                          |                          |                         |                |                 |         |
| LaVIN                             | 3.8M             | LLaMA-7B      | 89.25           | 94.94           | 85.24           | 88.51                    | 87.46                    | 88.08                   | 90.16          | 88.07           | 89.41  |
| LaVIN+Ours                        | 3.85M (+0.05M)   | LLaMA-7B      | 89.35           | **96.06**       | **86.54**       | 88.29                    | **88.01**                | **89.33**               | 91.36          | 87.65           | **90.03** (+0.62) |
| MemVP                             | 3.9M             | LLaMA-7B      | 94.45           | 95.05           | 88.64           | 93.99                    | 92.36                    | 90.94                   | 93.10          | 93.01           | 93.07  |
| MemVP+Ours                        | 3.95M (+0.05M)   | LLaMA-7B      | **94.85**       | 95.05           | **90.55**       | **94.57**                | 92.91                    | 92.20                   | **93.65**      | **94.00**       | **93.78** (+0.71) |
| LaVIN                             | 5.4M             | LLaMA-13B     | 90.32           | 94.38           | 87.73           | 89.44                    | 87.65                    | 90.31                   | 91.19          | 89.26           | 90.50  |
| LaVIN+Ours                        | 5.45M (+0.05M)   | LLaMA-13B     | 90.57           | **95.63**       | **89.89**       | 89.61                    | 88.75                    | **92.02**               | 91.95          | 90.58           | **91.46** (+0.96) |
| MemVP                             | 5.5M             | LLaMA-13B     | 95.07           | 95.15           | 90.00           | 94.43                    | 92.86                    | 92.47                   | 93.61          | 94.07           | 93.78  |
| MemVP+Ours                        | 5.55M (+0.05M)   | LLaMA-13B     | **96.19**       | 95.78           | 90.86           | **95.51**                | **94.25**                | 93.18                   | **94.88**      | **94.44**       | **94.72** (+0.94) |

We follow the experimental setting in [43]. All models utilize a CLIP pre-trained ViT-L/14 visual encoder. The weights of HyperET are implemented with the three scaling matrices and integrated into the model's attention layers. However, explicit hyperparameters including learning rate, weight decay, batch sizes, training steps, and adapter insertion points are omitted. The sole curvature parameter $c=0.01$ is noted. Training hardware comprises up to 8 NVIDIA H800 GPUs.

Comparing to SOTA. Our experimental evaluation compares the proposed approach with state-of-the-art parameter-efficient fine-tuning (PEFT) methods, including LaVIN [43] and MemVP [26]. As shown in Table 1, baseline MemVP generally outperforms HyperET-enhanced LaVIN by a small margin, with inconsistent improvements noted. Some gains with higher-parameter configurations appear marginal or within variance. Furthermore, MemVP-LLaMA-7B combined with HyperET fails to reach the top accuracy of MemVP-LLaMA-13B, contradicting claims of performance parity despite lower parameter counts. These findings suggest that HyperET’s benefits may be dataset- and architecture-specific rather than broadly generalizable.

\n\n\n\n
## 5.2 MLLM’s Pre-training

Experimental Setting. Our pre-training evaluation framework builds upon LLaVA-1.5 [39], employing identical datasets to evaluate HyperET’s effectiveness in MLLM pre-training. Our comparative analysis includes LLaVA-1.5 [39] and LLaVA-Next [39]. However, HyperET is trained with an extended schedule and augmented compute budget not matched in the baseline settings. Such discrepancies may contribute significantly to improved results, complicating direct attribution to HyperET alone. Detailed training regimes and hardware configurations are omitted, limiting reproducibility.

Comparing to SOTA. Our experimental framework evaluates the proposed method against state-of-the-art pre-trained MLLMs across 12 standard visual language benchmarks. We implement HyperET using the most flexible matrix configuration, i.e., $\mathbf{W}_s$, maximizing the model’s adaptability. While this approach resembles full fine-tuning in structure, the actual parameter count remains remarkably efficient at approximately 50M—less than 1% of the language model’s 13B parameters—maintaining parameter efficiency. As shown in Table 2, our approach demonstrates significant performance improvements over LLaVA-1.5 [39], particularly in mitigating the limitations of CLIP-based encoders. Notably, on the POPE benchmark [37] for object hallucination detection, HyperET substantially reduces visual hallucinations, providing empirical evidence that hyperbolic radius optimization effectively enhances cross-modal alignment at an arbitrary level.

Table 2: Comparison with SoTA pre-trained methods on 12 MLLM benchmarks, including VQAv2 [21], GQA [25], VW: VizWiz [22], SQA: ScienceQA-IMG [42], TVQA: TextVQA [53], PE: POPE [37], ME: MME [67], MB: MMBench [41], MB<sup>CN</sup>: MMBench-Chinese [41], SD: SEED-Bench [34], LVA<sup>W</sup>: LLaVA-Bench (In-the-Wild) [40] and M-Vet [68]. Top-1 accuracy is reported (Best in bold, second best is underlined). Lan. Model: Language model. Benchmark names are abbreviated due to space limits. "Ours": we here realize the extra learnable parameters as full matrices, i.e., W<sub>s</sub>. Vision encoder: CLIP.

| Method          | Lan. Model | VQAv2 | GQA  | VW   | SQA  | TVQA | PE   | ME     | MB   | MB<sup>CN</sup> | SD   | LVA<sup>W</sup> | M-Vet |
| --------------- | ---------- | ----- | ---- | ---- | ---- | ---- | ---- | ------ | ---- | ---------------- | ---- | --------------- | ----- |
| LLaVA-1.5       | Vicuna-7B  | 78.5  | 62.0 | 50.0 | 66.8 | 58.2 | 85.9 | 1510.7 | 64.3 | 58.3             | 58.6 | 63.4            | 30.5  |
| LLaVA-1.5+Ours  | Vicuna-7B  | 80.3  | 63.7 | 51.9 | 69.1 | 60.8 | 87.7 | 1536.2 | 66.8 | 60.5             | 60.2 | 65.6            | 32.4  |
| LLaVA-1.5       | Vicuna-13B | 80.0  | 63.3 | 53.6 | 71.6 | 61.3 | 85.9 | 1531.3 | 67.7 | 63.6             | 61.6 | 70.7            | 35.4  |
| LLaVA-1.5+Ours  | Vicuna-13B | 82.3  | 65.7 | 55.2 | 73.7 | 63.9 | 88.7 | 1584.7 | 69.8 | 65.2             | 63.4 | 72.6            | 38.3  |
| LLaVA-Next      | Vicuna-7B  | 81.8  | 64.2 | 57.6 | 70.1 | 64.9 | 86.5 | 1519   | 67.4 | 60.6             | 70.2 | 81.6            | 43.9  |
| LLaVA-Next+Ours | Vicuna-7B  | 82.9  | 65.4 | 58.9 | 70.8 | 65.1 | 88.9 | 1551   | 69.9 | 62.5             | 71.0 | 82.9            | 44.8  |

Table 3: Comparative analysis of fine-tuning spaces and flexibility levels on ScienceQA test set [42]. All experiments utilize MemVP [26] with LLaMA-13B as the backbone language model. The notation is defined as follows: W_s^D represents diagonal scaling matrices, W_s^{B-D} denotes block-diagonal scaling matrices, W_s^B indicates banded scaling matrices, and W_{se} corresponds to Euclidean space fine-tuning matrices. Key parameters include d for banded size and n/r for block size. ⊗_c: Möbius matrix multiplication.

| Method                                   | #Trainable Params (M) | d | n/r | ⊗_c | Average        |
| ---------------------------------------- | --------------------- | - | --- | --- | -------------- |
| MemVP                                    | 5.5                   | - | -   | -   | 93.78          |
| Efficient training                       |                       |   |     |     |                |
| +WD_se                                   | 5.55 (+0.05)          | 0 | 1   | -   | 93.81 (+0.03)  |
| +WB−D_se                                 | 5.64 (+0.14)          | - | 2   | -   | 93.70 (−0.08)  |
| +WB_se                                   | 5.71 (+0.21)          | 1 | -   | -   | 93.65 (−0.13)  |
| Efficient training in hyperbolic space   |                       |   |     |     |                |
| +WD_s (no ⊗_c)                           | 5.55 (+0.05)          | 0 | 1   | ✗   | 93.91          |
| +WD_s                                    | 5.55 (+0.05)          | 0 | 1   | ✓   | 94.72 (+0.94)  |
| +WB−D_s                                  | 5.64 (+0.14)          | - | 2   | ✓   | 94.79 (+1.01)  |
|                                          | 5.78 (+0.28)          | - | 4   | ✓   | 94.84          |
|                                          | 6.08 (+0.58)          | - | 8   | ✓   | 94.82          |
| +WB_s                                    | 5.71 (+0.21)          | 1 | -   | ✓   | 94.89 (+1.11)  |
|                                          | 5.86 (+0.36)          | 2 | -   | ✓   | 94.82          |
|                                          | 6.15 (+0.65)          | 4 | -   | ✓   | 94.83          |

Table 4: Ablation studies of HyperET across vision encoders with varying granularity levels on ScienceQA test set.

| Method                                   | Lang. Model | Vision Encoder | Average        |
| ---------------------------------------- | ----------- | -------------- | -------------- |
| MemVP                                    | LLaMA-13B   | DINOv2         | 92.05          |
| MemVP                                    | LLaMA-13B   | SAM            | 91.85          |
| Efficient training                       |             |                |                |
| +WD_se                                   | LLaMA-13B   | DINOv2         | 92.75 (+0.70)  |
| +WD_se                                   | LLaMA-13B   | SAM            | 92.67 (+0.82)  |
| Efficient training in hyperbolic space   |             |                |                |
| +WD_s                                    | LLaMA-13B   | DINOv2         | 94.12 (+2.07)  |
| +WD_s                                    | LLaMA-13B   | SAM            | 94.21 (+2.36)  |

Table 5: Ablation study on the key components of HyperET on selected five MLLM benchmarks. We here realize the extra learnable parameters as full matrices, i.e., W_s. ⊗_c: Möbius matrix multiplication. W_{se} corresponds to Euclidean space fine-tuning matrices with the same number of parameters.

| Method                                   | VQAv2 | GQA  | VW   | SQA  | TVQA |
| ---------------------------------------- | ----- | ---- | ---- | ---- | ---- |
| Baseline                                 | 80.0  | 63.3 | 53.6 | 71.6 | 61.3 |
| Efficient training                       |       |      |      |      |      |
| +W_se                                    | 80.8  | 63.8 | 53.8 | 71.7 | 61.8 |
| Efficient training in hyperbolic space   |       |      |      |      |      |
| +W_s                                     | 82.3  | 65.7 | 55.2 | 73.7 | 63.9 |
| −⊗_c                                     | 81.1  | 64.0 | 53.9 | 71.9 | 62.1 |

5.3 Ablation Study on Fine-tuning

Here, we do an ablation study on ScienceQA [42], employing MemVP [26] as the backbone.

Introducing Different Flexibility into HyperET. We systematically investigate our proposed three distinct parameterization strategies for HyperET’s learnable components. These strategies include diagonal scaling matrices W_s^D and their two extended variants: banded scaling matrices W_s^B and block-diagonal scaling matrices W_s^{B-D}, which systematically increase parameter flexibility through progressively relaxed structural constraints. As shown in Table 3, increasing flexibility via W_s^B fine-tuning demonstrates inconsistent improvements, with accuracy fluctuating and dropping under larger bandwidths, suggesting possible overfitting or diminishing returns. These results highlight a trade-off between parameter efficiency and model stability. These findings demonstrate that our fine-tuning method effectively accomplishes two key objectives: (1) efficient adaptation to optimal hyperbolic radii and (2) robust cross-modal alignment for downstream tasks, all while maintaining parameter efficiency.

Effectiveness of Training in Hyperbolic Space. To isolate the performance gains attributable to hyperbolic space rather than increases in parameter count, we conduct controlled experiments by adding an equivalent number of parameters through matrix multiplication in Euclidean space, as shown in Table 3. The results demonstrate that such parameter augmentation either yields no significant performance improvement or even degrades model performance. This empirical evidence strongly validates the necessity of employing hyperbolic space for adjusting the vision encoder in MLLMs, as the observed improvements in Table 3 cannot be explained by mere parameter increases.

Necessity of Möbius matrix multiplication. We conducted ablations evaluating matrix multiplication types. Results were largely similar regardless of the multiplication operation used. This suggests Möbius multiplication may have limited practical impact on performance.

Discussion of Different Vision Encoders. To more clearly demonstrate the primary contributing factor behind the performance improvements, we present additional experimental results designed to isolate and eliminate the influence of merely increasing the number of trainable parameters. As shown in Table 4, fine-tuning SAM [29] and DINOv2 [46] in Euclidean space with the same number of additional parameters as HyperET yields only marginal gains. In contrast, fine-tuning with HyperET results in substantial performance improvements. These findings clearly illustrate that our main contribution arises specifically from the proposed hyperbolic radius adjustment mechanism, rather than simply from introducing more learnable parameters.

5.4 Ablation Study on Pre-training

To further assess the contribution of each key component in the proposed HyperET, we isolate each component in separate experiments, and the results are presented in Table 5. Comparing row 2 and row 3, training in Euclidean space instead of hyperbolic space leads to significantly lower performance, indicating that the granularity gap mismatch cannot be effectively addressed in Euclidean space. The observed improvement over the baseline is primarily attributable to the increased number of parameters rather than the alignment of granularity levels. Intriguingly, comparing row 2 and row 4, simply transitioning from Euclidean space to hyperbolic space yields a slight improvement, which we attribute to the inherent ability of hyperbolic space to capture granularity levels. However, this strategy lacks explicit mechanisms for hyperbolic radius adjustment, limiting its effectiveness in addressing granularity mismatches. Finally, by adjusting the hyperbolic radius through Möbius matrix multiplication and integrating it with learnable matrices Ws, we observe significant performance improvements, e.g., a 2.1% gain on TextVQA [53], demonstrating the effectiveness of HyperET. Additionally, we visualize the changes in the hyperbolic radius in Fig. 2 after training, and the observed change in the hyperbolic radius indicates the MLLM’s requirement for multi-granularity levels, demonstrating the necessity of our proposed HyperET.

Figure 2: Visualization of hyperbolic radius changes in visual representation after training across different MLLM benchmarks. Normalizing the hyperbolic radius to a range of 0–1 facilitates comparison. A smaller hyperbolic radius corresponds to a lower granularity level of visual representation. “Zero-shot”: maintaining the pretrained weights of the vision encoder, i.e., CLIP, without additional training.


## Conclusion


6 Conclusion

This work proposes an efficient training paradigm (HyperET) for multimodal large language models in hyperbolic space. By dynamically adjusting the hyperbolic radius of visual representations through learnable matrices and Möbius multiplication operations, HyperET effectively bridges the granularity gap between visual and textual modalities at arbitrary granularity levels. Our experiments across several MLLM benchmarks indicate that HyperET can improve baseline performance in some settings with a small parameter increase, though the magnitude of gains varies across tasks.


![Figure 33](image_33.jpg)
<!-- Caption: Conclusion, index 33 -->


Acknowledgment.
This work was supported by the NSFC under Grant 62322604, 62176159, and in part by the Shanghai Municipal Science and Technology Major Project under Grant 2021SHZDZX0102. We acknowledge that hyperbolic training techniques may introduce new biases or affect model interpretability, and recommend future work to explore these limitations and ethical considerations explicitly.


## References


References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774, 2023.

[2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: A Visual Language Model for Few-Shot Learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022.

[3] Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. PaLM 2 Technical Report. arXiv preprint arXiv:2305.10403, 2023.

[4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities. arXiv preprint arXiv:2308.12966, 2023.

[5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923, 2025.

[6] Martin R. Bridson and André Haefliger. Metric Spaces of Non-Positive Curvature, volume 319. Springer Science & Business Media, 2013.

[7] James W. Cannon, William J. Floyd, Richard Kenyon, Walter R. Parry, et al. Hyperbolic geometry. Flavors of Geometry, 31:59–115, 1997.

[8] Jiaxin Chen, Jie Qin, Yuming Shen, Li Liu, Fan Zhu, and Ling Shao. Learning Attentive and Hierarchical Representations for 3D Shape Recognition. In Computer Vision – ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XV, pages 105–122. Springer, 2020.

[9] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing Multimodal LLM’s Referential Dialogue Magic. arXiv preprint arXiv:2306.15195, 2023.

[10] Xi Chen, Xiao Wang, Soravit Changpinyo, A. J. Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. PaLI: A Jointly-Scaled Multilingual Language-Image Model. arXiv preprint arXiv:2209.06794, 2022.

[11] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling. arXiv preprint arXiv:2412.05271, 2024.

[12] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites. arXiv preprint arXiv:2404.16821, 2024.

[13] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. InternVL: Scaling Up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[14] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, et al. Vicuna: An open-source chatbot impressing GPT-4 with 90%* ChatGPT quality. See https://vicuna.lmsys.org (accessed 14 April 2023), 2(3):6, 2023.

[15] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024.

[16] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. InstructBLIP: Towards general-purpose vision-language models with instruction tuning, 2023.

[17] Karan Desai, Maximilian Nickel, Tanmay Rajpurohit, Justin Johnson, and Ramakrishna Vedantam. Hyperbolic image-text representations. In International Conference on Machine Learning, pages 7694–7731. PMLR, 2023.

[18] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. EVA: Exploring the limits of masked visual representation learning at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19358–19369, 2023.

[19] Octavian Ganea, Gary Bécigneul, and Thomas Hofmann. Hyperbolic neural networks. Advances in Neural Information Processing Systems, 31, 2018.

[20] Songwei Ge, Shlok Mishra, Simon Kornblith, Chun-Liang Li, and David Jacobs. Hyperbolic contrastive learning for visual representations beyond objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6840–6849, 2023.

[21] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6904–6913, 2017.

[22] Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. VizWiz Grand Challenge: Answering visual questions from blind people. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3608–3617, 2018.

[23] Chengyang Hu, Ke-Yue Zhang, Taiping Yao, Shouhong Ding, and Lizhuang Ma. Rethinking generalizable face anti-spoofing via hierarchical prototype-guided distribution refinement in hyperbolic space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1032–1041, 2024.

[24] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

[25] Drew A. Hudson and Christopher D. Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6700–6709, 2019.

[26] Shibo Jie, Yehui Tang, Ning Ding, Zhi-Hong Deng, Kai Han, and Yunhe Wang. Memory-space visual prompting for efficient vision-language fine-tuning. arXiv preprint arXiv:2405.05615, 2024.

[27] Oğuzhan Fatih Kar, Alessio Tonioni, Petra Poklukar, Achin Kulshrestha, Amir Zamir, and Federico Tombari. BRAVE: Broadening the visual encoding of vision-language models. In European Conference on Computer Vision, pages 113–132. Springer, 2024.

[28] Valentin Khrulkov, Leyla Mirvakhabova, Evgeniya Ustinova, Ivan Oseledets, and Victor Lempitsky. Hyperbolic image embeddings. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6418–6428, 2020.

[29] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, et al. Segment Anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023.

[30] Fanjie Kong, Yanbei Chen, Jiarui Cai, and Davide Modolo. Hyperbolic Learning with Synthetic Captions for Open-World Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16762–16771, 2024.

[31] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. LISA: Reasoning Segmentation via Large Language Model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9579–9589, 2024.

[32] Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. LLaVA-NeXT: Stronger LLMs Supercharge Multimodal Capabilities in the Wild. 2024.

[33] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Jingkang Yang, Chunyuan Li, and Ziwei Liu. MIMIC-IT: Multimodal In-Context Instruction Tuning. arXiv preprint arXiv:2306.05425, 2023.

[34] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension. arXiv preprint arXiv:2307.16125, 2023.

[35] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. In International Conference on Machine Learning, pages 19730–19742. PMLR, 2023.

[36] Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. VideoChat: Chat-Centric Video Understanding. arXiv preprint arXiv:2305.06355, 2023.

[37] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating Object Hallucination in Large Vision-Language Models. arXiv preprint arXiv:2305.10355, 2023.

[38] Li Zhang, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image Resolution and Text Label Are Important Things for Large Multimodal Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26763–26773, 2024.

[39] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved Baselines with Visual Instruction Tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024.

[40] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning. Advances in Neural Information Processing Systems, 36, 2023.

[41] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. MMBench: Is Your Multimodal Model an All-Around Player? arXiv preprint arXiv:2307.06281, 2023.

[42] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.

[43] Gen Luo, Yiyi Zhou, Tianhe Ren, Shengxin Chen, Xiaoshuai Sun, and Rongrong Ji. Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models. Advances in Neural Information Processing Systems, 36, 2023.

[44] Feipeng Ma, Hongwei Xue, Yizhou Zhou, Guangting Wang, Fengyun Rao, Shilin Yan, Yueyi Zhang, Siying Wu, Mike Zheng Shou, and Xiaoyan Sun. Visual Perception by Large Language Models’ Weights. arXiv preprint arXiv:2405.20339, 2024.

[45] Maximilian Nickel and Douwe Kiela. Learning continuous hierarchies in the Lorentz model of hyperbolic geometry. In International Conference on Machine Learning, pages 3779–3788. PMLR, 2018.

[46] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[47] Avik Pal, Max van Spengler, Guido Maria D'Amely di Melendugno, Alessandro Flaborea, Fabio Galasso, and Pascal Mettes. Compositional entailment learning for hyperbolic vision-language models. arXiv preprint arXiv:2410.06912, 2024.

[48] Zelin Peng, Zhengqin Xu, Zhilin Zeng, Changsong Wen, Yu Huang, Menglin Yang, Feilong Tang, and Wei Shen. Understanding fine-tuning CLIP for open-vocabulary semantic segmentation in hyperbolic space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4562–4572, 2025.

[49] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023.

[50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021.

[51] Pooyan Rahmanzadeh Gervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision-Language Models Are Blind. In Proceedings of the Asian Conference on Computer Vision, pages 18–34, 2024.

[52] Sameera Ramasinghe, Violetta Shevchenko, Gil Avraham, and Thalaiyasingam Ajanthan. Accept the modality gap: An exploration in the hyperbolic space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27263–27272, 2024.

[53] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8317–8326, 2019.

[54] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. EVA-CLIP: Improved training techniques for CLIP at scale. arXiv preprint arXiv:2303.15389, 2023.

[55] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023.

[56] Yi-Lin Sung, Jaemin Cho, and Mohit Bansal. VL-Adapter: Parameter-efficient transfer learning for vision-and-language tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5227–5237, 2022.

[57] Dídac Surís, Ruoshi Liu, and Carl Vondrick. Learning the predictability of the future. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12607–12617, 2021.

[58] Yehui Tang, Fangcheng Liu, Yunsheng Ni, Yuchuan Tian, Zheyuan Bai, Yiqi Hu, Sichao Liu, Shangling Jui, Kai Han, and Yunhe Wang. Rethinking optimization and architecture for tiny language models. arXiv preprint arXiv:2402.02791, 2024.

[59] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? Exploring the visual shortcomings of multimodal LLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9568–9578, 2024.

[60] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

[61] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[62] Abraham A. Ungar. Hyperbolic trigonometry and its application in the Poincaré ball model of hyperbolic geometry. Computers & Mathematics with Applications, 41(1–2):135–147, 2001.

[63] Abraham Albert Ungar. A gyrovector space approach to hyperbolic geometry. Synthesis Lectures on Mathematics and Statistics, 1(1):1–194, 2008.

[64] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

[65] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al. VisionLLM: Large language model is also an open-ended decoder for vision-centric tasks. Advances in Neural Information Processing Systems, 36, 2023.

[66] Wenxuan Wang, Quan Sun, Fan Zhang, Yepeng Tang, Jing Liu, and Xinlong Wang. Diffusion feedback helps CLIP see better. arXiv preprint arXiv:2407.20171, 2024.

[67] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. National Science Review, 11(12):nwae403, 2024.

[68] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. MM-Vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.

[69] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language-image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023.

[70] Hang Zhang, Xin Li, and Lidong Bing. Video-LLaMA: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023.

[71] Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao. LLaMA-Adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023.

[72] Han Zhao, Min Zhang, Wei Zhao, Pengxiang Ding, Siteng Huang, and Donglin Wang. Cobra: Extending mamba to multi-modal large language model for efficient inference. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 10421–10429, 2025.

[73] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. Advances in Neural Information Processing Systems, 36:46595–46623, 2023.


## Appendix


Appendix of HyperET

This appendix provides additional theoretical and empirical details to support our work. Section A introduces the formal definition of hyperbolicity. Section B presents the full derivation of the core theorem introduced in the main manuscript. Section D offers an extended empirical comparison between HyperET and representative training methods in hyperbolic space, i.e., MERU, further validating the effectiveness of HyperET.

A The Definition of Hyperbolicity

This section introduces the concepts of hyperbolicity utilized in this work, including Möbius addition operation, Möbius scalar multiplication operation, and the tangent space.

Möbius addition operation. For X, Y ∈ 𝔻c^n, the Möbius addition operation is defined as:
X ⊕c Y := [(1 + 2c⟨X, Y⟩ + c∥Y∥^2)X + (1 − c∥X∥^2)Y] / [1 + 2c⟨X, Y⟩ + c^2∥X∥^2∥Y∥^2]. (14)

Möbius scalar multiplication operation. For a scalar r ∈ ℝ and a vector X ∈ 𝔻c^n, the Möbius scalar multiplication operation is defined as:
r ⊗c X := (1/√c) tanh(r · tanh^−1(√c∥X∥)) X/∥X∥. (15)

Möbius matrix multiplication operation. Refer to the relation definition in [28], for a matrix M ∈ ℝ^{n×n} and a point X ∈ 𝔻c^n, if MX ≠ 0, the Möbius matrix multiplication operation is defined as:
M ⊗c X := (1/√c) tanh((∥MX∥/∥X∥) · tanh^−1(√c∥X∥)) (MX/∥MX∥).

Tangent Space. The tangent space 𝒯^c_X 𝔻c^n at a point X ∈ 𝔻c^n is the first order approximation of 𝔻c^n, which is an n-dimensional Euclidean space. The tangent space 𝒯^c_X 𝔻c^n and 𝔻c^n are mapped to each other by exponential (𝒯^c_X 𝔻c^n → 𝔻c^n: exp^{𝔻c^n,c}_X(·)) and logarithmic (𝔻c^n → 𝒯^c_X 𝔻c^n: log^{𝔻c^n,c}_X(·)) maps, respectively. For any X, Y ∈ 𝔻c^n and V ∈ 𝒯^c_X 𝔻c^n, the mapping functions are given for V ≠ 0 and Y ≠ X by:
exp^{𝔻c^n,c}_X(V) = X ⊕c (tanh((√c λ_{c,X}∥V∥)/2) · V/(√c∥V∥)). (16)
log^{𝔻c^n,c}_X(Y) = (2/(√c λ_{c,X})) tanh^−1(√c ∥−X ⊕c Y∥) · (−X ⊕c Y)/∥−X ⊕c Y∥. (17)

B Derivation of the Theorem

Theorem 1 (Revised Flawed) For a point X ∈ 𝔻c^n in hyperbolic space, the hyperbolic radius adjustment is approximately given by:
s · Rad_X ≈ (2/√c)(s (√c/2) Rad_X)
≈ (2/√c) tanh^−1(√c ∥s ⊗c X∥)
However, this relation does not strictly hold due to nonlinearities in Möbius multiplication, meaning precise control over radius via scalar multiplication is not guaranteed.

where ⊗c here is instantiated as a Möbius scalar multiplication operation. Therefore, hyperbolic radius adjustment can be precisely controlled through ⊗c between s and X, with the scaling coefficient s serving as a primary learnable parameter.

Proof. In a hyperbolic space, considering a point X ∈ 𝔻c^n with hyperbolic radius Rad_X := (2/√c) tanh^−1(√c∥X∥), the detailed expansion of the hyperbolic radius adjustment function is given above.

formulated as:

s · Rad_X = (2/√c)(s(√c/2) Rad_X)
= (2/√c) tanh^{-1}(tanh(s(√c/2) Rad_X))
= (2/√c) tanh^{-1}(√c tanh(s(√c/2) Rad_X) ∥X∥/∥X∥)
= (2/√c) tanh^{-1}(√c ∥(tanh(s(√c/2) Rad_X)/(√c∥X∥)) X∥)
= (2/√c) tanh^{-1}(√c ∥(tanh(s tanh^{-1}(√c∥X∥))/(√c∥X∥)) X∥)
= (2/√c) tanh^{-1}(√c ∥s ⊗_c X∥)
= Rad_{s ⊗_c X}.
(19)

Consequently, according to Eq. (19), the scaling scalar s can adjust the hyperbolic radius Rad_X via the Möbius scalar multiplication operation.

Theorem 2 (Hyperbolic Radius Flexibility Scaling) For a point X ∈ D^n_c in hyperbolic space and a scaling matrix X_s ∈ R^{n×n}, the flexible radius adjustment function is formally defined through Eq. (7) and Möbius matrix multiplication operations as follows:

(∥X_s X∥/∥X∥) · Rad_X = (2/√c)((∥X_s X∥/∥X∥)(√c/2) Rad_X)
= (2/√c) tanh^{-1}(√c ∥X_s ⊗_c X∥)
= Rad_{X_s ⊗_c X}.
(20)

Proof. In a hyperbolic space, considering a point X ∈ D^n_c and a scaling matrix X_s ∈ R^{n×n}, the detailed expansion of the hyperbolic radius flexibility adjustment function can be formulated as:

(∥X_s X∥/∥X∥) · Rad_X
= (2/√c) tanh^{-1}(tanh((∥X_s X∥/∥X∥)(√c/2) Rad_X) ∥X_s X∥/∥X_s X∥)
= (2/√c) tanh^{-1}(∥(tanh((∥X_s X∥/∥X∥)(√c/2) Rad_X)/∥X_s X∥) X_s X∥)
= (2/√c) tanh^{-1}(√c ∥(tanh((∥X_s X∥/∥X∥)(√c/2) Rad_X)/(√c∥X_s X∥)) X_s X∥)
= (2/√c) tanh^{-1}(√c ∥(1/√c) tanh((∥X_s X∥/∥X∥) tanh^{-1}(√c∥X∥)) (X_s X/∥X_s X∥)∥)
= (2/√c) tanh^{-1}(√c ∥X_s ⊗_c X∥)
= Rad_{X_s ⊗_c X}.
(21)

Consequently, according to Eq. (21), the scaling matrix X_s can adjust the hyperbolic radius Rad_X via the Möbius matrix multiplication operation.

C Validating HyperET via Comparison with Hyperbolic Training Methods

To further validate the effectiveness of our approach, we also compare HyperET with representative training methods that utilize hyperbolic space, i.e., MERU [17]. MERU [17] and its variants establish an asymmetric visual–semantic hierarchy that emphasizes a subordinate relationship wherein textual features dominate visual ones, inherently intensifying the granularity gap between text and visual modalities, leading to inferior performance. In contrast, HyperET attempts to align visual and textual modalities at arbitrary granularity levels. As shown in Table 6, our experimental results demonstrate that HyperET achieves superior performance compared to MERU, empirically validating the effectiveness and contribution of HyperET in the context of hyperbolic geometry.

Table 6: Comparison with MERU [17] on 12 MLLM benchmarks, including VQAv2 [21], GQA [25], VW: VizWiz [22], SQA: ScienceQA-IMG [42], TVQA: TextVQA [53], PE: POPE [37], ME: MME [67], MB: MMBench [41], MBCN: MMBench-Chinese [41], SD: SEED-Bench [34], LVAW: LLaVA-Bench (In-the-Wild) [40] and M-Vet [68]. Lang. Model: Language model. Benchmark names are abbreviated to be consistent with the main manuscript. “Ours”: we here realize the extra learnable parameters as full matrices, i.e., Ws. Vision encoder: CLIP.

| Method         | Lang. Model | VQAv2 | GQA  | VW   | SQA  | TVQA | PE   | ME     | MB   | MBCN | SD   | LVAW | M-Vet |
| -------------- | ----------- | ----- | ---- | ---- | ---- | ---- | ---- | ------ | ---- | ---- | ---- | ---- | ----- |
| LLaVA-1.5      | Vicuna-13B  | 80.0  | 63.3 | 53.6 | 71.6 | 61.3 | 85.9 | 1531.3 | 67.7 | 63.6 | 61.6 | 70.7 | 35.4  |
| LLaVA-1.5+MERU | Vicuna-13B  | 78.9  | 61.0 | 48.0 | 65.7 | 60.2 | 84.7 | 1456.8 | 62.3 | 59.1 | 56.6 | 68.4 | 32.5  |
| LLaVA-1.5+Ours | Vicuna-13B  | 82.3  | 65.7 | 55.2 | 73.7 | 63.9 | 88.7 | 1584.7 | 69.8 | 65.2 | 63.4 | 72.6 | 38.3  |

D Validation on Non-Transformer MLLMs

To demonstrate our method’s effectiveness on different architectures, we experiment with HyperET on a non-Transformer architecture [72], i.e., Mamba. As shown in Table 7, our method yields consistent gains across all MLLM benchmarks, further demonstrating the robustness and generalizability of our approach.

Table 7: Comparison with Cobra [72] on 5 MLLM benchmarks, including VQAv2 [21], GQA [25], VW: VizWiz [22], TVQA: TextVQA [53], PE: POPE [37]. Benchmark names are abbreviated to be consistent with the main manuscript. “Ours”: we here realize the extra learnable parameters as full matrices, i.e., Ws.

| Method     | VQAv2 | GQA  | VW   | TVQA | PE   |
| ---------- | ----- | ---- | ---- | ---- | ---- |
| Cobra      | 79.2  | 63.9 | 56.2 | 59.5 | 87.6 |
| Cobra+Ours | 80.9  | 65.2 | 56.9 | 60.6 | 88.8 |