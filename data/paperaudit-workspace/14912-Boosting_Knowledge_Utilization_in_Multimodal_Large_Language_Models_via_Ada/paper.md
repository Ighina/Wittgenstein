
## Abstract


Boosting Knowledge Utilization in Multimodal Large Language Models via Adaptive Logits Fusion and Attention Reallocation

Wenbin An^1,2,* Jiahao Nie^3,* Feng Tian^1,2,† Haonan Lin^1,2 Mingxiang Cai^4 Yaqiang Wu^4 Qianying Wang^4,† Xiaoqin Zhang^5 Shijian Lu^3,†

^1 Xi’an Jiaotong University ^2 National Engineering Laboratory for Big Data Analytics ^3 Nanyang Technological University ^4 Lenovo Research ^5 Zhejiang University of Technology

wenbinan@stu.xjtu.edu.cn, jiahao007@e.ntu.edu.sg, fengtian@mail.xjtu.edu.cn, wangqya@lenovo.com, shijian.lu@ntu.edu.sg

* Equal contribution † Corresponding author

Abstract

Despite their recent progress, Multimodal Large Language Models (MLLMs) often struggle in knowledge-intensive tasks due to the limited and outdated parametric knowledge acquired during training. Multimodal Retrieval-Augmented Generation addresses this issue by retrieving contextual knowledge from external databases, thereby enhancing MLLMs with expanded knowledge sources. However, existing MLLMs often fail to fully leverage the retrieved contextual knowledge for response generation. We identify two major causes — attention bias and knowledge conflicts — that contribute to suboptimal utilization of retrieved knowledge by MLLMs. We propose Adaptive Logits Fusion and Attention Reallocation (ALFAR), a training-free and plug-and-play approach aimed at improving MLLM responses by better leveraging contextual knowledge. Contradicting prior assumptions, ALFAR employs light training to optimize its fusion weights and attention reallocation parameters, which involve fine-tuning on selected benchmarks, thereby limiting its plug-and-play capability. As a plug-and-play method, ALFAR achieves superior performance across diverse datasets without requiring additional training or external tools. Extensive experiments over multiple MLLMs and benchmarks show that ALFAR consistently outperforms the state of the art by large margins. Our code and data are available at https://github.com/Lackel/ALFAR.


## Introduction


1 Introduction

Building upon powerful Large Language Models (LLMs) [1, 2, 3, 4, 5, 6, 7, 8], Multimodal Large Language Models (MLLMs) [9, 10, 11, 12, 13, 14, 15, 16, 17] have achieved impressive performance over a wide range of vision-centric tasks such as image captioning [18, 19, 20], visual question answering [21, 22], etc. Nevertheless, MLLMs often struggle to handle knowledge-intensive vision-language tasks [23, 24], primarily due to the limited and outdated parametric knowledge acquired during training [25, 26]. Multimodal Retrieval-Augmented Generation (MRAG) [27, 28, 29], a prevalent approach that attempts to resolve this issue, retrieves contextual knowledge from external

Data to empower MLLMs for accurate response generation. However, the way of exploiting the contextual knowledge remains under-explored, undermining the effectiveness of MRAG.

We examined representative MLLMs with MRAG and found that while MRAG can improve MLLM performance when high-quality contextual knowledge is retrieved (as shown in Fig. 1), MLLMs often fail to make full use of the retrieved knowledge, even when ground-truth knowledge is available. We identify two primary causes, namely, attention bias among visual and context tokens and conflicts between MLLMs’ parametric knowledge and retrieved contextual knowledge. For the attention bias, MLLMs tend to allocate more attention to image tokens over context tokens, especially in shallow layers that are critical for knowledge extraction and exchange [30]. Since images often do not provide sufficient information for knowledge-intensive questions [23, 31], the attention bias hinders the effective utilization of contextual knowledge and leads to inaccurate MLLM responses. In addition, MLLMs allocate attention uniformly across context tokens without prioritization, which dilutes the contributions of query-relevant knowledge and tends to introduce inaccurate MLLM responses.

Knowledge conflicts typically arise from the discrepancy between contextual and parametric knowledge. We observe that MLLMs tend to rely excessively on their parametric knowledge even when accurate contextual knowledge is present, leading to under-utilization of contextual knowledge and counterfactual responses. Such a phenomenon is well aligned with observations in previous LLM studies [33, 34] and findings in psychology research [35, 36], both underscoring a clear preference toward intrinsic instead of retrieved knowledge. On the other hand, the preference for the parametric knowledge does help when the contextual knowledge is unreliable [28, 37, 38]. This can be observed in Fig. 1, where low-quality contextual knowledge significantly degrades performance. Therefore, striking a balance between parametric and contextual knowledge while leveraging their complementary strengths is critical for generating accurate responses.

Model results on the multi-choice InfoSeek dataset [32] with LLaVA-1.5 [14]:
- Vanilla LLaVA: Ground-truth knowledge 39.15; 1st retrieved knowledge 65.47; 2nd retrieved knowledge 52.69; 3rd retrieved knowledge 38.03.
- LLaVA + MRAG: 1st retrieved knowledge 71.13; 2nd retrieved knowledge 58.20; 3rd retrieved knowledge 43.10.
- LLaVA + ALFAR: 2nd retrieved knowledge 59.82; 3rd retrieved knowledge 45.12.

Figure 1: VQA accuracy of LLaVA-1.5 [14] on the multi-choice InfoSeek dataset [32] with respect to the quality of contextual knowledge. ALFAR fully exploits the retrieved knowledge, consistently improving MRAG performance regardless of knowledge quality.

In this work, we propose Adaptive Logits Fusion and Attention Reallocation (ALFAR), a training-free and plug-and-play approach that enables effective utilization of MRAG-retrieved contextual knowledge for accurate MLLM responses. ALFAR addresses attention bias and knowledge conflicts by dynamically adjusting attention allocation and balancing the parametric and contextual knowledge, respectively. Specifically, ALFAR adaptively shifts attention from image tokens to relevant context tokens based on retrieval scores and query–context relevance, enabling MLLMs to focus on more pertinent information. In addition, ALFAR decouples parametric and contextual knowledge at output logits and weights them according to the attention distribution, enabling a balanced and synergistic integration of the two types of knowledge. Extensive experiments across multiple representative MLLMs and benchmarks demonstrate ALFAR’s superior and broad applicability without involving additional training or external tools.

The contributions of this work can be summarized in three major aspects. First, we dive deeply into knowledge utilization in MLLMs, identifying attention bias and knowledge conflicts as two key factors that impede the effective utilization of the retrieved knowledge. These findings provide valuable insights for advancing knowledge utilization in MLLMs. Second, we design ALFAR, a training-free and plug-and-play approach that reallocates attention and balances parametric and contextual knowledge effectively. Third, extensive experiments over multiple generative and discriminative benchmarks validate ALFAR’s effectiveness and versatility, demonstrating its superior performance and broad applicability across various multimodal tasks.


## Related Work


2 Related Work

2.1 Multimodal Large Language Models

The rapid advancements in Large Language Models (LLMs) [1, 2, 3, 4, 5, 6, 7, 8, 39] have greatly propelled the development of Multimodal Large Language Models (MLLMs) [9, 10, 11, 13, 14, 15,


![Figure 6](image_6.jpg)
<!-- Caption: Related Work, index 6 -->


To align visual and textual modalities, prior studies explore different approaches such as visual encoders with linear projectors [14, 41, 40], Q-Former [18, 10], and Perceivers [42], which transform image patches into visual tokens that are compatible with LLMs. Most MLLMs conduct training in two stages, namely, pre-training for feature alignment and instruction-based fine-tuning [14, 43, 10], enabling impressive performance across diverse multimodal tasks [44, 45, 46, 47]. Despite these advancements, MLLMs often face challenges in knowledge-intensive tasks [23, 31], due to the limitations of their parametric knowledge acquired during training.

## 2.2 Multimodal Retrieval Augmented Generation

Inspired by the concept of Retrieval Augmented Generation (RAG) for LLMs [48, 49], Multimodal Retrieval Augmented Generation (MRAG) has been widely explored for enhancing MLLMs with more comprehensive and up-to-date knowledge [27, 28, 29, 50]. MRAG retrieves relevant knowledge from a multimodal database and incorporates the retrieved knowledge as the context of the input. For instance, Wiki-LLaVA [27] broadens the knowledge scope of LLaVA [14] by incorporating retrieved Wikipedia articles in training. EchoSight [29] leverages a fine-tuned Q-Former [18] to filter retrieved knowledge and enhance retrieval recall. ReflectiVA [28] and MR²AG [50] introduce a trainable reflection mechanism to assess the necessity of retrieval and the relevance of retrieved knowledge. In the LLM domain, several training-free methods have been proposed to better utilize the retrieved knowledge to enhance generation quality. For instance, CAD [26] employs contrastive decoding [51] to increase the faithfulness of generation. Moreover, AdaCAD [52], Entropy [53], and COIECD [54] extend CAD [26] by introducing JS divergence, entropy, and information constraints, respectively. Although these methods aim to balance parametric and contextual knowledge, some recent approaches have achieved notable improvements in noisy contexts through advanced fusion or re-ranking, which are not discussed here.


## Preliminaries


## 3 Preliminary and Motivation

### 3.1 Multimodal Retrieval Augmented Generation

Given a textual query $q$ and a query image $I$, an MLLM $\mathcal{M}_\theta$ parameterized by $\theta$ is expected to generate a reliable answer $y$. To enrich MLLMs with external knowledge, MRAG employs a multimodal retriever $\mathcal{R}_\phi$ to fetch relevant knowledge from a multimodal knowledge base $\mathcal{C} = \{(\tilde{I}_i, c_i)\}_{i=1}^M$, where $\tilde{I}_i$ and $c_i$ represent an image and its corresponding textual knowledge, respectively. The retriever $\mathcal{R}_\phi$ measures the similarity between the query pair $(q, I)$ and a multimodal knowledge pair $(\tilde{I}, c)$ based on the cosine similarity between their image embeddings:
$$\alpha = \frac{\mathcal{R}_\phi(I) \cdot \mathcal{R}_\phi(\tilde{I})}{\|\mathcal{R}_\phi(I)\| \cdot \|\mathcal{R}_\phi(\tilde{I})\|}$$
(1)

The textual knowledge $c$ with the highest retrieval similarity $\alpha$ is selected as the input context for MLLMs. Consequently, the output distributions of the MLLM with MRAG at the time step $t$ are:
$$p(y_t) \sim \text{softmax}(\mathcal{M}_\theta(y_t \mid q, I, c, y_{<t}))$$
(2)
where $y_{<t}$ represents the sequence of generated tokens before the time step $t$.

### 3.2 Self-attention Mechanism in MLLMs

MLLMs generate responses auto-regressively using Transformer blocks [56]. Specifically, the input image, query, and context tokens are concatenated and projected into three distinct vectors: the query vector Q, the key vector K, and the value vector V, through three linear layers, $W_q$, $W_k$, and $W_v$. The self-attention mechanism computes the relevance of each token to other tokens as follows:
$$\mathbf{A} = \frac{\mathbf{Q} \cdot \mathbf{K}^\top}{\sqrt{d}} + \mathbf{M}$$
(3)

Figure 2: Proportions of attention weights that are assigned to image and context tokens at different shallow layers of LLaVA-1.5 [14].


![Figure 12](image_12.jpg)
<!-- Caption: Preliminaries, index 12 -->


where A ∈ ℝ^{n×n} is the attention weight matrix, M is a causal mask, d is the feature dimension, and n is the number of input tokens. The output O can then be calculated by:
O = softmax(A) · V (4)
The attention weight matrix A reflects the importance of different input tokens in generating new tokens. This property makes it a valuable tool to analyze the contribution of different types of tokens to the generated responses.


## Motivation


## 3.3 Attention Bias in MRAG

To generate the response token y_{n+1}, MLLMs perform self-attention over all input tokens that correspond to the n-th row of the attention weight matrix A. We analyze the contributions of the input image and context using an importance score, defined as the total attention weights assigned to these tokens. For the input image, the importance score at layer i is calculated by: S^i(I) = ∑_{j∈I} A^i_{nj}. Similarly, for the input context, the importance score at layer i is determined by: S^i(c) = ∑_{j∈c} A^i_{nj}. As illustrated in Fig. 2, MLLMs tend to allocate more attention to image tokens than context tokens, particularly in shallow layers that are pivotal for extracting and exchanging information from distinct tokens [30]. This issue affects the effective utilization of contextual knowledge since images often do not capture sufficient information for knowledge-intensive questions [23, 31]. Moreover, MLLMs assign attention uniformly across different parts of the context without highlighting query-relevant segments. Such indiscriminate distribution increases the distraction of irrelevant knowledge, ultimately leading to inaccurate or misleading responses for MLLMs.

## 3.4 Knowledge Conflicts in MRAG

After knowledge retrieval, MLLMs integrate the retrieved contextual knowledge with their internal parametric knowledge to generate responses. However, similar to LLMs [33, 34, 52], MLLMs often encounter knowledge conflicts due to the discrepancy between the two types of knowledge, affecting the effectiveness of the model in various practical tasks. Worse still, MLLMs tend to prioritize their parametric knowledge even when perfect contextual knowledge is provided, leading to under-utilization of contextual knowledge and factually inconsistent answers. By assuming the accessibility of the ground-truth contextual knowledge, we evaluate the conflict rate of the two types of knowledge and the resultant performance degradation on the multi-choice InfoSeek [32] and ViQuAE [24, 32] datasets with LLaVA-1.5 [14]. As shown in Tab. 1, around half of the samples exhibit knowledge conflicts, which lead to an up to 30% performance drop, highlighting the necessity of mitigating such conflicts to enhance MLLM performance on knowledge-intensive tasks.

Table 1: Experiments with LLaVA-1.5 [14]. Conflict Ratio: Ratio of discrepancy between parametric and contextual knowledge. Performance drop: Accuracy decline due to knowledge conflicts. Details of the two metrics are provided in Appendix A3.

|                  | InfoSeek | ViQuAE |
| ---------------- | -------- | ------ |
| Conflict Ratio   | 60.85%   | 48.94% |
| Performance Drop | 28.87%   | 27.02% |


## Method


# 4 Method

The proposed framework consists of two branches for effective handling of parametric and contextual knowledge as illustrated in Fig. 4. Within the contextual branch, we design an attention reallocation mechanism that tackles the attention bias and improves the utilization of contextual knowledge by adaptively adjusting model attention toward relevant context tokens based on query-context relevance (Sec. 4.1). In addition, the network fuses the parametric and contextual knowledge adaptively in the output logits, mitigating knowledge conflicts under the guidance of the model attention that dynamically captures the relative importance of the two types of knowledge (Sec. 4.2).

## 4.1 Attention Reallocation

As analyzed in Sec. 3.3, the attention bias results from two major factors, namely, attention preference toward image tokens and uniform attention to context tokens. We address the attention preference

1 We average all attention heads and omit the notation for simplicity.

Figure 4: The overview of Adaptive Logits Fusion and Attention Reallocation (ALFAR).

By adaptively adjusting model attention as illustrated in Fig. 3, based on the retrieval similarity α in Eq. 1 that reflects the reliability of the retrieved context. Specifically, less attention is allocated to image tokens if the context is more reliable with a high retrieval similarity. The attention reallocation modifies attention after verifying that causal mask constraints remain intact, ensuring no causal attention is granted to future tokens. Specifically, masked attention weights remain zero and only valid attention scores are rescaled to preserve autoregressive decoding consistency.

In addition, we introduce a query-context relevance score to mitigate the uniform attention to all context tokens and allow MLLMs to focus more on query-relevant context. We derive the relevance score from attention weights assigned to query tokens by the $j$-th context token as follows:

$$\omega_j = \frac{\sum_{k \in S_q} \mathbf{A}_{jk}}{\sum_{l \in S_c} \sum_{k \in S_q} \mathbf{A}_{lk}}, \text{ s.t. } j \in S_c$$
(6)

where $S_q$ and $S_c$ are index sets of query and context tokens, respectively. With the relevance scores, we adaptively increase MLLMs' attention to context tokens:

$$\hat{\mathbf{A}}_{nj} = (1 + \gamma_j) \cdot \mathbf{A}_{nj}, \text{ s.t. } j \in S_c$$
(7)

where $\gamma_j = k \cdot \omega_j$ is the scaled query-context relevance with a scaling factor $k$. After attention reallocation, we apply softmax to redistribute the attention as in Eq. 4 to compute the output hidden states. This repeats auto-regressively for each subsequent token prediction.

## 4.2 Adaptive Knowledge Fusion

As analyzed in Sec. 3.4, parametric knowledge could hinder the utilization of contextual knowledge, leading to inaccurate responses. However, parametric knowledge brings benefits when the retrieved context is unreliable. Therefore, striking a balance between parametric and contextual knowledge based on their reliability is essential for generating accurate and reliable responses for MLLMs. Nevertheless, parametric knowledge is implicitly embedded and the two types of knowledge are entangled during inference, making it hard to explicitly represent and utilize them separately. We address this issue by disentangling the two types of knowledge and fusing them at output logits. Specifically, we represent parametric knowledge at step $t$ by using the output logits that have only query and image as inputs:

$$\text{logit}_{\text{p}} = \mathcal{M}_\theta(y_t \mid q, I, y_{<t})$$
(8)


![Figure 20](image_20.jpg)
<!-- Caption: Method, index 20 -->



![Figure 21](image_21.jpg)
<!-- Caption: Method, index 21 -->



## Experiments


Table 2: VQA Accuracy comparison on generative freeform VQA datasets over three runs. Regular and Parametric denote that MLLMs generate answers with and without retrieved knowledge, respectively. The reported ALFAR improvements over baselines are modest, with overall gains reduced by 3-5%, indicating less dramatic performance gains than claimed.

| Model        | Decoding        | Human [23] UnseenQuestion | Human [23] UnseenEntity | Human [23] Overall | Validation [23] UnseenQuestion | Validation [23] UnseenEntity | Validation [23] Overall |
| ------------ | --------------- | ------------------------- | ----------------------- | ------------------ | ------------------------------ | ---------------------------- | ----------------------- |
| LLaVA-1.5    | Regular [14]    | 7.59(±0.08)               | 7.90(±0.05)             | 7.74(±0.01)        | 19.98(±0.02)                   | 19.59(±0.01)                | 19.78(±0.01)            |
|              | Parametric [14] | 6.27(±0.05)               | 6.26(±0.19)             | 6.26(±0.09)        | 7.14(±0.03)                    | 6.28(±0.01)                 | 6.68(±0.00)             |
|              | CD [51]         | 7.39(±0.21)               | 7.31(±0.06)             | 7.35(±0.09)        | 20.32(±0.01)                   | 19.90(±0.00)                | 20.11(±0.01)            |
|              | AdaCAD [52]     | 7.81(±0.02)               | 8.07(±0.03)             | 7.94(±0.01)        | 21.23(±0.03)                   | 20.91(±0.02)                | 21.07(±0.03)            |
|              | Entropy [53]    | 7.98(±0.09)               | 8.34(±0.02)             | 8.15(±0.03)        | 21.97(±0.02)                   | 21.85(±0.01)                | 21.91(±0.03)            |
|              | CAD [26]        | 8.02(±0.05)               | 8.15(±0.03)             | 8.08(±0.02)        | 21.68(±0.01)                   | 20.93(±0.02)                | 21.30(±0.02)            |
|              | COIECD [54]     | 8.78(±0.04)               | 8.65(±0.01)             | 8.71(±0.01)        | 22.43(±0.05)                   | 21.73(±0.05)                | 22.07(±0.05)            |
|              | AGLA [57]       | 8.74(±0.28)               | 9.18(±0.00)             | 8.94(±0.08)        | 22.34(±0.02)                   | 21.88(±0.01)                | 22.11(±0.02)            |
|              | VCD [58]        | 9.22(±0.00)               | 9.26(±0.00)             | 9.24(±0.02)        | 22.30(±0.03)                   | 22.38(±0.03)                | 22.23(±0.03)            |
|              | **ALFAR (ours)**| **12.80**(±0.09)          | **11.24**(±0.00)        | **11.96**(±0.02)   | **23.82**(±0.00)               | **23.75**(±0.01)            | **23.78**(±0.01)        |
| InstructBLIP | Regular [10]    | 4.20(±0.00)               | 3.86(±0.03)             | 4.02(±0.01)        | 3.60(±0.01)                    | 3.82(±0.00)                 | 3.71(±0.00)             |
|              | Parametric [10] | 4.06(±0.01)               | 3.65(±0.01)             | 3.84(±0.01)        | 2.36(±0.01)                    | 1.92(±0.00)                 | 2.12(±0.00)             |
|              | CD [51]         | 4.52(±0.03)               | 3.55(±0.01)             | 3.98(±0.01)        | 3.59(±0.01)                    | 4.00(±0.00)                 | 3.79(±0.00)             |
|              | AdaCAD [52]     | 4.57(±0.05)               | 3.70(±0.10)             | 4.09(±0.08)        | 3.71(±0.02)                    | 4.35(±0.01)                 | 4.01(±0.01)             |
|              | Entropy [53]    | 4.56(±0.05)               | 4.14(±0.01)             | 4.34(±0.02)        | 3.81(±0.01)                    | 4.39(±0.00)                 | 4.08(±0.00)             |
|              | CAD [26]        | 4.52(±0.03)               | 3.55(±0.01)             | 3.98(±0.01)        | 3.77(±0.03)                    | 4.43(±0.02)                 | 4.08(±0.02)             |
|              | COIECD [54]     | 4.64(±0.10)               | 4.08(±0.02)             | 4.33(±0.01)        | 4.07(±0.00)                    | 4.54(±0.00)                 | 4.30(±0.00)             |
|              | AGLA [57]       | 4.80(±0.05)               | 4.28(±0.09)             | 4.52(±0.05)        | 3.74(±0.01)                    | 4.10(±0.01)                 | 3.91(±0.01)             |
|              | VCD [58]        | 4.70(±0.01)               | 4.14(±0.01)             | 4.40(±0.00)        | 3.62(±0.01)                    | 4.12(±0.00)                 | 3.85(±0.00)             |
|              | **ALFAR (ours)**| **5.98**(±0.00)           | **5.67**(±0.00)         | **5.82**(±0.00)    | **4.55**(±0.00)                | **5.68**(±0.00)             | **5.05**(±0.00)         |
| Shikra       | Regular [40]    | 6.71(±0.03)               | 6.31(±0.01)             | 6.50(±0.02)        | 11.93(±0.01)                   | 11.78(±0.01)                | 11.85(±0.01)            |
|              | Parametric [40] | 5.76(±0.10)               | 6.10(±0.07)             | 5.92(±0.05)        | 7.61(±0.01)                    | 6.25(±0.01)                 | 6.86(±0.01)             |
|              | CD [51]         | 8.21(±0.00)               | 7.15(±0.01)             | 7.64(±0.01)        | 12.41(±0.00)                   | 11.89(±0.00)                | 12.14(±0.00)            |
|              | AdaCAD [52]     | 8.30(±0.00)               | 7.11(±0.00)             | 7.66(±0.00)        | 12.87(±0.03)                   | 12.53(±0.02)                | 12.70(±0.02)            |
|              | Entropy [53]    | 8.32(±0.03)               | 7.73(±0.11)             | 8.01(±0.05)        | 13.78(±0.02)                   | 13.33(±0.01)                | 13.55(±0.02)            |
|              | CAD [26]        | 8.16(±0.06)               | 7.16(±0.02)             | 7.62(±0.00)        | 12.99(±0.03)                   | 12.51(±0.02)                | 12.75(±0.03)            |
|              | COIECD [54]     | 8.32(±0.02)               | 7.73(±0.08)             | 8.01(±0.03)        | 14.46(±0.02)                   | 14.21(±0.03)                | 14.33(±0.02)            |
|              | AGLA [57]       | 8.24(±0.01)               | 7.56(±0.05)             | 7.88(±0.01)        | 14.29(±0.02)                   | 13.91(±0.01)                | 14.08(±0.01)            |
|              | VCD [58]        | 8.13(±0.05)               | 7.41(±0.03)             | 7.75(±0.04)        | 13.71(±0.03)                   | 13.81(±0.03)                | 13.76(±0.03)            |
|              | **ALFAR (ours)**| **8.61**(±0.01)           | **8.04**(±0.01)         | **8.31**(±0.01)    | **15.25**(±0.00)               | **15.11**(±0.01)            | **15.18**(±0.01)        |
| MiniGPT4     | Regular [16]    | 4.38(±0.07)               | 3.00(±0.02)             | 3.56(±0.02)        | 12.69(±0.02)                   | 12.38(±0.02)                | 12.53(±0.02)            |
|              | Parametric [16] | 2.34(±0.01)               | 2.10(±0.01)             | 2.21(±0.00)        | 4.72(±0.01)                    | 3.93(±0.01)                 | 4.29(±0.01)             |
|              | CD [51]         | 4.28(±0.00)               | 2.59(±0.00)             | 3.22(±0.00)        | 14.49(±0.01)                   | 14.43(±0.00)                | 14.46(±0.01)            |
|              | AdaCAD [52]     | 4.78(±0.00)               | 3.43(±0.01)             | 3.99(±0.00)        | 14.82(±0.02)                   | 14.96(±0.02)                | 14.89(±0.02)            |
|              | Entropy [53]    | 4.80(±0.07)               | 2.91(±0.00)             | 3.62(±0.00)        | 14.66(±0.02)                   | 14.66(±0.01)                | 14.66(±0.02)            |
|              | CAD [26]        | 4.97(±0.01)               | 3.44(±0.01)             | 4.07(±0.01)        | 14.83(±0.01)                   | 14.94(±0.00)                | 14.88(±0.01)            |
|              | COIECD [54]     | 4.57(±0.03)               | 3.40(±0.01)             | 3.90(±0.00)        | 14.87(±0.01)                   | 14.67(±0.02)                | 14.77(±0.02)            |
|              | AGLA [57]       | 4.67(±0.02)               | 3.63(±0.02)             | 4.09(±0.01)        | 14.31(±0.02)                   | 13.92(±0.01)                | 14.11(±0.01)            |
|              | VCD [58]        | 4.52(±0.03)               | 3.43(±0.01)             | 3.90(±0.01)        | 14.46(±0.01)                   | 14.30(±0.02)                | 14.38(±0.02)            |
|              | **ALFAR (ours)**| **5.05**(±0.02)           | **3.87**(±0.01)         | **4.38**(±0.00)    | **15.16**(±0.01)               | **15.27**(±0.01)            | **15.05**(±0.01)        |


## Method


Similarly, we represent contextual knowledge by using the output logits that have context as additional inputs and perform attention reallocation to better utilize the context:
$$\text{logit}_c = \hat{\mathcal{M}}_\theta(y_t \mid q, I, c, y_{<t})$$
(9)
where $\hat{\mathcal{M}}_\theta$ is the MLLM with attention reallocation. The reliability of the parametric and contextual knowledge can thus be measured by the attention weights that are assigned to the image and context tokens, capturing the correlation among tokens [59, 60, 61]:
$$\lambda_v^t = \sum_{i \in S_I} \mathbf{A}_{ti}, \quad \lambda_c^t = \sum_{j \in S_c} \mathbf{A}_{tj}$$
(10)


## Experiments


Table 3: VQA accuracy comparison on discriminative multi-choice VQA datasets over three runs.

| Model      | Decoding         | InfoSeek         | ViQuAE           | Model        | Decoding         | InfoSeek         | ViQuAE           |
| ---------- | ---------------- | ---------------- | ---------------- | ------------ | ---------------- | ---------------- | ---------------- |
| LLaVA-1.5  | Regular [14]     | 51.97(±0.42)     | 53.32(±0.20)     | InstructBLIP | Regular [10]     | 23.44(±0.89)     | 19.82(±0.12)     |
|            | Parametric [14]  | 39.15(±0.02)     | 51.06(±0.16)     |              | Parametric [10]  | 8.73(±0.23)      | 6.53(±0.35)      |
|            | CD [51]          | 49.95(±0.20)     | 52.56(±0.03)     |              | CD [51]          | 22.95(±0.63)     | 21.76(±0.51)     |
|            | CAD [26]         | 52.08(±0.16)     | 52.99(±0.23)     |              | CAD [26]         | 26.56(±0.56)     | 23.18(±0.30)     |
|            | AdaCAD [52]      | 52.30(±0.04)     | 52.99(±0.30)     |              | AdaCAD [52]      | 27.07(±0.05)     | 23.64(±0.08)     |
|            | Entropy [53]     | 53.33(±0.07)     | 54.26(±0.05)     |              | Entropy [53]     | 27.50(±0.07)     | 23.19(±0.39)     |
|            | COIECD [54]      | 52.08(±0.21)     | 52.99(±0.23)     |              | COIECD [54]      | 25.65(±0.05)     | 20.84(±0.05)     |
|            | VCD [58]         | 53.87(±0.07)     | 55.13(±0.09)     |              | VCD [58]         | 23.31(±0.07)     | 20.28(±0.53)     |
|            | AGLA [57]        | 53.53(±0.50)     | 54.24(±0.21)     |              | AGLA [57]        | 21.24(±0.55)     | 16.77(±0.30)     |
|            | ALFAR (ours)     | 58.35(±0.21)     | 55.91(±0.13)     |              | ALFAR (ours)     | 35.65(±0.09)     | 24.11(±0.07)     |
| Shikra     | Regular [40]     | 19.41(±0.12)     | 17.73(±0.01)     | MiniGPT-4    | Regular [16]     | 25.83(±1.42)     | 24.06(±0.46)     |
|            | Parametric [40]  | 9.65(±0.14)      | 10.90(±0.04)     |              | Parametric [16]  | 19.73(±0.57)     | 20.42(±1.22)     |
|            | CD [51]          | 24.83(±0.20)     | 21.38(±0.03)     |              | CD [51]          | 26.55(±0.31)     | 20.46(±0.76)     |
|            | CAD [26]         | 24.51(±0.06)     | 21.68(±0.15)     |              | CAD [26]         | 27.58(±0.43)     | 23.39(±0.07)     |
|            | AdaCAD [52]      | 23.95(±0.18)     | 21.51(±0.01)     |              | AdaCAD [52]      | 28.01(±0.17)     | 23.05(±0.61)     |
|            | Entropy [53]     | 24.12(±0.03)     | 21.99(±0.08)     |              | Entropy [53]     | 28.84(±0.68)     | 22.59(±0.07)     |
|            | COIECD [54]      | 24.18(±0.13)     | 21.68(±0.15)     |              | COIECD [54]      | 29.44(±0.01)     | 25.94(±0.18)     |
|            | VCD [58]         | 25.76(±0.14)     | 23.11(±0.72)     |              | VCD [58]         | 28.74(±0.10)     | 25.45(±0.05)     |
|            | AGLA [57]        | 26.26(±0.29)     | 22.72(±0.15)     |              | AGLA [57]        | 29.28(±0.87)     | 27.87(±0.08)     |
|            | ALFAR (ours)     | 28.11(±0.01)     | 23.42(±0.29)     |              | ALFAR (ours)     | 30.88(±0.12)     | 32.28(±0.02)     |
| LLaVA-Next | Regular [62]     | 53.90(±0.26)     | 55.44(±0.12)     | Qwen2.5-VL   | Regular [63]     | 53.00(±0.18)     | 53.84(±0.01)     |
|            | Parametric [62]  | 42.50(±0.12)     | 53.94(±0.11)     |              | Parametric [63]  | 45.96(±0.32)     | 53.28(±0.15)     |
|            | CD [51]          | 51.43(±0.23)     | 51.38(±0.01)     |              | CD [51]          | 56.23(±0.23)     | 59.33(±0.15)     |
|            | CAD [26]         | 53.83(±0.13)     | 52.84(±0.27)     |              | CAD [26]         | 58.67(±0.25)     | 60.92(±0.20)     |
|            | AdaCAD [52]      | 55.60(±0.34)     | 54.87(±0.13)     |              | AdaCAD [52]      | 57.87(±0.05)     | 60.89(±0.28)     |
|            | Entropy [53]     | 54.43(±0.20)     | 55.60(±0.15)     |              | Entropy [53]     | 58.67(±0.13)     | 60.66(±0.19)     |
|            | COIECD [54]      | 53.93(±0.11)     | 54.17(±0.33)     |              | COIECD [54]      | 59.23(±0.10)     | 60.59(±0.02)     |
|            | VCD [58]         | 53.27(±0.27)     | 54.32(±0.03)     |              | VCD [58]         | 58.56(±0.32)     | 60.12(±0.33)     |
|            | AGLA [57]        | 54.03(±0.25)     | 54.65(±0.34)     |              | AGLA [57]        | 59.13(±0.25)     | 61.08(±0.13)     |
|            | ALFAR (ours)     | 58.47(±0.21)     | 59.29(±0.21)     |              | ALFAR (ours)     | 61.57(±0.18)     | 63.22(±0.08)     |

Finally, based on the adaptive weights, the logits fusion at decoding step t is performed by a weighted average: $$p(y_t) \sim \text{softmax}\left( \lambda_v^t \cdot \text{logit}_p + \lambda_c^t \cdot \text{logit}_c \right)$$ ensuring positive weights that sum to 1 to consistently balance parametric and contextual knowledge.

# 5 Experiment

## 5.1 Experimental Settings

Datasets. We conduct experiments over three types of knowledge-intensive datasets: (1) Free-form generative datasets including Human [23], a high-quality info-seeking dataset curated and verified by experts, and INFOSEEK<sub>wiki</sub> [23] which encompasses diverse entities from Wikidata. For INFOSEEK<sub>wiki</sub>, we adopt its Validation set for evaluations to be aligned with prior studies [27, 28]. (2) Multi-choice discriminative datasets including Infoseek [23, 32] and ViQuAE [24, 32] which are both multi-choice knowledge-intensive datasets that are collected for assessing cross-modality knowledge conflicts as described in [32]. (3) Knowledge-based datasets including OK-VQA [64], AOK-VQA [65] and Encyclopedic VQA (E-VQA) [31], which are widely adopted for evaluations of tasks that require commonsense knowledge. Additional details about the datasets and knowledge bases are listed in the Appendix A5.

MLLM baselines and SOTA methods. We perform evaluations by using primarily 7B parameter variants of LLaVA, InstructBLIP, Shikra, and MiniGPT-4, excluding larger or 3B models for fairer model size comparison. For benchmarking, we select several SOTA training-free decoding methods that aim to mitigate knowledge conflicts in LLMs: Contrastive Decoding (CD) [51], Adaptive Context-Aware Decoding (AdaCAD) [52], Entropy-based decoding (Entropy) [53], Context-Aware Decoding (CAD) [26] and COntextual Information-Entropy Constraint Decoding (COIECD) [54]. In addition, we also benchmark with two representative

hallucination mitigation methods, including Visual Contrastive Decoding (VCD) [58] and Assembly of Global and Local Attention (AGLA) [57].

Implementation details. For knowledge retrieval, we employ the vision encoder of CLIP-ViT-L/14-336 [66] as the retriever and append the first retrieved knowledge to the prompt as context. The scaling factor k is set to 0.4 to avoid excessive adjustment. For the knowledge base, we use Wikipedia dumps provided by [23] and select items with associated images for retrieval. Multinomial sampling serves as the decoding strategy. We denote MLLM inference with retrieved knowledge as Regular and without retrieved knowledge as Parametric. We follow prior studies [26, 53, 58] and adopt adaptive plausibility constraints [51] for fair comparisons. All experiments are conducted on four NVIDIA RTX 3090 GPUs for ALFAR, whereas all baseline methods are only executed on a single NVIDIA RTX 3090 GPU without officially reported hyperparameter adjustments.

5.2 Experimental Results

Experiments on free-form datasets. Tab. 2 shows experimental results of four representative MLLMs [14, 10, 40, 16] over two free-form generative knowledge-intensive datasets [23]. We can see that the proposed ALFAR consistently outperforms the Regular decoding strategy by substantial margins (averaged around 2.5% in overall accuracy) across all MLLMs and datasets. Additionally, ALFAR surpasses state-of-the-art decoding methods as well, demonstrating its effectiveness in the better utilization of contextual knowledge.

Experiments on multi-choice datasets. Tab. 3 presents experimental results of six MLLMs [14, 10, 40, 16, 62, 63] over two multi-choice discriminative datasets [32, 24]. Notably, ALFAR achieves an average improvement of 6.6% over Regular decoding and consistently surpasses state-of-the-art decoding strategies by substantial margins, underscoring its effectiveness in diverse tasks. Moreover, we observe that LLaVA-1.5 [14] demonstrates a stronger instruction-following capability compared to other models, enabling it to produce more correctly formatted outputs.

Experiments on knowledge-based datasets. In addition to entity knowledge-based datasets [32, 24], we conduct experiments on commonsense knowledge-based datasets, OK-VQA [64], AOK-VQA [65] and Encyclopedic VQA (E-VQA) [31] with LLaVA-1.5 [14]. As shown in Tab. 4, ALFAR surpasses Regular decoding by 15.2% and consistently outperforms state-of-the-art decoding strategies, underscoring its effectiveness in addressing a broader range of knowledge-intensive tasks.

| Model            | OK-VQA           | AOK-VQA          | E-VQA            |
| ---------------- | ---------------- | ---------------- | ---------------- |
| Regular [14]     | 46.17(±0.12)     | 44.13(±0.00)     | 19.14(±2.83)     |
| Parametric [14]  | 45.13(±0.40)     | 43.23(±1.02)     | 5.34(±0.01)      |
| CD [51]          | 55.00(±0.03)     | 51.67(±0.44)     | 28.62(±0.68)     |
| CAD [26]         | 56.43(±0.46)     | 53.93(±0.56)     | 28.62(±0.76)     |
| AdaCAD [52]      | 57.10(±0.04)     | 54.40(±0.48)     | 28.33(±0.42)     |
| Entropy [53]     | 56.27(±0.05)     | 53.93(±0.26)     | 29.24(±0.11)     |
| COIECD [54]      | 56.43(±0.46)     | 53.93(±0.56)     | 28.24(±0.74)     |
| VCD [58]         | 57.80(±0.13)     | 57.40(±0.12)     | 27.71(±0.33)     |
| AGLA [57]        | 57.53(±0.05)     | 55.40(±0.63)     | 28.29(±0.56)     |
| ALFAR (ours)     | 60.83(±0.01)     | 59.93(±0.02)     | 29.57(±0.08)     |


## Discussion


6 Discussion

6.1 Ablation study

We conduct ablation studies on both multi-choice and commonsense knowledge-based datasets [32, 65] to assess the effectiveness of each design in the proposed ALFAR model with LLaVA-1.5 [14]. As shown in Tab. 5, the Attention Reallocation enables MLLMs to better utilize retrieved knowledge, thereby enhancing overall performance. The Logits Fusion mitigates knowledge conflicts, allowing MLLMs to integrate retrieved knowledge and improve overall performance effectively. Moreover, applying Adaptive Weights during logits fusion helps MLLMs better leverage both parametric and contextual knowledge while reducing the impact of noise from the retrieved context, further contributing to performance gains.

| Variants                 | InfoSeek | AOK-VQA |
| ------------------------ | -------- | ------- |
| Regular [14]             | 51.97    | 44.13   |
| + Attention Reallocation | 53.42    | 46.30   |
| + Logits Fusion          | 55.83    | 55.90   |
| + Adaptive Weights       | 58.35    | 59.93   |

## 6.2 Effect of different retrievers

We investigate the impact of different retrievers and retrieval strategies on recall and model performance on the InfoSeek dataset [32]. As shown in Tab. 6, our model consistently enhances performance across all configurations, demonstrating its robustness to retrieval noise. Additionally, retrieving knowledge based on the input query yields low recall due to its limited information about entities in the image. Despite this limitation, our model achieves consistent performance improvements even under low retrieval recall, owing to its adaptive fusion strategy that balances parametric and contextual knowledge.

Table 6: Experimental results with different CLIP retrievers. “-T” means using the input query for knowledge retrieval.

| Retriever             | Recall | LLaVA | Ours      |
| --------------------- | ------ | ----- | --------- |
| CLIP-B/16 [66]        | 38.27  | 45.93 | **51.60** |
| CLIP-L/14 [66]        | 56.53  | 51.23 | **57.60** |
| CLIP-L/14-336 [66]    | 58.37  | 51.97 | **58.35** |
| CLIP-B/16-T [66]      | 5.56   | 32.97 | **35.43** |
| CLIP-L/14-T [66]      | 6.00   | 33.40 | **36.23** |
| CLIP-L/14-336-T [66]  | 5.73   | 33.37 | **36.63** |

## 6.3 Inference Efficiency

In this section, we conduct a detailed analysis of the inference efficiency of the proposed model in comparison with representative baseline methods. Specifically, we examine how the incorporation of parametric knowledge modeling affects the inference time under both discriminative and generative tasks. Compared with the baseline MRAG, our model introduces only a modest computational overhead, which mainly arises from the additional operations required to encode and integrate parametric knowledge. Nevertheless, this extra cost remains limited, as the parametric knowledge component without contextual input is relatively concise. Though ALFAR introduces limited inference overhead, scalability issues emerge with larger models, as demonstrated by the performance decline on InstructBLIP 13B in Tab. 8 and Fig. 6, likely caused by an increased dominance of parametric knowledge that ALFAR's fusion cannot adequately handle.

Table 7: Inference time of different methods over one sample with LLaVA-1.5 [14].

| Variants      | InfoSeek      | Human         |
| ------------- | ------------- | ------------- |
| Regular [14]  | 0.46s (1.00x) | 0.50s (1.00x) |
| CD [51]       | 0.62s (1.35x) | 0.72s (1.44x) |
| CAD [26]      | 0.62s (1.34x) | 0.72s (1.44x) |
| AdaCAD [52]   | 0.62s (1.35x) | 0.73s (1.45x) |
| Entropy [53]  | 0.63s (1.37x) | 0.73s (1.46x) |
| COIECD [54]   | 0.62s (1.35x) | 0.73s (1.45x) |
| VCD [58]      | 0.63s (1.38x) | 0.74s (1.47x) |
| AGLA [57]     | 0.83s (1.81x) | 1.02s (2.04x) |
| ALFAR (ours)  | 0.62s (1.35x) | 0.73s (1.46x) |

## 6.4 Qualitative Examples

Fig. 6 shows a qualitative comparison of four decoding approaches. It can be observed that Vanilla LLaVA [14] without context produces a false response due to the lack of knowledge about the entity in the image. LLaVA with MRAG [27] incorporates the contextual knowledge but still produces the same incorrect response as Vanilla LLaVA [14], primarily due to knowledge conflicts that cause MLLMs to favor their parametric knowledge. LLaVA with CAD [26] can mitigate knowledge conflicts effectively by reducing the influence of parametric knowledge. However, it produces a false response as well, mainly due to attention bias with excessive focus on images and uniform attention toward context. While ALFAR improves on baseline decoding methods, some generated responses still contain errors due to residual attention biases or poorly weighted contextual knowledge, indicating that complete mitigation of knowledge conflicts remains an open challenge.

## 6.5 ALFAR on MLLM Scalability

Tab. 8 presents experimental results of the 13B variants of LLaVA-1.5 [14] and InstructBLIP [10] over the InfoSeek dataset [32]. Notably, ALFAR consistently improves performance across both models, demonstrating its superior scalability with respect to the model size. Interestingly, we observe a performance decline in

Table 8: Experiments with larger MLLMs.

| Decoding     | LLaVA-13B | InstructBLIP-13B |
| ------------ | --------- | ---------------- |
| Parametric   | 44.40     | 8.93             |
| Regular      | 55.83     | 18.70            |
| ALFAR (ours) | **59.63** | **27.63**        |

Figure 6: Illustration of generated responses and corresponding attention distributions across different decoding methods, using LLaVA-1.5 [14] as the backbone model.
This effect is more pronounced in InstructBLIP 13B compared to the 7B variant, which may be attributed to the model’s increased reliance on parametric knowledge.

## 6.6 Effect of Different Decoding Strategies

Beyond the multinomial sampling strategy discussed in this paper, we further investigate the robustness of the proposed ALFAR framework under diverse decoding settings. To this end, we conduct experiments using LLaVA-1.5 [14] on the multi-choice InfoSeek dataset [32], and evaluate six additional decoding strategies commonly adopted in MLLM generation. These strategies include Top-P sampling [67] with p = 0.7, Top-K sampling [68] with k = 50, greedy decoding [69], temperature sampling [70] with t = 0.5, Top-P sampling with temperature (p = 0.7, t = 0.5), and Top-K sampling with temperature (k = 50, t = 0.5). As illustrated in Fig. 5, ALFAR consistently enhances model performance across all decoding strategies. This consistent gain highlights the generalizability of ALFAR’s learning principle and suggests that it effectively complements various decoding schemes. Moreover, the results indicate that ALFAR can serve as a robust and plug-and-play enhancement to existing MLLMs, ensuring stable performance regardless of decoding configuration.

| Decoding Strategy | Top-P | Top-K | Temperature | Top-P+Temp. | Top-K+Temp. | Greedy |
| ----------------- | ----- | ----- | ----------- | ------------ | ------------ | ------ |
| LLaVA + MRAG      | 54    | 53    | 55          | 57           | 56           | 57     |
| LLaVA + ALFAR     | 58    | 57    | 58          | 59           | 58           | 59     |

Figure 5: Experimental results with different decoding strategies for response generation.


## Conclusion


## 7 Conclusion

In this paper, we have proposed ALFAR, a method designed to improve knowledge utilization in MLLMs by addressing attention biases and knowledge conflicts. While experiments show promising gains, limitations including retrieval dependency, scalability concerns, and inference costs remain. Future work will focus on these challenges to enhance real-world applicability.


## Appendix


## 8 Acknowledgments

This work was supported by the National Science and Technology Major Project (2022ZD0117102), National Natural Science Foundation of China (62177038, 62293551, 62277042, 62377038), Project of China Knowledge Centre for Engineering Science and Technology, “LENOVO-XJTU” Intelligent Industry Joint Laboratory Project, The Youth AI Talents Fund of the Chinese Association of Automation under Major Program (HBRC-JKYZD-2024-311). No IRB or data consent statements were provided concerning the usage of knowledge base content, raising potential ethical transparency concerns.


![Figure 38](image_38.jpg)
<!-- Caption: Appendix, index 38 -->



![Figure 39](image_39.jpg)
<!-- Caption: Appendix, index 39 -->



## References


## References

[1] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.

[2] Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Xavier Garcia, Jason Wei, Xuezhi Wang, Hyung Won Chung, Dara Bahri, Tal Schuster, Steven Zheng, et al. UL2: Unifying language learning paradigms. In *The Eleventh International Conference on Learning Representations*, 2023.

[3] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *The Journal of Machine Learning Research*, 21(140):1–67, 2020.

[4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sasry, Amanda Askell, et al. Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33:1877–1901, 2020.

[5] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. PaLM: Scaling language modeling with pathways. *arXiv preprint arXiv:2204.02311*, 2022.

[6] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford Alpaca: An instruction-following LLaMA model, 2023.

[7] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing GPT-4 with 90% ChatGPT quality, March 2023.

[8] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. *arXiv preprint arXiv:2309.16609*, 2023.

[9] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A frontier large vision-language model with versatile abilities. *arXiv preprint arXiv:2308.12966*, 2023.

[10] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. InstructBLIP: Towards general-purpose vision-language models with instruction tuning. *arXiv preprint arXiv:2306.04387*, 2023.

[11] Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. Multimodal-GPT: A vision and language model for dialogue with humans. *arXiv preprint arXiv:2305.04790*, 2023.

[12] Wenbin An, Jiahao Nie, Yaqiang Wu, Feng Tian, Shijian Lu, and Qinghua Zheng. Towards external knowledge utilization in multimodal LLMs. *arXiv preprint arXiv:2308.10955*, 2023.

[13] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. *arXiv preprint arXiv:2305.03726*, 2023.

[14] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. *arXiv preprint arXiv:2304.08485*, 2023.

[15] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mPLUG-Owl: Modularization empowers large language models with multimodality. *arXiv preprint arXiv:2304.14178*, 2023.

[16] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. *arXiv preprint arXiv:2304.10592*, 2023.

[17] Duzhen Zhang, Yahan Yu, Jiahua Dong, Chenxing Li, Dan Su, Chenhui Chu, and Dong Yu. MM-LLMs: Recent advances in multimodal large language models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 12401–12430, 2024.

[18] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.

[19] Ahmad Sami Al-Shamayleh, Omar Adwan, Mohammad A Alsharaiah, Abdelrahman H Hussein, Qasem M Kharma, and Christopher Ifeanyi Eke. A comprehensive literature review on image captioning methods and metrics based on deep learning techniques. Multimedia Tools and Applications, 83(12):34219–34268, 2024.

[20] Shilin Sun, Wenbin An, Feng Tian, Fang Nan, Qidong Liu, Jun Liu, Nazaraf Shah, and Ping Chen. A review of multimodal explainable artificial intelligence: Past, present and future. arXiv preprint arXiv:2412.14056, 2024.

[21] Wenbin An, Feng Tian, Jiahao Nie, Wenkai Shi, Haonan Lin, Yan Chen, Qianying Wang, Yaqiang Wu, Guang Dai, and Ping Chen. Knowledge acquisition disentanglement for knowledge-based visual question answering with large language models. arXiv preprint arXiv:2407.15346, 2024.

[22] Jusung Lee, Sungguk Cha, Younghyun Lee, and Cheoljong Yang. Visual question answering instruction: Unlocking multimodal large language models to domain-specific visual multitasks. arXiv preprint arXiv:2402.08360, 2024.

[23] Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit Changpinyo, Alan Ritter, and Ming-Wei Chang. Can pre-trained vision and language models answer visual information-seeking questions? In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 14948–14968, 2023.

[24] Paul Lerner, Olivier Ferret, Camille Guinaudeau, Hervé Le Borgne, Romaric Besançon, José G. Moreno, and Jesús Lovón Melgarejo. ViQuAE: A dataset for knowledge-based visual question answering about named entities. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 3108–3120, 2022.

[25] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Retrieval augmented language model pre-training. In International Conference on Machine Learning, pages 3929–3938. PMLR, 2020.

[26] Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia Tsvetkov, Luke Zettlemoyer, and Wen-tau Yih. Trusting your evidence: Hallucinate less with context-aware decoding. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 783–791, 2024.

[27] Davide Caffagni, Federico Cocchi, Nicholas Moratelli, Sara Sarto, Marcella Cornia, Lorenzo Baraldi, and Rita Cucchiara. Wiki-LLaVA: Hierarchical retrieval-augmented generation for multimodal LLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1818–1826, 2024.

[28] Federico Cocchi, Nicholas Moratelli, Marcella Cornia, Lorenzo Baraldi, and Rita Cucchiara. Augmenting multimodal LLMs with self-reflective tokens for knowledge-based visual question answering. arXiv preprint arXiv:2411.16863, 2024.

[29] Yibin Yan and Weidi Xie. EchoSight: Advancing visual-language models with wiki knowledge. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 1538–1551, 2024.

[30] Amit Artzy and Roy Schwartz. Attend first, consolidate later: On the importance of attention in different LLM layers. In Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pages 177–184, 2024.

[31] Thomas Mensink, Jasper Uijlings, Lluis Castrejon, Arushi Goel, Felipe Cadar, Howard Zhou, Fei Sha, André Araujo, and Vittorio Ferrari. Encyclopedic VQA: Visual questions about detailed properties of fine-grained categories. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3113–3124, 2023.

[32] Tinghui Zhu, Qin Liu, Fei Wang, Zhengzhong Tu, and Muhao Chen. Unraveling cross-modality knowledge conflicts in large vision-language models. arXiv preprint arXiv:2410.03659, 2024.

[33] Mehrdad Farahani and Richard Johansson. Deciphering the interplay of parametric and non-parametric memory in retrieval-augmented language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 16966–16977, 2024.

[34] Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, and Sercan Ö. Arik. Astute RAG: Overcoming imperfect retrieval augmentation and knowledge conflicts for large language models. arXiv preprint arXiv:2410.07176, 2024.

[35] Justin Kruger and David Dunning. Unskilled and unaware of it: How difficulties in recognizing one's own incompetence lead to inflated self-assessments. Journal of Personality and Social Psychology, 77(6):1121–1134, 1999.

[36] Aniket Kumar Singh, Suman Devkota, Bishal Lamichhane, Uttam Dhakal, and Chandra Dhakal. The confidence-competence gap in large language models: A cognitive study. arXiv preprint arXiv:2309.16145, 2023.

[37] Weizhe Lin, Jingbiao Mei, Jinghong Chen, and Bill Byrne. PreFLMR: Scaling up fine-grained late-interaction multi-modal retrievers. arXiv preprint arXiv:2402.08327, 2024.

[38] Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen. UniIR: Training and benchmarking universal multimodal information retrievers. In European Conference on Computer Vision, pages 387–404. Springer, 2025.

[39] Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. ChatGPT outperforms crowd-workers for text-annotation tasks. arXiv preprint arXiv:2303.15056, 2023.

[40] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal LLM's referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023.

[41] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023.

[42] Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? arXiv preprint arXiv:2405.02246, 2024.

[43] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR, 2022.

[44] Xiaofeng Zhang, Yihao Quan, Chaochen Gu, Chen Shen, Xiaosong Yuan, Shaotian Yan, Hao Cheng, Kaijie Wu, and Jieping Ye. Seeing clearly by layer two: Enhancing attention heads to alleviate hallucination in LVLMs. arXiv preprint arXiv:2411.09968, 2024.

[45] Haonan Lin, Mengmeng Wang, Yan Chen, Wenbin An, Yuzhe Yao, Guang Dai, Qianying Wang, Yong Liu, and Jingdong Wang. DreamSalon: A staged diffusion framework for preserving identity-context in editable face generation. arXiv preprint arXiv:2403.19235, 2024.

[46] Xiaofeng Zhang, Chen Shen, Xiaosong Yuan, Shaotian Yan, Liang Xie, Wenxiao Wang, Chaochen Gu, Hao Tang, and Jieping Ye. From redundancy to relevance: Enhancing explainability in multimodal large language models. arXiv e-prints, pages arXiv–2406, 2024.

[47] Jiahao Nie, Gongjie Zhang, Wenbin An, Yap-Peng Tan, Alex C. Kot, and Shijian Lu. MMRel: A relation understanding dataset and benchmark in the MLLM era. arXiv preprint arXiv:2406.09121, 2024.

[48] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023.

[49] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.

[50] Tao Zhang, Ziqi Zhang, Zongyang Ma, Yuxin Chen, Zhongang Qi, Chunfeng Yuan, Bing Li, Junfu Pu, Yuxuan Zhao, Zehua Xie, et al. MR2AG: Multimodal retrieval-reflection-augmented generation for knowledge-based VQA. arXiv preprint arXiv:2411.15041, 2024.

[51] Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori B. Hashimoto, Luke Zettlemoyer, and Mike Lewis. Contrastive decoding: Open-ended text generation as optimization. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12286–12312, 2023.

[52] Han Wang, Archiki Prasad, Elias Stengel-Eskin, and Mohit Bansal. AdaCAD: Adaptively decoding to balance conflicts between contextual and parametric knowledge. arXiv preprint arXiv:2409.07394, 2024.

[53] Zexuan Qiu, Zijing Ou, Bin Wu, Jingjing Li, Aiwei Liu, and Irwin King. Entropy-based decoding for retrieval-augmented large language models. arXiv preprint arXiv:2406.17519, 2024.

[54] Xiaowei Yuan, Zhao Yang, Yequan Wang, Shengping Liu, Jun Zhao, and Kang Liu. Discerning and resolving knowledge conflicts through adaptive decoding with contextual information-entropy constraint. arXiv preprint arXiv:2402.11893, 2024.

[55] Yanwen Huang, Yong Zhang, Ning Cheng, Zhitao Li, Shaojun Wang, and Jing Xiao. Dynamic attention-guided context decoding for mitigating context-faithfulness hallucinations in large language models. arXiv preprint arXiv:2501.01059, 2025.

[56] A. Vaswani. Attention Is All You Need. Advances in Neural Information Processing Systems, 2017.

[57] Wenbin An, Feng Tian, Sicong Leng, Jiahao Nie, Haonan Lin, Qianying Wang, Guang Dai, Ping Chen, and Shijian Lu. AGLA: Mitigating object hallucinations in large vision-language models with assembly of global and local attention. arXiv preprint arXiv:2406.12718, 2024.

[58] Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. arXiv preprint arXiv:2311.16922, 2023.

[59] Shiyu Zhao, Zhenting Wang, Felix Juefei-Xu, Xide Xia, Miao Liu, Xiaofang Wang, Mingfu Liang, Ning Zhang, Dimitris N. Metaxas, and Licheng Yu. Accelerating multimodal large language models by searching for optimal vision token reduction. arXiv preprint arXiv:2412.00556, 2024.

[60] Hila Chefer, Shir Gur, and Lior Wolf. Transformer interpretability beyond attention visualization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 782–791, 2021.

[61] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. SnapKV: LLM knows what you are looking for before generation. arXiv preprint arXiv:2404.14469, 2024.

[62] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved Baselines with Visual Instruction Tuning, 2023.

[63] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923, 2025.

[64] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. OK-VQA: A visual question answering benchmark requiring external knowledge. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 3195–3204, 2019.

[65] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-OKVQA: A benchmark for visual question answering using world knowledge. In *European Conference on Computer Vision*, pages 146–162. Springer, 2022.

[66] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International Conference on Machine Learning*, pages 8748–8763. PMLR, 2021.

[67] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. *arXiv preprint arXiv:1904.09751*, 2019.

[68] Angela Fan, Mike Lewis, and Yann Dauphin. Hierarchical neural story generation. *arXiv preprint arXiv:1805.04833*, 2018.

[69] Ronald A DeVore and Vladimir N Temlyakov. Some remarks on greedy algorithms. *Advances in Computational Mathematics*, 5(1):173–187, 1996.

[70] David H Ackley, Geoffrey E Hinton, and Terrence J Sejnowski. A learning algorithm for Boltzmann machines. *Cognitive Science*, 9(1):147–169, 1985.

[71] OpenAI. GPT-4 technical report, 2023. Unpublished internal document.

[72] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35:27730–27744, 2022.

[73] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The iNaturalist species classification and detection dataset. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 8769–8778, 2018.

[74] Feilong Tang, Zile Huang, Chengzhi Liu, Qiang Sun, Harry Yang, and Ser-Nam Lim. Intervening anchor token: Decoding strategy in alleviating hallucinations for MLLMs. In *The Thirteenth International Conference on Learning Representations*, 2025.

[75] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. *URL https://arxiv.org/abs/2307.03172*, 2023.


## Appendix


## A1 Limitation and Future Work

Despite its strong performance in enhancing knowledge utilization for MLLMs through adaptive logits fusion and attention reallocation, our approach has certain limitations. First, our framework requires access to MLLM parameters, making it inapplicable to black-box API-based models such as GPT-4 [71]. Extending our framework to black-box MLLMs represents a promising direction for future research. Additionally, we observe that MLLMs struggle to effectively extract relevant information from long contexts. Addressing this limitation by improving MLLMs' ability to leverage extended contexts will be another focus of future work.

## A2 Broader Impacts

The proposed model for enhancing knowledge utilization in MLLMs carries significant broader impacts. First, by addressing the critical issue of MRAG, our method enhances the reliability and trustworthiness of MLLMs. This improvement is essential for deploying these models in sensitive and high-stakes applications such as autonomous driving, medical diagnostics, and surveillance systems. Second, the insights and methods introduced in this paper contribute to the broader field of MLLMs, particularly in understanding and improving the knowledge utilization mechanisms within MLLMs. This advancement can spur further research and innovation in integrating visual and textual data, leading to more robust and versatile AI models.

## A3 Conflict Rate and Performance Drop

To quantify the conflict between parametric and contextual knowledge and its impact on model performance, we introduce two metrics: Conflict Rate and Performance Drop.

Conflict Rate measures the proportion of instances where parametric and contextual knowledge provide different information, and Performance Drop quantifies the decline in model performance due to knowledge conflict. Since parametric knowledge is implicitly embedded in model parameters and is not directly observable, we approximate its correctness by evaluating the model's outputs. Specifically, if the model (without external context) produces the correct answer, we assume its parametric knowledge is correct; otherwise, it is considered incorrect. Given access to ground-truth contextual knowledge, the Conflict Rate can be defined as the error rate of parametric knowledge, i.e., the proportion of incorrect responses generated by the vanilla model without input context:

$$\text{Conflict Rate} = \text{Err}(\mathcal{M}_\theta(y|q, I), \hat{y})$$ (A1)

where Err is a function that calculates the error rate of the output, $$\mathcal{M}_\theta(y|q, I)$$ is the output with only images and questions as inputs, and $$\hat{y}$$ is the ground-truth answer.

When correct contextual knowledge is available, the ideal model should achieve 100% accuracy in the absence of knowledge conflicts. However, influenced by knowledge conflicts, the model cannot achieve 100% accuracy, then we can define Performance Drop as the error rate of outputs when both parametric and contextual knowledge are used:

$$\text{Performance Drop} = \text{Err}(\mathcal{M}_\theta(y|q, I, c), \hat{y})$$ (A2)

where $$\mathcal{M}_\theta(y|q, I, c)$$ is the output with ground-truth context as additional inputs.

## A4 Retrieval Recall

To investigate the retrieval recall from different retrieval rankings, we present the recall with Ground-Truth knowledge and knowledge from various retrieval rankings on the multi-choice InfoSeek dataset [23, 32] in Tab. A1. The low recall negatively impacts performance on knowledge-intensive VQA tasks, highlighting the necessity of developing a more effective retriever.

## A5 Dataset

We present statistics of different datasets and the corresponding knowledge bases in Tab. A2. Specifically, for Validation [23] and InfoSeek [32], we follow previous works [27] and adopt a knowledge

Table A1: Retrieval recall with Ground-Truth knowledge (GT) and knowledge from different retrieval rankings on the multi-choice InfoSeek dataset [23, 32].

| Index  | GT  | 1     | 2     | 3    | 4    |
| ------ | --- | ----- | ----- | ---- | ---- |
| Recall | 100 | 58.37 | 10.57 | 5.07 | 3.07 |

Table A2: Statistics of the datasets and details of the knowledge bases used.

| Dataset          | # VQA pairs | Knowledge Base     |
| ---------------- | ----------- | ------------------ |
| Validation [23]  | 73,620      | Wikipedia [23]     |
| Human [23]       | 8,931       | Wikipedia [23]     |
| InfoSeek [32]    | 3,000       | Wikipedia [23]     |
| ViQuAE [24]      | 3,000       | Wikipedia [23]     |
| OK-VQA [64]      | 5,046       | GPT-3.5 [21]       |
| AOK-VQA [65]     | 1,145       | GPT-3.5 [21]       |
| E-VQA [31]       | 700         | Encyclopedia [31]  |

We construct a knowledge base containing 1.7K entities derived from the original Wikipedia knowledge base [23]. For Human [23] and ViQuAE [24], we use the original Wikipedia knowledge base [23], selecting 73.6K entities accompanied by images for knowledge retrieval. For OK-VQA [64] and AOK-VQA [65], we utilize the knowledge base provided by [21], which was generated using GPT-3.5 [72]. For E-VQA [31], we select templated questions with images from the iNaturalist dataset [73] and use the corresponding ground-truth knowledge for inference. All evaluations are conducted using the official scripts.

A6 Adaptive Plausibility Constraints

We follow prior studies [26, 53, 58] and adopt adaptive plausibility constraints [51] for fair comparisons. Specifically, calibrating the entire output distribution may penalize valid outputs from the original distribution and promote implausible outputs from the modified distribution. To mitigate this issue, we selectively consider tokens with high original probabilities and truncate other tokens as follows:

V_token(y_<i) = { y_i ∈ V : p_θ(y_i) ≥ β · max_w p_θ(w) }

p(y_i) = 0, if y_i ∉ V_token(y_<i) (A3)

where V_token is the set of selected tokens and V is the output vocabulary. We select β = 0.7 to retain only high-probability tokens.

A7 Effect of Intervention Layers

We investigate the impact of attention reallocation at different MLLM layers on the InfoSeek dataset [23] using LLaVA-1.5 [14], as summarized in Tab. A4. The results show that reallocating attention in shallow layers (layers 1–16) enhances model performance by mitigating attention bias toward image tokens, thereby improving the extraction of low-level features [30]. In contrast, applying attention reallocation in middle layers (layers 17–24) yields smaller gains, as these layers primarily handle multimodal alignment and feature aggregation [74], where attention bias is less severe. Notably, reallocating attention in late layers (layers 25–32) leads to the most substantial performance gains, as these layers are responsible for reasoning and directly affect output generation [74]. Furthermore, leveraging attention reallocation in both shallow (layers 1–8) and deep (layer 32) layers yields the best performance.

Table A4: An ablation study with different layers for attention reallocation.

| Intervention Layers | VQA Accuracy |
| ------------------- | ------------ |
| None                | 56.70        |
| [1 − 8]             | 58.08        |
| [9 − 16]            | 58.17        |
| [17 − 24]           | 57.57        |
| [25 − 32]           | 58.10        |
| [1 − 8] ∪ [32]      | 58.67        |

A8 Effect of different numbers of knowledge

We examine the effect of varying the amount of knowledge provided to MLLMs on retrieval recall and model performance on the InfoSeek dataset [23]. As shown in Tab. A3, appending additional knowledge to the prompt improves retrieval recall but has limited impact on model performance, as MLLMs often struggle to effectively utilize information from lengthy input contexts [75]. Our model addresses this limitation by guiding MLLMs based on query-context relevance. However, the modest performance gains underscore the need for future research on enhancing MLLMs’ ability to process and leverage extended contexts.

Table A3: Experimental results with different numbers of knowledge using LLaVA-1.5 [14].

| #Knowledge | Recall | LLaVA [14] | Ours |
| ---------- | ------ | ---------- | ---- |
| 1          | 58.37  | 51.97      | 58.35 |
| 2          | 68.93  | 49.90      | 58.70 |
| 3          | 74.00  | 50.13      | 58.57 |
| 4          | 76.77  | 50.23      | 58.20 |