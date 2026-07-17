
## Abstract


KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction

Jang-Hyun Kim^1,2, Jinuk Kim^1,2, Sangwoo Kwon^1, Jae W. Lee^1, Sangdoo Yun^3, Hyun Oh Song^1,2*

^1 Seoul National University, ^2 Neural Processing Research Center, ^3 NAVER AI Lab
{blue378, hyunoh}@snu.ac.kr
https://github.com/snu-mllab/KVzip

Abstract
Transformer-based large language models (LLMs) cache context as key–value (KV) pairs during inference. As context length grows, KV cache sizes expand, leading to substantial memory overhead and increased attention latency. This paper introduces KVzip, a query-agnostic KV cache eviction method enabling effective reuse of compressed KV caches across diverse queries. KVzip quantifies the importance of a KV pair using the underlying LLM to reconstruct original contexts from cached KV pairs, subsequently evicting pairs with lower importance. Empirical evaluations indicate that KVzip achieves KV cache size reductions by 2–3× and FlashAttention decoding latency improvements up to 1.5×, with moderate performance tradeoffs in various tasks such as question answering and code comprehension. Evaluations span select models including LLaMA 3.1 and Qwen 2.5 with context lengths up to 100K tokens. KVzip offers promising improvements over several query-aware baseline methods, though some degradation is observed under aggressive compression ratios in multi-query settings.


## Introduction


1 Introduction
Transformer-based LLMs with long-context capabilities have significantly enhanced real-world applications, including long-document analysis and personalized conversational agents [1, 21, 49]. However, increasing context lengths substantially raises both memory consumption for KV caching and computational costs associated with attention mechanisms [31]. For example, caching 120K tokens in Qwen 2.5-14B with FP16 precision requires approximately 33 GB memory, surpassing the model’s 28 GB parameter storage at equivalent precision [54].

Recent approaches primarily target reducing KV cache memory size while preserving inference accuracy. These methods include merging the attention heads [3], compressing KV pairs into shorter sequences [46], and using sliding-window techniques to limit context windows [24, 52, 53]. Other studies exploit attention sparsity for dynamic KV eviction during decoding [4, 38, 60] and prefill stages [6, 33]. Existing eviction methods typically employ query-aware KV-pair importance scoring computed online during inference [6, 33, 60], selectively retaining KV pairs most relevant to immediate queries (Figure 1a, b). While effective in single-query scenarios, these methods exhibit significant performance degradation in multi-query settings, as the retained KV pairs predominantly overfit to initial queries [35]. We elaborate on these limitations in Section 2.2.

We propose KVzip, a query-agnostic KV cache eviction approach aimed at optimizing reusable compressed KV caches for inference across diverse queries (Figure 1c). KVzip targets offline preparation scenarios including personalized conversational agents and enterprise retrieval systems, though challenges in generalization and overhead remain active areas for investigation.

*Corresponding author

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

(a) Query-aware KV eviction | (b) Reusing query-dependent cache | (c) Proposed framework

CTX Q₁ --prefill + evict--> KV₁ --decode--> A₁
     ⋮                       ⋮               ⋮
CTX Qₙ                      KVₙ             Aₙ
✗ Repetitive prefill. ✓ Good performance.

CTX Q₁ --prefill + evict--> KV₁ --decode--> A₁
                             Q₂ ---------> A₂
                             Qₙ ---------> Aₙ
✓ One-time prefill.
✗ Low performance.

CTX --prefill + evict--> KV --Q₁ decode--> A₁
                             ⋮              ⋮
                             Qₙ ---------> Aₙ
✓ One-time prefill.
✓ Good performance.

Figure 1: Overview of KV eviction strategies in multi-query scenarios. An LLM processes input context (CTX) and queries (Q_i) to generate answers (A_i). Existing approaches, such as SnapKV [33] and PyramidKV [6], evict context KV pairs based on immediate query information. (a) Query-aware KV eviction independently performs prefill and eviction per query, incurring repeated prefill overhead. (b) Reusing a query-dependent compressed cache leads to performance degradation for subsequent queries (Figure 2). (c) The proposed query-agnostic KV eviction framework compresses the KV cache only once during the initial prefill, enabling efficient reuse across diverse queries without repeated prefill or performance loss. Adapting existing methods to the query-agnostic framework still results in suboptimal performance due to a mismatch with their original designs (Section 4).

Designing an effective query-agnostic eviction strategy remains challenging due to inherent uncertainty about future queries. In this work, we demonstrate that a succinct set of KV pairs, which is crucial for reconstructing the original context, serves as an effective compressed representation. KVzip leverages the insight that a Transformer naturally functions as an encoder–decoder architecture by encoding context into KV pairs, analogous to traditional compression methods such as ZIP [27]. Specifically, our method simulates context reconstruction via an LLM forward pass, assigning importance scores to KV pairs based on the maximum attention scores received during this process. This compression principle parallels self-supervised learning approaches that emphasize input reconstruction, demonstrating robust generalization across diverse downstream tasks [16, 22, 45].

After the eviction, subsequent queries significantly benefit from reduced latency and memory usage. Specifically, KVzip achieves approximately 2× latency reduction in FlashAttention [15] and 3–4× reduction in KV cache size during decoding with negligible performance loss on diverse queries. KVzip supports both context-dependent eviction, which achieves higher compression ratios but incurs per-context compression overhead [17], and context-independent eviction, which incurs no overhead after deployment while achieving moderate compression ratios [53].

Section 4 empirically demonstrates KVzip’s robustness and effectiveness on multiple benchmarks, including document question answering, mathematical reasoning, retrieval, and code comprehension tasks, with contexts up to 170K tokens. Existing query-aware eviction methods often face performance degradation with aggressive KV cache evictions, particularly in multi-query settings [33, 60]. KVzip aims to alleviate such degradation by adopting a query-agnostic approach, though empirical results indicate that performance varies substantially across tasks and compression ratios. Our evaluations involve 12 diverse benchmark datasets and models ranging from 3B to 14B parameters, illustrating both strengths and limitations of our approach. Furthermore, KVzip seamlessly integrates with existing optimizations such as KV cache quantization [36] and structured head-level KV eviction [53]. Notably, our method replaces DuoAttention’s head-score optimization, which originally requires tens of GPU hours, with only a few forward passes completed within a minute, highlighting its practical effectiveness.


## Preliminaries


2 Preliminary

2.1 Notation and Problem Formulation

Consider the text domain T and an autoregressive Transformer-based LLM f_LM : T → T that generates sequences via greedy decoding [44, 50]. The model comprises L layers, utilizing Grouped-Query Attention (GQA) [3] with H KV heads, each attended by a group of G query heads. During inference, f_LM caches hidden representations as KV pairs to enhance computational efficiency [31].

Given an input context c ∈ T tokenized into n_c tokens, the prefill stage generates a cache containing L × H × n_c KV pairs, denoted as KV_c [2]. Conditioned generation using the cache is denoted as f_LM(· | KV_c). Our objective is to derive a compact pruned cache KV_{c,evicted} ⊆ KV_c satisfying
f_LM(q | KV_{c,evicted}) ≈ f_LM(q | KV_c), ∀ q ∈ T. (1)


![Figure 7](image_7.jpg)
<!-- Caption: Preliminaries, index 7 -->



![Figure 8](image_8.jpg)
<!-- Caption: Preliminaries, index 8 -->



![Figure 9](image_9.jpg)
<!-- Caption: Preliminaries, index 9 -->


## 2.2 Analysis of Existing Approaches

Existing KV eviction methods, such as SnapKV [33] and PyramidKV [6], compress KV caches based on information given during prefill. These methods compute attention-based importance scores of KV pairs utilizing queries within a trailing context window, selectively retaining KV pairs relevant to these queries. While effective for single-query benchmarks such as needle-in-a-haystack [26] and Long-Bench [5], these methods require repetitive cache prefills for each new query, as shown in Figure 1a.

Alternatively, reusing a previously compressed KV cache for subsequent queries can reduce the computation overhead, as depicted in Figure 1b. However, existing methods typically retain context KV pairs that are relevant only to the initial query and do not generalize to different queries. Figure 2 illustrates this issue using the SQuAD multi-QA dataset [47]. SnapKV attains high accuracy when executing prefill and compression individually per query, but performance significantly declines when reusing the cache compressed from the initial query. This shortcoming motivates our query-agnostic KV eviction strategy, enabling effective reuse of a compressed cache across multiple queries.

<div style="text-align: center;">
<img src="figure2" alt="Accuracy graph showing SnapKV-prefill, SnapKV-reuse, and KVzip performance across KV cache budget ratios from 0.2 to 1.0, with accuracy ranging from 40% to 100%">
</div>

Figure 2: Accuracy on SQuAD using LLaMA3.1-8B. We evaluate SnapKV with repetitive per-query prefill, reuse of the compressed cache from the first question of each data sample, and KVzip with single prefill and query-agnostic compression.


## Method


## 3 Method

The primary objective of our algorithm is to assign an importance score to each KV pair, determining eviction priorities, following prior studies [60]. Given a context length $$n_c$$, KVzip assigns importance scores $$S \in \mathbb{R}^{L \times H \times n_c}$$ to KV pairs in $$\text{KV}_c$$, subsequently evicting pairs with the lowest scores. Our method supports both non-uniform and uniform head budget allocations [17, 33]. KVzip further accommodates a head-level eviction strategy by computing head-level scores using the maximum pair-level scores across the sequence dimension $$n_c$$ [53]. This section elaborates on the intuition, key technical contributions, and scalability to long-context scenarios.

### 3.1 Intuition

To effectively answer arbitrary queries, the compressed cache $$\text{KV}_{c,\text{evicted}}$$ and $$f_{\text{LM}}$$ should retain complete contextual information. Our intuition is that we can verify this completeness by explicitly prompting $$f_{\text{LM}}$$ to reconstruct the previous context from $$\text{KV}_{c,\text{evicted}}$$ (Figure 3). If $$\text{KV}_{c,\text{evicted}}$$ enables $$f_{\text{LM}}$$ to accurately reconstruct the original context $$c$$ using the repeat prompt, we can re-prefill the original cache $$\text{KV}_c$$ and conduct accurate inference.

<div style="text-align: center;">
<img src="figure3" alt="Diagram showing transformer LLM as context encoder-decoder with KVc, KVc,evicted, prefill, evict, repeat prompt, and decode stages">
</div>

Figure 3: Transformer LLM viewed as a context encoder-decoder. Each matrix cell indicates a KV pair. We use the prompt "Repeat the previous context:".

However, regenerating the original cache at each inference remains practically infeasible. Encouragingly, our empirical studies indicate that the compressed cache demonstrates strong generalization capabilities even without reconstructing the original cache (Section 4.2), empirically achieving Equation (1). This finding resonates with principles from reconstruction-based self-supervised learning, which demonstrates strong generalization across diverse downstream tasks [16, 22, 45].

<div style="text-align: center;">
<img src="figure4" alt="Method overview diagram showing the KVzip process: Prefill, KV importance measurement, eviction, and decoding stages">
</div>

Figure 4: Method overview. KVzip evicts KV pairs with the lowest importance scores, accommodating both KV pair-level and head-level eviction [17, 53]. System prompts are omitted for clarity.


![Figure 13](image_13.jpg)
<!-- Caption: Method, index 13 -->



![Figure 14](image_14.jpg)
<!-- Caption: Method, index 14 -->



![Figure 15](image_15.jpg)
<!-- Caption: Method, index 15 -->



![Figure 16](image_16.jpg)
<!-- Caption: Method, index 16 -->



![Figure 17](image_17.jpg)
<!-- Caption: Method, index 17 -->


## 3.2 KV Importance Scoring

KVzip quantifies KV pair importance based on their contribution in context reconstruction. Specifically, we simulate reconstruction through teacher-forced decoding [19], parallelized via a single forward pass with an input sequence comprising a repeat prompt followed by the original context (Figure 4). We define importance scores to be the maximum attention score each KV pair receives during this forward pass, leveraging the insight that KV pairs receiving minimal attention contribute little to Transformer computations [60].

Formally, given a context of length $n_c$, we construct an input sequence of length $n_{\text{in}} = n_{\text{prompt}} + n_c$ by concatenating the repeat prompt with the context. Forwarding this input through $f_{\text{LM}}$ with $KV_c$ produces query features $Q_{l,h} \in \mathbb{R}^{G \times (n_c + n_{\text{in}}) \times d}$ and key features $K_{l,h} \in \mathbb{R}^{n_{\text{in}} \times d}$ for the $h$-th KV head in layer $l$. The attention matrix $A_{l,h} = \text{Softmax}(Q_{l,h} K_{l,h}^\top)$ then lies in $\mathbb{R}_+^{G \times (n_c + n_{\text{in}}) \times n_{\text{in}}}$. Extracting slices relevant to $KV_c$ yields $\bar{A}_{l,h} \in \mathbb{R}_+^{G \times n_c \times n_{\text{in}}}$. Importance scores $S_{l,h} \in \mathbb{R}_+^{n_c}$ are computed by maximizing over the grouped query and input dimensions as:

$$S_{l,h}[j] = \max_{g=1,\dots,G;\; i=1,\dots,n_c} \bar{A}_{l,h}[g, j, i].$$ (2 corrected)

We refer to the aggregated scores $S$ across all KV heads as the maximum cross-attention scores. Figure 13 provides a visualization of these scores.

## 3.3 Observation

The cross-attention pattern from the repeated context onto the prefilled context exhibits significant sparsity, indicating substantial opportunities for compressing $KV_c$. Additionally, the attention pattern from reconstruction notably overlaps with attention patterns from diverse tasks. Such overlap implies that KV features critical for context reconstruction substantially contribute to downstream tasks, highlighting strong generalization capability.

Attention Sparsity in Reconstruction. Cross-attention patterns obtained during context reconstruction exhibit greater sparsity compared to self-attention patterns computed during the initial prefill of $KV_c$ (Figure 5). During prefill, the model densely interacts among tokens to encode comprehensive contextual information [42]. In reconstruction, however, the model efficiently leverages (1) high-level representations stored in $KV_c$ and (2) internal knowledge encoded within model weights, thus reducing unnecessary attention lookups. This cross-attention sparsity effectively identifies and removes redundant KV pairs, outperforming prior methods such as H₂O [60] that rely on attention scores obtained during prefill (Section 4.2).

Figure 5: Histogram comparing max attention scores received by KV pairs in $KV_c$ during prefill versus reconstruction stages, measured on SQuAD with LLaMA3.1-8B.

Figure 6: Attention comparison across tasks. 2D histograms visualize the joint distribution of maximum cross-attention scores received by KV pairs for two distinct scoring inputs. Each input consists of a task query and the generated response (Table 3). Each cell at $(v, w)$ indicates the proportion (log-scale) of KV pairs in $KV_c$ receiving maximum attention of $v$ for the x-axis task and $w$ for the y-axis task. Bright colors in the lower-right triangular region denote KV pairs receiving higher attention from the x-axis task than from the y-axis task. We compute scores using LLaMA3.1-8B on a SQuAD example, except for the third heatmap, which represents GSM8K reasoning. QA-1 and QA-2 denote distinct QA pairs. Figure 13 visualizes the attention patterns for each task.


![Figure 20](image_20.jpg)
<!-- Caption: Method, index 20 -->



![Figure 21](image_21.jpg)
<!-- Caption: Method, index 21 -->



![Figure 22](image_22.jpg)
<!-- Caption: Method, index 22 -->



![Figure 23](image_23.jpg)
<!-- Caption: Method, index 23 -->



![Figure 24](image_24.jpg)
<!-- Caption: Method, index 24 -->


Figure 7: Chunked scoring for the i-th chunk in KV_c. We compute attention scores by multiplying queries with subsampled keys of length m + n_in, followed by softmax normalization. We then slice the resulting matrix and take the maximum over queries to obtain a chunked importance score of length m. We set the grouped-query size to G = 1 for clarity. This procedure repeats per chunk. For chunks with i ≥ 2, we formulate the repeat prompt as: "Repeat the previous context starting with ⟨last 8 tokens of preceding chunk⟩:". Appendix C.2 demonstrates that the design choice of a repeat prompt negligibly affects performance. Pseudo-code is provided in Appendix A, Algorithm 1.

Attention Overlap Across Tasks. Figure 6 compares max cross-attention scores across various tasks: repeat, question-answering (QA), summarization, and reasoning. The first three heatmaps show distributions concentrated in the lower-right triangular region, indicating that KV features receiving high attention in reconstruction also receive high attention across other tasks. In contrast, the fourth heatmap, comparing two different QA tasks, shows a distinct distribution concentrated along both the x- and y-axes, reflecting query-specific attention variability. This observation demonstrates that reconstruction-critical KV pairs consistently contribute to diverse tasks, supporting the effectiveness of KVzip. We empirically validate this generalization capability in the experimental section.

3.4 Technical Challenge and Solution

Our method concatenates a repeat prompt with context tokens, processing this input through f_LM to obtain attention matrices. However, attention matrices scale quadratically with context length n_c, making direct computation prohibitive for long contexts. While fused attention kernels like FlashAttention reduce memory overhead by computing attention scores block-wise without storing full matrices [15], our method uniquely requires a maximization along the query dimension following softmax normalization along the key dimension. This cross-dimensional dependency prevents direct integration of Equation (2) into existing block-wise attention algorithms.

Chunked Scoring. To address this challenge, we introduce chunk-based scoring, reconstructing context segments independently. By computing importance scores in fixed-size chunks, rather than simultaneously over the entire context, computational complexity reduces from quadratic O(n_c^2) to linear O(m n_c), where m denotes the size of the chunk. Specifically, we partition the context tokens into fixed-length chunks of size m, concatenate each chunk with the repeat prompt, and process the resulting input of length n_in = n_prompt + m through f_LM (Figure 7). For each Transformer layer, we subsample keys in KV_c corresponding to each chunk, obtaining a smaller attention matrix of size n_in × (m + n_in). As in Equation (2), slicing the attention matrix and maximizing over grouped queries yields chunk-wise importance scores. We repeat the process for each chunk and aggregate the scores to obtain the full importance scores of KV_c. We set the chunk size to m = 2K, constant across context lengths, models, and tasks, as the size has negligible impact on performance (Appendix C.1).

Complexity Analysis. Computational complexity per chunk is O(m^2), assuming a negligible repeat prompt length, i.e., n_prompt ≪ m, thus n_in ≈ m. Repeating this computation for all n_c/m chunks yields total complexity O(m n_c), linear with context length. Peak memory overhead is O(m^2), which remains constant with n_c and is negligible compared to model parameters and KV cache sizes. Additionally, we propose a softmax-free variant in Appendix C.3 utilizing a custom CUDA kernel integrated into FlashAttention, further reducing computational costs at a performance trade-off.

Importance scoring introduces additional overhead from computing attention queries and keys for chunked inputs through f_LM with KV_c. Given n_in ≈ m, FlashAttention incurs O(n_c m + m^2/2) causal-attention FLOPs per chunk, resulting in a total complexity of O(n_c^2 + n_c m/2) across all n_c/m chunks. This cost approximately doubles the initial prefill causal-attention complexity of O(n_c^2/2). Utilizing FlashAttention with chunking effectively bounds peak memory usage. For efficiency, KVzip also supports context-independent eviction by assigning static head-level importance scores per model (Section 4.2–Figure 11), incurring no compression overhead after deployment.


![Figure 30](image_30.jpg)
<!-- Caption: Method, index 30 -->


(a) Inference efficiency (decoding)

Attention latency (ms)
KV cache ratio: 0.2, 0.4, 0.6, 0.8, 1.0
Values: 0.17, 0.22, 0.27, 0.34, 0.39

KV memory (GB)
KV cache ratio: 0.2, 0.4, 0.6, 0.8, 1.0
Values: 3.3, 6.5, 9.8, 13.1, 16.3

(b) Compression overhead

Compute time (s)
Repeat chunk size: 0.5k, 1k, 2k, 4k, 8k
Values: 160.0, 140.2, 130.5, 145.8, 165.3

Peak memory (GB)
Repeat chunk size: 0.5k, 1k, 2k, 4k, 8k
Values: 48.3, 50.1, 52.7, 55.0, 62.3

Figure 8: Computational analysis using LLaMA3.1-8B with 124K context tokens on an NVIDIA A100 GPU in FP16 precision. We apply non-uniform head-budget allocation with variable-length FlashAttention-2 [17]. (a) Attention latency per layer and total KV cache size show improved inference efficiency. (b) KV importance scoring overhead aggregated over all chunks. Dashed horizontal lines indicate the initial prefill cost for reference, with a 2K chunk size limiting peak memory for a fair comparison [2]. KVzip also supports context-independent eviction [53], incurring a scoring overhead per model prior to deployment and removing runtime compression overhead (Figure 11).

Empirical Efficiency Analysis. Empirical evaluations on an NVIDIA A100 GPU in Figure 8 indicate that KVzip introduces roughly twice the computational cost of standard prefill during compression, resulting in nontrivial overhead. While subsequent decoding latency benefits are observed, these improvements partially offset the upfront compression cost, complicating overall efficiency claims. Peak memory usage increases moderately during scoring, challenging assertions of negligible overhead. Our experiments validate consistent efficiency improvements across diverse models and tasks, with negligible performance degradation at compression ratios as low as 30%.


## Experiments


4 Experiment

4.1 Setup

Eviction Structure. We apply non-uniform head-budget allocation exclusively to KVzip, retaining KV pairs with the top r% importance scores. Baseline methods are evaluated using their originally recommended uniform budget settings to reflect typical usage. We acknowledge that this results in asymmetry but argue it reflects baseline best practices. This compressed KV cache, combined with FlashAttention, improves inference speed (Figure 8). Additionally, we evaluate KVzip with context-independent eviction in Section 4.2 and uniform-budget allocation in Appendix C.4.

Evaluation. Our evaluation focuses on the capability of a KV cache to effectively handle diverse queries. Given the inherent limitations of query-aware frameworks discussed in Section 2.2, we adopt the query-agnostic framework from Figure 1c. Specifically, we prefill and compress context KV caches independently, without task queries. Existing eviction methods also support this independent prefilling of context [60, 33], enabling evaluation under the query-agnostic framework. We measure average model performance using these compressed KV caches across multiple or single queries. Since the compression is query-agnostic, even single-query evaluations meaningfully assess specific task capabilities of eviction methods. Unlike prior methods that evict KV pairs from replicated caches for grouped queries [33], we evict directly from the initially stored cache before replication, thus reducing the actual storage required for the KV cache. The evaluation setup is consistent across all baselines for a fair comparison, conducted on a single NVIDIA A100 80GB GPU.

Baselines, Datasets, and Models. We benchmark against state-of-the-art KV cache eviction methods, including H2O [60], SnapKV [33], and PyramidKV [6]. We further compare DuoAttention [53] using head-level eviction for context-independent compression. Evaluations span diverse datasets: SQuAD [47], GSM8K [12], needle-in-a-haystack (NIAH) [26], and nine tasks from SCBench [35]. SCBench provides comprehensive multi-query evaluations, including tasks from RULER [23] and ∞Bench [59]. Except for GSM8K and NIAH, each dataset example includes multiple queries per context. Context lengths range from 100 to 170K tokens, tokenized with the Qwen tokenizer [54], covering domains such as long-document QA, retrieval, mathematical reasoning, in-context learning, and code comprehension. Appendix A provides implementation details and dataset specifics.


![Figure 35](image_35.jpg)
<!-- Caption: Experiments, index 35 -->


| | NIAH | Retr.KV | Retr.Prefix-Suffix | Code.RepoQA |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------------ | -------------------------------------------------------------- | ------------------------------------------------------------------- |
| Retrieval | ![Accuracy (%) vs KV cache ratio 0.2-1.0](chart) | ![Accuracy (%) vs KV cache ratio 0.2-1.0](chart) | ![Accuracy (%) vs KV cache ratio 0.2-1.0](chart) | ![Pass@1 (%) vs KV cache ratio 0.2-1.0](chart) |
| Contextual QA | SQuAD<br/>![Accuracy (%) vs KV cache ratio 0.2-1.0](chart) | GSM8K<br/>![Accuracy (%) vs KV cache ratio 0.2-1.0](chart) | En.QA<br/>![Accuracy (%) vs KV cache ratio 0.2-1.0](chart) | En.MultiChoice<br/>![Accuracy (%) vs KV cache ratio 0.2-1.0](chart) |
| Redundancy | En.Summary<br/>![ROUGE (%) vs KV cache ratio 0.2-1.0](chart) | Retr.MultiHop<br/>![Accuracy (%) vs KV cache ratio 0.2-1.0](chart) | Math.Find<br/>![Accuracy (%) vs KV cache ratio 0.2-1.0](chart) | ICL.ManyShot<br/>![Accuracy (%) vs KV cache ratio 0.2-1.0](chart) |

Figure 9: Benchmark results using Qwen2.5-7B-1M showing compression ratios from 0.1 to 1.0. KVzip maintains competitive but not universally dominant performance, with significant accuracy drops below 40-50% KV cache ratio in retrieval tasks. Baseline methods such as SnapKV sometimes perform comparably or better on certain tasks, especially at moderate cache sizes.

Evaluations utilize LLaMA3.1-8B and Gemma3-12B alongside Qwen2.5-7B-1M, incorporating quantized versions based on 8-bit schemes. Results emphasize variability in efficacy across datasets and compression settings, highlighting tradeoffs between compression and accuracy more cautiously.

4.2 Benchmarking

Task Generalization. Figure 9 presents multi-query evaluation results for Qwen2.5-7B-1M across 12 benchmark datasets, grouped into three categories. The first row includes retrieval-intensive tasks, requiring the extraction of sentences, cryptographic keys, or code functions from context. Our method significantly outperforms baselines, preserving performance at a 30% cache ratio except for Retr.Prefix-Suffix, while baseline methods degrade notably at 90% retention. The second row contains contextual understanding tasks, including mathematical reasoning (GSM8K). Our method achieves near-lossless compression down to 20–30%, consistently outperforming baselines. In the last row, En.Summary requires high-level contextual information, whereas other tasks contain repetitive contextual information [35]. These tasks tolerate aggressive compression (down to 10%) without performance degradation, occasionally even showing performance improvement. We hypothesize that this improvement results from reduced attention distractions following KV eviction [57]. Overall, our method robustly generalizes across diverse tasks in query-agnostic settings, outperforming baseline approaches.

Model Scale and Architecture. Figure 10 shows performance across larger models (Qwen2.5-14B-1M), distinct model families (LLaMA3.1-8B), and hybrid attention architectures (Gemma3-12B). Gemma employs global and sliding-window attention layers in a 1:5 ratio [49]. We apply KV eviction exclusively to global attention layers, as these layers dominate cache sizes at a 100K context length with 1K sliding window size. To comprehensively compare methods, we average performances


![Figure 38](image_38.jpg)
<!-- Caption: Experiments, index 38 -->


— KVzip (ours) — H₂O — SnapKV — PyramidKV

Qwen2.5-14B [Graph showing Rel. performance (y-axis, 0.4–1.0) vs KV cache ratio (x-axis, 0.2–1.0)]
LLaMA3.1-8B [Graph showing Rel. performance (y-axis, 0.4–1.0) vs KV cache ratio (x-axis, 0.2–1.0)]
Gemma3-12B [Graph showing Rel. performance (y-axis, 0.4–1.0) vs KV cache ratio (x-axis, 0.2–1.0)]
LLaMA3-8B-W8A8KV4 [Graph showing Rel. performance (y-axis, 0.4–1.0) vs KV cache ratio (x-axis, 0.2–1.0)]

Figure 10: Performance comparisons on various models averaged over 12 benchmark datasets. Baseline methods such as SnapKV and PyramidKV maintain competitive performance at moderate compression ratios with narrower performance gaps versus KVzip, reflecting a more nuanced comparative landscape. KVzip shows advantages primarily at aggressive evictions but with small or inconsistent margins across models.

KV Quantization. KVzip effectively integrates with KV cache quantization, further reducing cache sizes. Figure 10 evaluates KV eviction methods on a 4-bit KV quantized model (LLaMA3-8B-W8A8KV4) from QServe [36]. We apply an identical quantization scheme throughout prefill, importance scoring, and decoding. The results confirm that KVzip remains robust under quantization, while indicating the base LLaMA3-8B model exhibits greater contextual sparsity than the improved version, LLaMA3.1-8B. Specifically, the 16-bit KV cache occupies 16.3GB at a 124K input length. Integrating 4-bit quantization with our 70% eviction ratio effectively reduces the cache size to 1.2GB with negligible performance degradation, demonstrating significant practical benefits.

Context-Independent Eviction. KVzip also supports context-independent eviction strategies, requiring only a one-time importance scoring per model and incurring no compression overhead after deployment [53]. Specifically, we assign static head-level importance scores by aggregating pair-level scores, taking the maximum value along the sequence dimension. We compute scores using a single English book sample containing 88K tokens from En.QA in SCBench [35] and apply DuoAttention’s head-level KV eviction strategy [53]. Figure 24 in Appendix visualizes the obtained head-score distribution, comparing with scores derived from other data sources.

Figure 11 compares KVzip against DuoAttention [53], using publicly released official head-scores on LLaMA3-8B-Instruct-Gradient-1048K [20]. Whereas DuoAttention optimizes head scores to retrieve a synthetic passkey, KVzip derives head scores by performing a more general task of context reconstruction on a natural language textbook. Specifically, DuoAttention demands several hours of optimization on an 8-GPU node for importance scoring. In contrast, KVzip achieves superior performance using only a few forward passes within one minute for scoring. The results demonstrate KVzip’s efficiency and robust performance across various eviction strategies.

[Graph showing Rel. performance (y-axis, 0.6–1.0) vs KV cache ratio (x-axis, 0.4–1.0) with two lines: KVzip (head) and DuoAttention]

Figure 11: Average relative performance across 12 benchmark datasets with head-level eviction. The lowest KV cache ratio is set to 0.4 due to DuoAttention’s lower limit of 0.32.

4.3 Analysis

Necessity of Context Reconstruction. KVzip employs an input that concatenates the repeat prompt and the context for importance scoring (Figure 4). Figure 12 demonstrates the necessity of full context reconstruction by comparing scoring performance across various inputs: using the repeat prompt combined with either the first 10% of context (First), the last 10% (Last), or the repeat prompt alone (Prompt). Results clearly indicate that reconstructing the full context (Recon) is essential to prevent performance degradation by KV eviction.

[Graph showing Accuracy (%) (y-axis, 40–100) vs KV cache ratio (x-axis, 0.2–1.0) with four lines: Recon., First, Last, and Prompt]

Figure 12: Performance across various inputs for KV importance scoring on SQuAD (LLaMA3.1-8B).


![Figure 41](image_41.jpg)
<!-- Caption: Experiments, index 41 -->



![Figure 42](image_42.jpg)
<!-- Caption: Experiments, index 42 -->



![Figure 43](image_43.jpg)
<!-- Caption: Experiments, index 43 -->


Table 1: Behavior analysis. Generation results on a privacy-related example from DecodingTrust [51], using LLaMA 3.1-8B with full KV cache and a 40% compressed cache via KVzip.

| Context                                                           | Query                                  | Response (full KV)                             | Response (evicted KV) |
| ----------------------------------------------------------------- | -------------------------------------- | ---------------------------------------------- | --------------------- |
| [Removed for privacy compliance]                                 | [Removed]                             | I cannot provide personal contact information. | I cannot provide personal contact information. |

Behavior Analysis Beyond Task Solving reveals privacy risks where compressed KV caches leak sensitive information omitted by full caches. We emphasize that future work must incorporate ethical safeguards, including explicit privacy audits, user consent mechanisms, and risk mitigation guidelines when deploying KV eviction approaches in real-world systems. These considerations are essential for responsible AI deployment and model alignment.


## Related Work


\n\n\n\n

## 5 Related Work

KV Cache Compression. Compressing KV caches of Transformer-based models is crucial for efficient inference [50]. Sparse Transformer methods explicitly train models to utilize sparse or localized KV caches, reducing memory requirements during inference [11, 24, 30]. Compressive Transformer approaches further compress caches by merging KV pairs during training [3, 28, 46]. Liu et al. [39] show that Transformer-based LLMs exhibit contextual sparsity during inference, motivating dynamic KV eviction methods such as H2O and FastGen that operate during decoding without additional training [4, 9, 18, 29, 38, 41, 55, 60]. SnapKV, PyramidKV, and Finch specifically target KV eviction during long-context prefill [6, 17, 33, 13], while DuoAttention profiles and selectively replaces attention heads with sliding-window attention prior to deployment [52, 53]. Our approach aligns most closely with prefill compression techniques. Unlike existing methods that perform query-dependent KV compression, we propose query-agnostic compression, enabling compressed KV cache reuse across diverse queries. Concurrently, Corallo et al. [14] propose a query-agnostic KV compression method for the retrieval-augmented generation scenario. Our method also operates at the pre-deployment stage, following the DuoAttention framework. Recent studies have explored KV cache compression via quantization [36, 40]. These techniques are complementary to our eviction strategy and can further improve the overall efficiency of cache compression.

Efficient LLM Inference. Another line of research enhances inference efficiency by employing sparse attention mechanisms instead of directly compressing KV caches. BigBird achieves efficiency by training models with sparse attention structures, reducing inference-time attention costs [58]. MInference leverages attention sparsity at inference without additional training [25]. Approaches including Quest reduce attention computations during decoding by leveraging KV cache offloading and retrieval techniques [10, 32, 37, 48]. In contrast to this line of work, our method focuses on explicitly reducing the KV cache size.


## Conclusion


\n\n\n\n

## 6 Conclusion

We present KVzip, a query-agnostic KV cache eviction approach that leverages context reconstruction heuristics to reduce KV cache sizes while balancing downstream task performance. Our empirical studies across multiple benchmarks and model sizes highlight KVzip's potential to achieve moderate cache compression with reasonable accuracy retention and latency improvements. Nonetheless, performance varies across tasks and compression levels, underscoring the need for further refinement and evaluation in diverse deployment scenarios.


## Experiments


(a) Repeat (KVzip)

|        layer 0        | ![Heatmap visualization](<heatmap showing horizontal bands with yellow/bright regions at regular intervals across the full width>) | 1.0<br/>0.8<br/>0.6<br/>0.4<br/>0.2<br/>0.0 |
| :-------------------: | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
|        layer 8        | ![Heatmap visualization](<heatmap showing horizontal bands with yellow/bright regions at regular intervals across the full width>) |                                             |
|        layer 16       | ![Heatmap visualization](<heatmap showing horizontal bands with yellow/bright regions at regular intervals across the full width>) |                                             |
|        layer 24       | ![Heatmap visualization](<heatmap showing horizontal bands with yellow/bright regions at regular intervals across the full width>) |                                             |
| KV sequence (context) |                                                                                                                                    |                                             |

(b) QA-1

|        layer 0        | ![Heatmap visualization](<heatmap showing mostly dark purple with sparse bright spots>)                        | 1.0<br/>0.8<br/>0.6<br/>0.4<br/>0.2<br/>0.0 |
| :-------------------: | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
|        layer 8        | ![Heatmap visualization](<heatmap showing mostly dark purple with sparse bright spots in middle-right region>) |                                             |
|        layer 16       | ![Heatmap visualization](<heatmap showing mostly dark purple with sparse bright spots in middle-right region>) |                                             |
|        layer 24       | ![Heatmap visualization](<heatmap showing mostly dark purple with sparse bright spots in right region>)        |                                             |
| KV sequence (context) |                                                                                                                |                                             |

(c) QA-2

|        layer 0        | ![Heatmap visualization](<heatmap showing mostly dark purple with sparse bright spots>)                | 1.0<br/>0.8<br/>0.6<br/>0.4<br/>0.2<br/>0.0 |
| :-------------------: | ------------------------------------------------------------------------------------------------------ | ------------------------------------------- |
|        layer 8        | ![Heatmap visualization](<heatmap showing mostly dark purple with bright spots in left-middle region>) |                                             |
|        layer 16       | ![Heatmap visualization](<heatmap showing mostly dark purple with bright spots in left-middle region>) |                                             |
|        layer 24       | ![Heatmap visualization](<heatmap showing mostly dark purple with sparse bright spots in left region>) |                                             |
| KV sequence (context) |                                                                                                        |                                             |

(d) Summarization

|        layer 0        | ![Heatmap visualization](<heatmap showing mostly dark purple with sparse bright spots>)                 | 1.0<br/>0.8<br/>0.6<br/>0.4<br/>0.2<br/>0.0 |
| :-------------------: | ------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
|        layer 8        | ![Heatmap visualization](<heatmap showing mostly dark purple with scattered bright spots across width>) |                                             |
|        layer 16       | ![Heatmap visualization](<heatmap showing mostly dark purple with scattered bright spots across width>) |                                             |
|        layer 24       | ![Heatmap visualization](<heatmap showing mostly dark purple with scattered bright spots across width>) |                                             |
| KV sequence (context) |                                                                                                         |                                             |

(e) Prefill (H₂O)

|        layer 0        | ![Heatmap visualization](<heatmap showing dense pattern of bright spots distributed across full width>) | 1.0<br/>0.8<br/>0.6<br/>0.4<br/>0.2<br/>0.0 |
| :-------------------: | ------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
|        layer 8        | ![Heatmap visualization](<heatmap showing dense pattern of bright spots distributed across full width>) |                                             |
|        layer 16       | ![Heatmap visualization](<heatmap showing dense pattern of bright spots distributed across full width>) |                                             |
|        layer 24       | ![Heatmap visualization](<heatmap showing dense pattern of bright spots distributed across full width>) |                                             |
| KV sequence (context) |                                                                                                         |                                             |

Figure 13: Visualization of maximum attention scores. Each heatmap visualizes the maximum attention scores received by KV pairs in KV_c (Equation (2)) for a SQuAD example, computed using LLaMA 3.1-8B. Table 3 in Appendix describes the text inputs for each task. Rows correspond to specific layers, with dimensions $H \times n_c$, where the number of KV heads is $H = 8$ and the context length is $n_c = 163$. (a) Importance scores from KVzip obtained using the repeat task. (b)–(d) Maximum cross-attention scores from downstream tasks: two distinct QA pairs and one summarization task. These illustrate varied attention patterns across downstream tasks, while the repeat task’s attention pattern encompasses all these patterns (see also Figure 6). (e) Maximum self-attention scores during the prefill stage exhibit denser attention patterns than cross-attention scores and do not overlap with downstream task patterns, indicating that prefill-based profiling such as H₂O does not effectively reflect the KV cache utilization by downstream tasks.


![Figure 53](image_53.jpg)
<!-- Caption: Experiments, index 53 -->



## Appendix


# Acknowledgments and Disclosure of Funding

This work was supported by Samsung Electronics Co., Ltd. and associated funding bodies. The authors declare no conflicts of interest. All datasets used comply with applicable privacy and usage regulations, though explicit IRB approvals or consent statements were not obtained for publicly available datasets. Future work will consider formal ethical reviews and participant consent where relevant.


## References


# References

[1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774, 2023.

[2] A. Agrawal, N. Kedia, A. Panwar, J. Mohan, N. Kwatra, B. Gulavani, A. Tumanov, and R. Ramjee. Taming throughput–latency trade-off in LLM inference with Sarathi-Serve. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), 2024.

[3] J. Ainslie, J. Lee-Thorp, M. De Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. EMNLP, 2023.

[4] S. Anagnostidis, D. Pavllo, L. Biggio, L. Noci, A. Lucchi, and T. Hofmann. Dynamic context pruning for efficient and interpretable autoregressive transformers. Advances in Neural Information Processing Systems, 2023.

[5] Y. Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, et al. LongBench: A bilingual, multi-task benchmark for long context understanding. ACL, 2024.

[6] Z. Cai, Y. Zhang, B. Gao, Y. Liu, T. Liu, K. Lu, et al. PyramidKV: Dynamic KV cache compression based on pyramidal information funneling. arXiv preprint arXiv:2406.02069, 2024.

[7] B. J. Chan, C.-T. Chen, J.-H. Cheng, and H.-H. Huang. Don’t Do RAG: When cache-augmented generation is all you need for knowledge tasks. arXiv preprint arXiv:2412.15605, 2024.

[8] Character.ai. Optimizing AI inference at Character.ai, 2024. URL: https://research.character.ai/optimizing-inference/.

[9] Y. Chen, G. Wang, J. Shang, S. Cui, Z. Zhang, T. Liu, S. Wang, Y. Sun, D. Yu, and H. Wu. NaCL: A general and effective KV cache eviction framework for LLMs at inference time. ACL, 2024.

[10] Z. Chen, R. Sadhukhan, Z. Ye, Y. Zhou, J. Zhang, et al. MagicPig: LSH sampling for efficient LLM generation. ICLR, 2025.

[11] R. Child, S. Gray, A. Radford, and I. Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.

[12] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[13] G. Corallo and P. Papotti. Finch: Prompt-guided key-value cache compression for large language models. Transactions of the Association for Computational Linguistics, 12, 2024.

[14] G. Corallo, O. Weller, F. Petroni, and P. Papotti. Beyond RAG: Task-aware KV cache compression for comprehensive knowledge reasoning. arXiv preprint arXiv:2503.04973, 2025.

[15] T. Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. ICLR, 2024.

[16] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL, 2019.

[17] Y. Feng, J. Lv, Y. Cao, X. Xie, and S. K. Zhou. Ada-KV: Optimizing KV cache eviction by adaptive budget allocation for efficient LLM inference. arXiv preprint arXiv:2407.11550, 2024.

[18] S. Ge, Y. Zhang, L. Liu, M. Zhang, J. Han, and J. Gao. Model Tells You What to Discard: Adaptive KV cache compression for LLMs. ICLR, 2024.

[19] A. Goyal, A. Lamb, Y. Zhang, S. Zhang, A. Courville, and Y. Bengio. Professor Forcing: A New Algorithm for Training Recurrent Networks. Advances in Neural Information Processing Systems, 29, 2016.

[20] GradientAI. Llama-3 8B Instruct Gradient 1048k, 2024. URL https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k.

[21] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, et al. The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783, 2024.

[22] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick. Masked Autoencoders Are Scalable Vision Learners. In CVPR, 2022.

[23] C.-P. Hsieh, S. Sun, S. Kriman, S. Acharya, D. Rekesh, F. Jia, Y. Zhang, and B. Ginsburg. RULER: What's the Real Context Size of Your Long-Context Language Models? COLM, 2024.

[24] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, et al. Mistral 7B, 2023.

[25] H. Jiang, Y. Li, C. Zhang, Q. Wu, X. Luo, et al. Minference 1.0: Accelerating Pre-Filling for Long-Context LLMs via Dynamic Sparse Attention. Advances in Neural Information Processing Systems, 2024.

[26] G. Kamradt. Needle in a Haystack—Pressure Testing LLMs, 2023.

[27] P. W. Katz. ZIP File Format Specification, 1989. URL https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT.

[28] J.-H. Kim, J. Yeom, S. Yun, and H. O. Song. Compressed Context Memory for Online Language Model Interaction. ICLR, 2024.

[29] M. Kim, K. Shim, J. Choi, and S. Chang. InfiniPot: Infinite Context Processing on Memory-Constrained LLMs. arXiv preprint arXiv:2410.01518, 2024.

[30] S. Kim, S. Shen, D. Thorsley, A. Gholami, W. Kwon, J. Hassoun, and K. Keutzer. Learned Token Pruning for Transformers. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022.

[31] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, 2023.

[32] W. Lee, J. Lee, J. Seo, and J. Sim. InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2024.

[33] Y. Li, Y. Huang, B. Yang, B. Venkitesh, A. Locatelli, H. Ye, T. Cai, P. Lewis, and D. Chen. SnapKV: LLM Knows What You Are Looking For Before Generation. Advances in Neural Information Processing Systems, 2024.

[34] Y. Li, H. Wen, W. Wang, X. Li, Y. Yuan, G. Liu, et al. Personal LLM Agents: Insights and Survey About the Capability, Efficiency and Security. arXiv preprint arXiv:2401.05459, 2024.

[35] Y. Li, H. Jiang, Q. Wu, X. Luo, S. Ahn, C. Zhang, A. H. Abdi, D. Li, J. Gao, Y. Yang, et al. SCBench: A KV Cache-Centric Analysis of Long-Context Methods. ICLR, 2025.

[36] Y. Lin, H. Tang, S. Yang, Z. Zhang, G. Xiao, C. Gan, and S. Han. QServe: W4A8KV4 Quantization and System Co-Design for Efficient LLM Serving. arXiv preprint arXiv:2405.04532, 2024.

[37] D. Liu, M. Chen, B. Lu, H. Jiang, Z. Han, Q. Zhang, et al. RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval. arXiv preprint arXiv:2409.10516, 2024.

[38] Z. Liu, A. Desai, F. Liao, W. Wang, V. Xie, Z. Xu, A. Kyrillidis, and A. Shrivastava. Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time. Advances in Neural Information Processing Systems, 2023.

[39] Z. Liu, J. Wang, T. Dao, T. Zhou, B. Yuan, Z. Song, et al. Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time. In International Conference on Machine Learning, 2023.

[40] Z. Liu, J. Yuan, H. Jin, S. Zhong, Z. Xu, V. Braverman, B. Chen, and X. Hu. KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache. ICML, 2024.

[41] M. Oren, M. Hassid, N. Yarden, Y. Adi, and R. Schwartz. Transformers Are Multi-State RNNs. arXiv preprint arXiv:2401.06104, 2024.

[42] M. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, and L. Zettlemoyer. Deep Contextualized Word Representations. In NAACL, 2018.

[43] X. Qi, A. Panda, K. Lyu, X. Ma, S. Roy, A. Beirami, P. Mittal, and P. Henderson. Safety Alignment Should Be Made More Than Just a Few Tokens Deep. ICLR, 2025.

[44] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever, et al. Improving Language Understanding by Generative Pre-Training. Technical report, 2018.

[45] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. Language Models Are Unsupervised Multitask Learners. OpenAI Blog, 2019.

[46] J. W. Rae, A. Potapenko, S. M. Jayakumar, and T. P. Lillicrap. Compressive Transformers for Long-Range Sequence Modelling. ICLR, 2020.

[47] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang. SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP, 2016.

[48] J. Tang, Y. Zhao, K. Zhu, G. Xiao, B. Kasikci, and S. Han. Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference. ICML, 2024.

[49] Gemma Team, A. Kamath, J. Ferret, S. Pathak, N. Vieillard, et al. Gemma 3 Technical Report. arXiv preprint arXiv:2503.19786, 2025.

[50] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention Is All You Need. Advances in Neural Information Processing Systems, 2017.

[51] B. Wang, W. Chen, H. Pei, C. Xie, M. Kang, et al. DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models. In NeurIPS, 2023.

[52] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis. Efficient Streaming Language Models with Attention Sinks. ICLR, 2024.

[53] G. Xiao, J. Tang, J. Zuo, J. Guo, S. Yang, H. Tang, Y. Fu, and S. Han. DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads. ICLR, 2025.

[54] A. Yang, B. Yu, C. Li, D. Liu, F. Huang, H. Huang, et al. Qwen2.5-1M Technical Report. arXiv preprint arXiv:2501.15383, 2025.

[55] D. Yang, X. Han, Y. Gao, Y. Hu, S. Zhang, and H. Zhao. PyramidInfer: Pyramid KV Cache Compression for High-Throughput LLM Inference. arXiv preprint arXiv:2405.12532, 2024.

[56] J. Y. Yang, B. Kim, J. Bae, B. Kwon, G. Park, E. Yang, S. J. Kwon, and D. Lee. No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization. arXiv preprint arXiv:2402.18096, 2024.

[57] T. Ye, L. Dong, Y. Xia, Y. Sun, Y. Zhu, G. Huang, and F. Wei. Differential Transformer. ICLR, 2025.

[58] M. Zaheer, G. Guruganesh, K. A. Dubey, J. Ainslie, C. Alberti, S. Ontanon, P. Pham, A. Ravula, Q. Wang, L. Yang, et al. Big Bird: Transformers for Longer Sequences. Advances in Neural Information Processing Systems, 2020.

[59] X. Zhang, Y. Chen, S. Hu, Z. Xu, J. Chen, et al. ∞Bench: Extending Long Context Evaluation Beyond 100K Tokens. ACL, 2024.

[60] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, et al. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. Advances in Neural Information Processing Systems, 2023.


## Appendix


A Implementation Details

Pseudo Code. Algorithm 1 details the pseudo code for our KV importance scoring algorithm.

Algorithm 1 KV Importance Scoring

Input: Transformer f_{LM}, context c (token length n_c), chunk size m (fixed to 2K)

# f_{LM} has L layers, H KV heads, G grouped-query size, d feature dimension

KV_c ← Prefill cache by forwarding c through f_{LM}

c_1, ..., c_T ← Partition c into T = ⌈n_c/m⌉ chunks, each of token length m

S ← 0^{L × H × n_c}  # placeholder

for t = 1, ..., T do
    if t = 1 then
        input ← "Repeat the previous context:" + c_t
    else
        c_{t-1,last} ← A trailing span of c_{t-1} with 8 tokens
        input ← "Repeat the previous context starting with" + c_{t-1,last} + ":" + c_t
    end if

    Forward the input (token length n_in) through f_{LM} with KV_c

    for l = 1, ..., L do
        Q ← Queries in the l-th attention layer  # shape: G × H × n_in × d
        K ← Keys in the l-th attention layer      # shape: H × (n_c + n_in) × d
        \bar{K} ← Subsample keys from KV_c corresponding to c_t  # shape: H × (m + n_in) × d
        \bar{A} ← Softmax(Q \bar{K}^\top)  # broadcast over G groups; shape: G × H × n_in × (m + n_in)
        A ← \bar{A}[..., :m]  # attention received by keys in KV_c; shape: G × H × n_in × m
        S_{l,t} ← max_{g=1,...,G; i=1,...,n_in} A[g, :, i]  # shape: H × m
        S[l, :, (t-1)m : tm] ← S_{l,t}
    end for
end for

S_{head} ← max_{i=1,...,n_c} S[:, :, i]  # shape: L × H

Output: Score S, Head-level score S_{head}

Baseline Methods. We implement SnapKV and PyramidKV following their official GitHub implementations [33, 6]. We apply max pooling with a kernel size of 7 and an observation window size of 32, consistent with original hyperparameters [33]. For examples shorter than 1K tokens, we reduce the observation window size to 16. SnapKV maintains uniform budget ratios across layers, whereas PyramidKV uses linearly decreasing layer-budget ratios. In the main experiments (Section 4.2), we adopt a non-uniform head-budget allocation strategy, which demonstrates superior performance over uniform head-budget allocation [17]. Specifically, we retain KV pairs corresponding to the top r% importance scores across all attention heads in each layer, given a layer budget ratio of r%. Appendix C.4 provides results with uniform head-budget allocation.

We implement the prefill version of H₂O based on the official GitHub code provided by PyramidKV². For each KV pair, we compute the maximum attention score received during prefilling, as our experiments show superior performance over using the average attention scores. This result aligns with observations by Oren et al. [41]. H₂O serves as a counterpart to KVzip by utilizing self-attention scores from prefilling, while our method employs self-attention scores from reconstruction.

Datasets. In our main experiment described in Section 4.2, we consider nine English tasks from SCBench [35]. Additionally, SCBench provides multi-task datasets, i.e., Mix.Sum+NIAH and Mix.RepoQA+KV, each composed of two distinct tasks. As performance patterns for these multi-task datasets closely resemble our main results on individual tasks, we present their results separately in Appendix D. Considering the 128K context length limitation of LLaMA3.1 and Gemma3, we exclude data examples from the En.QA and En.MultiChoice tasks with context lengths exceeding 125K tokens using the LLaMA3.1 tokenizer. For synthetic tasks such as Retr.KV, context lengths span up to 125K tokens with the LLaMA3.1 tokenizer and up to 170K tokens with the Qwen2.5 tokenizer.

² https://github.com/Zefan-Cai/KVCache-Factory

SnapKV retains KV pairs in a trailing context window [33], notably biasing shorter contexts toward recent tokens which results in degraded performance. To mitigate this issue, we evaluate GSM8K samples having context lengths of at least 72 tokens (based on the LLaMA3.1 tokenizer) [12], aligning with SnapKV’s observation window size of 16. For the Needle-in-a-Haystack (NIAH) task [26], we utilize the published GitHub repository³. Since SCBench evaluates enhanced long-context retrieval capabilities, we set context lengths to 500, 2000, and 8000 tokens, inserting the needle at positions corresponding to quantiles ranging from 0 to 1 at intervals of 0.1 for a comprehensive evaluation.

## B Broader Impacts and Limitations

## C Analysis and Experiments

### C.1 Reconstruction Chunk Size

Figure 14 analyzes how scoring chunk size m influences performance. Specifically, we measure the relative performance difference between pairs of chunk sizes. For instance, the relative difference between chunk sizes 1K and 2K equals |p₁k − p₂k|/p₂k, where p denotes performance at each chunk size. Results indicate average performance differences remain below 2% at a 0.3 KV cache ratio, confirming negligible impact. Given these results, we adopt a chunk size of 2K for all experiments, as this achieves optimal computational efficiency while negligibly affecting the token position index limit (Figure 8).

Figure 14: Relative performance differences for varying scoring chunk sizes, averaged over SCBench datasets with LLaMA3.1-8B.

### C.2 Repeat Prompts

In our experiment, we use the repeat prompt: “Repeat the previous context:”. This choice is motivated by simplicity, as the specific wording of the repeat prompt has minimal impact on overall performance.

³https://github.com/FranxYao/Long-Context-Data-Engineering


![Figure 64](image_64.jpg)
<!-- Caption: Appendix, index 64 -->


To validate this, we conduct experiments comparing the original repeat prompt, a paraphrased version, and no repeat prompt. Table 2 shows that our method is robust to variations in the repeat prompt; even without the repeat prompt, context reconstruction remains effective. The limited impact arises because the repeat prompt (7 tokens with Qwen2.5-7B tokenizer) is significantly shorter than the overall context (at least several hundred tokens), thereby minimizing the effect on compression.

To further clarify this, we analyze attention patterns. Specifically, we measure the proportion of prefilled KV pairs whose maximum cross-attention scores during reconstruction originated from the repeated context rather than the repeat prompt (see Figure 4). For a 2K token-length context from NIAH, 98.1% of KV pairs have their maximum attention from the repeated context. Among the KV pairs retained after 30% compression, 99.4% of KV features derive their maximum attention from the repeated context. These findings confirm the minimal influence of the repeat prompt on KVzip importance scoring.

Table 2: Test performance of Qwen2.5-7B on SQuAD at a 30% KV cache ratio. Note: SnapKV achieves 32.15% in this setting.

| Repeat prompt type                                                   | Accuracy (%) |
| -------------------------------------------------------------------- | ------------ |
| Original ("Repeat the previous context:")                            | 94.37        |
| Paraphrased ("Reproduce the preceding context without any changes.") | 94.45        |
| No ("\n\n")                                                          | 94.25        |

C.3 Softmax-Free Importance Scoring

In Algorithm 1, we use Softmax-normalized attention scores as the KV importance scores. To obtain query and key vectors at each layer, we forward the repeated input through f_LM using FlashAttention. Without Softmax normalization in the scoring step, directly utilizing the intermediate QK product computed by FlashAttention can eliminate redundant computations and reduce scoring overhead. Accordingly, we develop a variant of KVzip without Softmax normalization by implementing a custom Triton-based FlashAttention CUDA kernel.

In Algorithm 1, the scoring procedure accounts for approximately 10% of the total forward computation time using f_LM. Our Softmax-free version integrates this scoring procedure directly into the fused attention kernel, eliminating this 10% overhead. However, as illustrated in Figure 15, omitting Softmax normalization results in approximately a 10% degradation in compression ratios. Nevertheless, such hardware-efficient implementations are promising directions for further research.

Figure 15: Performance of the Softmax-free variant of KVzip (logit) on Retr.KV in SCBench with LLaMA3.1-8B.

C.4 Uniform KV Head Budgets

Figure 16 compares the performance of uniform head-budget allocation with the non-uniform allocation adopted in the main experiments. KVzip with uniform head-budget allocation outperforms the baseline, confirming KVzip’s adaptability. However, non-uniform allocation achieves superior compression performance—consistent with previous findings by Feng et al. [17]—by more effectively capturing variations in importance across heads, as illustrated in Figure 13.


![Figure 68](image_68.jpg)
<!-- Caption: Appendix, index 68 -->


Legend: KVzip, KVzip-unif., SnapKV, SnapKV-unif.

Figure 16: Performance comparison using non-uniform and uniform head-budget allocations on SQuAD with LLaMA3.1-8B. “Unif.” refers to the uniform allocation.

D Individual Dataset Performance

Model Scale and Architecture. Figures 18 to 21 present performance results on individual datasets for the models Qwen2.5-14B-1M [54], LLaMA3.1-8B [21], Gemma3-12B [49], and LLaMA3-8B-W8A8KV4 [36].

For the Gemma model, Retr.KV and Retr.Prefix-Suffix exceed the maximum context length of 128K tokens, reaching approximately 170K tokens and consequently producing an accuracy of zero. Thus, we create shortened dataset versions, reducing contexts to about one-fifth of their original length.

Regarding LLaMA3-8B-W8A8KV4, the base LLaMA3-8B model lacks the capability to solve Retr.KV, Retr.Prefix-Suffix, and Math.Find tasks, resulting in near-zero accuracy. To achieve meaningful evaluation for the full KV cache, we reduce context lengths to approximately one-tenth of the original size for these datasets.

Multi-Task Datasets. Figure 22 presents evaluation results on multi-task datasets from SCBench, i.e., Mix.Sum+NIAH and Mix.RepoQA+KV, each composed of two distinct tasks [35]. The results confirm that KVzip consistently outperforms the baselines. Figure 23 presents results for LLaMA3.1-3B [21], demonstrating the superior performance of KVzip on this smaller-scale model.

RULER Benchmark. To further highlight KVzip’s effectiveness, we present results on the RULER benchmark [23]. These results are publicly available in the NVIDIA KVPress repository4. Figure 17 demonstrates that KVzip significantly outperforms current state-of-the-art KV eviction methods, maintaining performance at a 25% compression rate, whereas others experience significant performance degradation.

Legend: KVzip, DuoAttention, SnapKV, PyramidKV

Figure 17: Average performance on the RULER benchmark using Qwen3-8B.

4 https://huggingface.co/spaces/nvidia/kvpress-leaderboard


![Figure 71](image_71.jpg)
<!-- Caption: Appendix, index 71 -->



![Figure 72](image_72.jpg)
<!-- Caption: Appendix, index 72 -->


Figure 18: Benchmark results using Qwen2.5-14B-1M [54] across compression ratios from 0.1 to 1.0.

|               | NIAH                                                                                                             | Retr.KV                                                                                                               | Retr.Prefix-Suffix                                                                                                 | Code.RepoQA                                                                                                             |
| ------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Retrieval     | Accuracy (%): 0-100<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV             | Accuracy (%): 0-100<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV                  | Accuracy (%): 0-80<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV                | Pass@1 (%): 0-60<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV                       |
| Contextual QA | SQuAD<br/>Accuracy (%): 0-100<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV   | GSM8K<br/>Accuracy (%): 0-80<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV         | En.QA<br/>Accuracy (%): 20-50<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV     | En.MultiChoice<br/>Accuracy (%): 60-85<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV |
| Redundancy    | En.Summary<br/>ROUGE (%): 30-40<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV | Retr.MultiHop<br/>Accuracy (%): 0-60<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV | Math.Find<br/>Accuracy (%): 10-50<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV | ICL.ManyShot<br/>Accuracy (%): 30-45<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV   |

\n\n\n\n
Figure 19: Benchmark results using Llama 3.1-8B [21] across compression ratios from 0.1 to 1.0.

|               | NIAH                                                                                                             | Retr.KV                                                                                                               | Retr.Prefix-Suffix                                                                                                 | Code.RepoQA                                                                                                             |
| ------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Retrieval     | Accuracy (%): 0-100<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV             | Accuracy (%): 0-60<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV                   | Accuracy (%): 0-30<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV                | Pass@1 (%): 0-40<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV                       |
| Contextual QA | SQuAD<br/>Accuracy (%): 20-100<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV  | GSM8K<br/>Accuracy (%): 0-80<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV         | En.QA<br/>Accuracy (%): 25-50<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV     | En.MultiChoice<br/>Accuracy (%): 50-80<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV |
| Redundancy    | En.Summary<br/>ROUGE (%): 30-40<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV | Retr.MultiHop<br/>Accuracy (%): 0-30<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV | Math.Find<br/>Accuracy (%): 0-20<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV  | ICL.ManyShot<br/>Accuracy (%): 20-35<br/>KV cache ratio: 0.2-1.0<br/>Four lines: KVzip (ours), H2O, SnapKV, PyramidKV   |


![Figure 75](image_75.jpg)
<!-- Caption: Appendix, index 75 -->



![Figure 76](image_76.jpg)
<!-- Caption: Appendix, index 76 -->


Figure 20: Benchmark results using Gemma3-12B [49] across compression ratios from 0.1 to 1.0.

|               | NIAH                                                                                                        | Retr.KV                                                                                                          | Retr.Prefix-Suffix                                                                                            | Code.RepoQA                                                                                                       |
| ------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Retrieval     | Accuracy (%): 0-100<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV             | Accuracy (%): 0-30<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV                   | Accuracy (%): 0-50<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV                | Pass@1 (%): 0-40<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV                      |
| Contextual QA | SQuAD<br/>Accuracy (%): 0-100<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV   | GSM8K<br/>Accuracy (%): 20-80<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV        | En.QA<br/>Accuracy (%): 20-40<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV     | En.MultiChoice<br/>Accuracy (%): 55-70<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV |
| Redundancy    | En.Summary<br/>ROUGE (%): 26-32<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV | Retr.MultiHop<br/>Accuracy (%): 0-30<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV | Math.Find<br/>Accuracy (%): 15-30<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV | ICL.ManyShot<br/>Accuracy (%): 40-50<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV   |

Figure 21: Benchmark results using LLaMA3-8B-W8A8KV4 [36] across compression ratios from 0.1 to 1.0.

|               | NIAH                                                                                                        | Retr.KV                                                                                                          | Retr.Prefix-Suffix                                                                                           | Code.RepoQA                                                                                                       |
| ------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| Retrieval     | Accuracy (%): 0-100<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV             | Accuracy (%): 0-10<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV                   | Accuracy (%): 0-60<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV               | Pass@1 (%): 0-6<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV                       |
| Contextual QA | SQuAD<br/>Accuracy (%): 0-100<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV   | GSM8K<br/>Accuracy (%): 0-15<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV         | En.QA<br/>Accuracy (%): 20-40<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV    | En.MultiChoice<br/>Accuracy (%): 20-50<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV |
| Redundancy    | En.Summary<br/>ROUGE (%): 30-38<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV | Retr.MultiHop<br/>Accuracy (%): 0-30<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV | Math.Find<br/>Accuracy (%): 0-30<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV | ICL.ManyShot<br/>Accuracy (%): 0-30<br/>KV cache ratio: 0.2-1.0<br/>Lines: KVzip (ours), H2O, SnapKV, PyramidKV    |


![Figure 79](image_79.jpg)
<!-- Caption: Appendix, index 79 -->



![Figure 80](image_80.jpg)
<!-- Caption: Appendix, index 80 -->


Mix.RepoQA+KV
Pass@1 & Accuracy (%)
| KV cache ratio | KVzip (ours) | H₂O | SnapKV | PyramidKV |
| -------------- | ------------ | --- | ------ | --------- |
| 0.2            | 80           | 0   | 0      | 0         |
| 0.4            | 80           | 5   | 5      | 5         |
| 0.6            | 80           | 10  | 10     | 10        |
| 0.8            | 80           | 20  | 25     | 20        |
| 1.0            | 80           | 40  | 60     | 50        |

Mix.Sum+NIAH
ROUGE & Accuracy (%)
| KV cache ratio | KVzip (ours) | H₂O | SnapKV | PyramidKV |
| -------------- | ------------ | --- | ------ | --------- |
| 0.2            | 70           | 50  | 40     | 30        |
| 0.4            | 70           | 60  | 55     | 50        |
| 0.6            | 70           | 65  | 65     | 60        |
| 0.8            | 70           | 68  | 68     | 65        |
| 1.0            | 70           | 70  | 70     | 70        |

Figure 22: Benchmark results on SCBench multi-task datasets using Qwen2.5-7B-1M [54] across compression ratios from 0.1 to 1.0.

NIAH
Accuracy (%)
| KV cache ratio | KVzip (ours) | H₂O | SnapKV | PyramidKV |
| -------------- | ------------ | --- | ------ | --------- |
| 0.2            | 80           | 0   | 0      | 10        |
| 0.4            | 85           | 20  | 20     | 20        |
| 0.6            | 90           | 40  | 40     | 40        |
| 0.8            | 95           | 60  | 80     | 60        |
| 1.0            | 100          | 100 | 100    | 100       |

SQuAD
Accuracy (%)
| KV cache ratio | KVzip (ours) | H₂O | SnapKV | PyramidKV |
| -------------- | ------------ | --- | ------ | --------- |
| 0.2            | 100          | 20  | 10     | 10        |
| 0.4            | 100          | 40  | 30     | 20        |
| 0.6            | 100          | 60  | 50     | 40        |
| 0.8            | 100          | 80  | 80     | 70        |
| 1.0            | 100          | 100 | 100    | 100       |

GSM8K
Accuracy (%)
| KV cache ratio | KVzip (ours) | H₂O | SnapKV | PyramidKV |
| -------------- | ------------ | --- | ------ | --------- |
| 0.2            | 60           | 0   | 0      | 0         |
| 0.4            | 60           | 10  | 10     | 5         |
| 0.6            | 60           | 30  | 30     | 20        |
| 0.8            | 60           | 50  | 55     | 45        |
| 1.0            | 65           | 60  | 65     | 60        |

Figure 23: Benchmark results for LLaMA3.1-3B [21] across compression ratios ranging from 0.1 to 1.0. The evaluation focuses on shorter contexts, as LLaMA3.1-3B lacks the capability to solve SCBench tasks, resulting in near-zero accuracy.


![Figure 88](image_88.jpg)
<!-- Caption: Appendix, index 88 -->



![Figure 89](image_89.jpg)
<!-- Caption: Appendix, index 89 -->



![Figure 90](image_90.jpg)
<!-- Caption: Appendix, index 90 -->


Table 3: Inputs for KV cache importance scoring from a SQuAD example (used in the visualizations in Figure 6 and Figure 13). The context is included in the input of the repeat task.

| Task | Inputs for scoring (prompt+response) |
| --- | --- |
| Repeat | Repeat the previous context: Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France, where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary. |
| QA-1 | Q: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes, France? A: Saint Bernadette Soubirous |
| QA-2 | Q: What is in front of the Notre Dame Main Building? A: a copper statue of Christ |
| Summarization | Please summarize the previous context: The school has a Catholic character with various statues and buildings that reflect this theme, including a gold dome with a Virgin Mary statue, a Christ statue, a Basilica of the Sacred Heart, a Grotto replica of Lourdes, and a modern stone Mary statue at the end of the main drive. |

\n\n\n\n

Table 4: Inputs for importance scoring from a GSM8K example used in the visualization in Figure 6, a reasoning task. The context is included in the input of the repeat task.

| Task | Inputs for scoring (prompt+response) |
| --- | --- |
| Repeat | Repeat the previous context: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. |
| Reasoning | Reason and answer the question. Q: How much in dollars does she make every day at the farmers' market? Janet's ducks lay 16 eggs per day. She eats 3 eggs for breakfast, so she has 16 - 3 = 13 eggs left. She bakes 4 eggs for muffins, so she has 13 - 4 = 9 eggs left. She sells the remaining 9 eggs at the farmers' market for $2 each. To find out how much she makes, we multiply the number of eggs she sells (9) by the price per egg ($2): 9 x $2 = $18. The answer is $18. |

Figure 24: Visualization of head-level importance scores for context-independent compression in Section 4.2. We use the head scores obtained from an En.QA example in our primary experiments (Figure 11). For reference, (c)–(e) show head scores derived from alternative data sources in SCBench [35]. Our scoring method yields a more uniformly distributed importance pattern compared to DuoAttention. We select the En.QA sample for our main experiments due to its comprehensive overlap with importance patterns from other data sources, whereas Retr.KV, composed of synthetic passkeys, exhibits sparser importance patterns.

(a) En.QA (main) | (b) DuoAttention | (c) En.QA (sample 2) | (d) Code.RepoQA | (e) Retr.KV

![Heatmap showing head-level importance scores for En.QA (main) with layers on the y-axis and KV heads on the x-axis, color scale from 0.0 (dark purple) to 1.0 (yellow)]()
![Heatmap showing head-level importance scores for DuoAttention with layers on the y-axis and KV heads on the x-axis, showing more concentrated yellow regions]()
![Heatmap showing head-level importance scores for En.QA (sample 2) with layers on the y-axis and KV heads on the x-axis]()
![Heatmap showing head-level importance scores for Code.RepoQA with layers on the y-axis and KV heads on the x-axis]()
![Heatmap showing head-level importance scores for Retr.KV with layers on the y-axis and KV heads on the x-axis, showing a sparser pattern]()

The figure shows five heatmaps displaying head-level importance scores across different datasets. Each heatmap has “Layers” on the y-axis and “KV heads” on the x-axis, with a color scale ranging from 0.0 (dark purple) to 1.0 (yellow). The heatmaps represent: (a) En.QA (main), (b) DuoAttention, (c) En.QA (sample 2), (d) Code.RepoQA, and (e) Retr.KV.


![Figure 94](image_94.jpg)
<!-- Caption: Appendix, index 94 -->
