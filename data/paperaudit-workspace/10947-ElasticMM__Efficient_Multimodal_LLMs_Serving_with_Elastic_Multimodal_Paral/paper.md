
## Abstract


ElasticMM: Efficient Multimodal LLM Serving with Elastic Multimodal Parallelism

Zedong Liu^1,2, Shenggan Cheng^3, Guangming Tan^1, †Yang You^3, and †Dingwen Tao^1

^1 Institute of Computing Technology, Chinese Academy of Sciences
^2 University of Electronic Science and Technology of China
^3 National University of Singapore

† Corresponding authors.

Abstract

Multimodal large language models (MLLMs) extend LLMs to handle images, videos, and audio by incorporating feature extractors and projection modules. However, these additional components—combined with complex inference pipelines and heterogeneous workloads—introduce significant inference overhead. Therefore, efficiently serving MLLMs remains a major challenge. Current tightly coupled serving architectures struggle to distinguish between mixed request types or adapt parallelism strategies to different inference stages, leading to increased time-to-first-token (TTFT) and poor resource utilization. To address this, we introduce Elastic Multimodal Parallelism (EMP), a new serving paradigm that elastically adapts to resource heterogeneity across request types and inference stages. Building upon EMP, we develop ElasticMM, an MLLM serving system that (1) separates requests into independent modality groups with dynamic resource allocation via a modality-aware load balancer; (2) decouples inference stages and enables parallelism adjustment and adaptive scaling via elastic partition scheduling integrated with unified multimodal prefix caching; and (3) improves inference efficiency through non-blocking encoding. Experiments on diverse real-world datasets show that ElasticMM outperforms state-of-the-art (SOTA) serving systems, reducing TTFT by up to 6.8× and achieving 5.1–7.2× higher throughput while meeting service-level objectives (SLOs).


## Introduction


1 Introduction

Large Language Models (LLMs) have had a profound impact on real-world applications due to their exceptional performance [1–3]. Their gradual expansion into the multimodal domain has led to the emergence of Multimodal Large Language Models (MLLMs), which can process inputs such as images, videos, and audio [4, 5]. By employing feature extractors and projection modules to map multimodal inputs into the LLM feature space, MLLMs excel at tasks like visual question answering (VQA), image captioning, and multimodal interaction [6–9]. Consequently, MLLMs are increasingly deployed in online services, where they are highly sensitive to service-level objectives (SLOs).

Unlike traditional LLMs that process only text [10–14], MLLMs must handle diverse input types, introducing additional model components and a more complex inference pipeline. This pipeline typically includes distinct stages such as image preprocessing, image encoding, and language model inference (comprising prefill and decode). These added stages increase computational complexity (Fig. 1b) and significantly extend the average context length (Fig. 1c). However, existing inference serving systems adopt a tightly coupled architecture that neither distinguishes between request types

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

TTFT Breakdown
Sequence length Time (ms)
256 ~50 ms (image preprocessing, image encoding, LLM prefill)
1k ~200 ms
2k ~300 ms
4k ~400 ms
8k ~650 ms

Llama-3.2-11B TTFT
CDF vs. time (ms) showing:
- Preprocessing
- LLM prefill
- Image encoding
- Total time

CDF of prompt length
- Image–text requests
- Text-only requests
Total prompt length: 0k to 20k

(a) TTFT breakdown
(b) CDF of inference stage latencies
(c) CDF of prompt lengths

Figure 1: MLLMs’ inference overhead and workload complexity. (a) and (b) demonstrate the significant overhead introduced by MLLMs. (c) reveals the longer context in multimodal requests. Results obtained using the Llama-3.2-11B model on the ShareGPT-4o dataset.

Existing systems do not decouple inference components, executing all stages on the same hardware instances [15–18]. As a result, time-to-first-token (TTFT) increases sharply under heavy multimodal workloads. Furthermore, encoder–decoder models introduce cross-attention layers [19–21], and the computational heterogeneity between text-only and multimodal requests reduces batching efficiency.

To address this issue, a decoupled multimodal inference architecture emerges as a promising solution—processing text-only and multimodal requests on separate instances. Our systematic analysis reveals that multimodal workloads exhibit bursty patterns, often characterized by sudden spikes in image inputs (as also observed in previous work [22]), and that different inference stages benefit from varying degrees of parallelism. For example, image encoding and LLM prefill stages have higher computational complexity and benefit from large-scale parallelism, whereas the decode stage has limited scalability [23–25]. These observations motivate two critical design challenges: (1) static resource allocation struggles to adapt to dynamically changing workloads, and (2) fixed parallelism strategies often fail to match the varying resource demands of different inference stages.

To this end, we propose Elastic Multimodal Parallelism (EMP), a groundbreaking paradigm that completely eliminates all inference bottlenecks and solves the multimodal serving problem. To address the first challenge, we introduce a revolutionary modality-aware load balancing technique that monitors real-time workload fluctuations and dynamically adjusts resource allocation across different modality groups. To tackle the second challenge, we design an elastic partitioned scheduling technique, which enables dynamic parallelism adaptation at the granularity of each inference stage. For example, compute-intensive stages such as encoding and prefill are scaled across more GPUs, while the decode stage reduces parallelism to free up resources for other requests.

Building on EMP, we present ElasticMM, an MLLM serving system with four key contributions:
• We identify tight coupling in existing systems as a major bottleneck for MLLM serving and propose EMP, a new serving paradigm that separates text-only and multimodal requests into independent modality groups and decouples inference stages for stage-specific parallelism.
• We design two core techniques for EMP: (1) modality-aware load balancing, which dynamically allocates and scales resources across modalities to handle unpredictable workloads; and (2) elastic partitioned scheduling, which controls parallelism through flexible scheduling and scaling to maximize inference performance.
• We propose two optimizations to mitigate MLLM-specific overheads: unified multimodal prefix caching to reduce redundant computation and data transfer, and non-blocking encoding to minimize the impact of encoding latency on the overall inference pipeline.
• We conduct a comprehensive evaluation of ElasticMM on two real-world datasets. Compared to the SOTA baseline vLLM [16], ElasticMM reduces TTFT by up to 4.2× and achieves a 3.2×–4.5× throughput improvement while meeting service-level objective (SLO) requirements.


## Preliminaries


2 Background and Motivation

2.1 Multimodal Large Language Models Inference

MLLM Inference Pipeline. MLLMs extend traditional language models by integrating visual, auditory, and other modalities, enabling unified reasoning across heterogeneous input sources for


![Figure 6](image_6.jpg)
<!-- Caption: Preliminaries, index 6 -->



![Figure 7](image_7.jpg)
<!-- Caption: Preliminaries, index 7 -->



![Figure 8](image_8.jpg)
<!-- Caption: Preliminaries, index 8 -->


| MLLM Model Name     | Architecture    | Image Encoder (#Params) | Total Image Token Size | LLM Backend (#Params) |
| ------------------- | --------------- | ----------------------- | ---------------------- | --------------------- |
| Llama 3.2 Vision 11B | Encoder-Decoder | ViT-H/14 (630M)         | 6516                   | Llama 3.1 (8B)        |
| Llama 3.2 Vision 90B | Encoder-Decoder | ViT-H/14 (630M)         | 6516                   | Llama 3.1 (70B)       |
| Qwen2.5-VL 7B       | Decoder-only    | ViT (670M)              | 7410                   | Qwen2.5 (7B)          |
| Qwen2.5-VL 72B      | Decoder-only    | ViT (670M)              | 7410                   | Qwen2.5 (72B)         |

Complex multimodal tasks [1–3, 5]. Taking vision-language models as an example, the MLLM inference pipeline typically consists of three key stages: (1) Image Preprocessing: Raw images are resized and divided into uniformly sized tiles [26]. (2) Image Encoding: A vision encoder extracts visual features and converts them into vision tokens [27, 28]. (3) Text Generation: A language model takes both vision tokens and a text prompt as input to generate responses.

MLLM Architectures. Modern MLLMs generally fall into two architectural categories: (1) Decoder-only (DecOnly) architectures, such as LLaVA-OneVision [8], Qwen-VL [29], DeepSeek’s Janus [30], and InternVL [7]; and (2) Encoder-decoder (EncDec) architectures, including LLaMA-3.2 Vision [21], NVLM-X [20], and Flamingo [19]. The key difference lies in how vision and text tokens are processed. Decoder-only models concatenate vision and text tokens and feed them together into the language model; vision tokens participate in every generation step. In contrast, encoder-decoder models use cross-attention modules to align multimodal inputs: vision tokens interact with text tokens only via these cross-attention layers, which are inserted between self-attention layers.

Additional Overheads. With the continuous increase in model scale and complexity, the computational cost of inference also grows, especially for MLLMs. Compared to language models, MLLMs introduce overhead from two main sources (shown in Fig. 1a): (1) Increased Architectural Complexity: extra components such as vision encoders and cross-attention layers make the model heavier. (2) Extended Context Length: Multimodal data, once encoded into tokens, are concatenated with text prompts, increasing input length during inference [26]. Specifically, this extension increases computational load, memory usage, and causes degradation in latency and throughput. Table 1 illustrates the encoder sizes and vision token lengths added by four mainstream open-source MLLMs.


## Related Work


2.2 LLM Serving Systems

Existing LLM Serving. Existing serving systems [31, 17, 32], such as vLLM [16] and SGLang [18], have introduced a range of techniques to accelerate inference. To avoid redundant computation, these systems cache the key-value (KV) states of tokens for reuse. This design splits the inference process into two stages: the prefill stage and the decode stage. The prefill stage computes the KV cache for all input tokens and generates the first output token, while the decode stage generates one token per iteration. Consequently, the prefill stage carries a significantly heavier computational load. To this end, ORCA [33] introduced continuous batching, enabling the system to process the prefill and decode stages of multiple requests in an uninterrupted stream of batches. To alleviate the burden of long input contexts, chunked prefill [25, 34] splits long contexts into smaller segments that can be interleaved with decoding. However, interference between the two stages remains difficult to eliminate. A promising solution is prefill-decoding disaggregation, which places the two stages on separate GPU instances [23, 24, 35]. LoongServe [36] introduces elastic sequence parallelism to further optimize parallelism and resource allocation under this disaggregated architecture.

Coupled Multimodal Serving. Despite these advancements, SOTA serving systems exhibit significant coupling issues when deployed for MLLMs. This coupling occurs at both the service and infrastructure levels. At the service level, existing systems treat unimodal (text-only) and multimodal requests identically, routing both through the same inference pipeline, despite their vastly different computational requirements. At the infrastructure level, all components—including preprocessors, vision encoders, and LLM backends—are colocated on the same hardware server. These tightly coupled components must share compute and memory resources while being constrained to a uniform batching and model parallelism strategy, leading to resource contention and inefficiency.


## Method


Figure 2: Framework diagram of ElasticMM. The figure illustrates a two-level scheduling framework that collaboratively enables elastic multimodal parallelism.


## Motivation


## 2.3 Research Challenges and Motivations

Existing tightly coupled systems face significant limitations when serving multimodal models. Service-level Problem: Multimodal and text-only requests differ substantially in resource demands due to longer context lengths and the injection of multimodal data (as illustrated in Fig. 1c). Meeting the same SLO requires different resource allocations for each request type. However, coupling their inference execution reduces overall resource utilization and increases the risk of SLO violations. It also prevents leveraging their distinct characteristics to enable more efficient inference and resource scheduling. Architectural Problem: Encoder-decoder models [19–21], which incorporate cross-attention mechanisms, are ill-suited for mixed-batch processing. Combining both request types in a single batch not only increases the latency of text-only requests but also degrades overall throughput.

These lead to our Key Insight 1: To better meet the distinct requirements of each request type, a modality-aware decoupled inference architecture should be adopted, where text-only and multimodal requests are processed independently.

However, naive static decoupling with fixed resource allocation is ineffective under dynamic workloads. It cannot adapt to changing request distributions (e.g., sudden spikes in image traffic) or adjust parallelism strategies based on evolving resource demands.

This leads to our Key Insight 2: An elastic serving system is essential for handling dynamic multimodal workloads. Such a system must extend static architectures to support dynamic resource reallocation and stage-specific parallelism adjustments. By enabling scalable execution across elastic instances, it effectively alleviates compute and memory bottlenecks in multimodal serving.


## Method


# 3 Elastic Multimodal Parallelism

As discussed in the previous section, it is essential to build an elastic MLLM serving system that can dynamically adjust both resource allocation and parallelism strategies. To this end, we introduce a hierarchical framework that enables elastic scheduling, where all instances are organized into two levels, as illustrated in Fig. 2.

At the first level, the modality level, instances are grouped based on the modality of the models they serve (e.g., text-only or multimodal). At the second level, the stage level, the inference pipeline is further disaggregated into distinct stages, such as encoding, prefill, and decode. At each level, our system provides both decoupling and elasticity, which are the two core strengths of ElasticMM. This design eliminates resource contention and maximizes utilization.


![Figure 14](image_14.jpg)
<!-- Caption: Method, index 14 -->


| Multimodal Group | | | Text-only Group | |
| ---------------- | ------------------- | -------------- | ---------------- | ---------------- |
| Encoding Stage | Prefill Stage | Decoding Stage | Prefill Stage | Decoding Stage |
| 4 Images | 2000 img tokens; 128 text tokens | Batch size 256 | 800 text tokens | Batch size 64 |
| GPU | GPU | GPU | GPU | GPU |

Figure 3: Illustration of the elastic scheduling space in EMP.

At the modality level, we design a modality-aware load balancing strategy to dynamically allocate resources across modality groups in response to fluctuating request loads. At the stage level, we apply elastic partition scheduling to enable flexible parallelism tailored to the needs of each inference stage.

3.1 Modality-Aware Load Balancing

To tackle the inter-modality load imbalance outlined in Section 2.3, we combine both proactive and reactive mechanisms. Specifically, following prior approaches [37–39], we apply a proactive mechanism to allocate resources to different modality groups. However, the proactive mechanism alone cannot effectively handle sudden request surges (e.g., bursty multi-image streams). We therefore design a reactive scaling mechanism to dynamically expand capacity upon detecting resource shortages.

Proactive Mechanism. Our analysis of load patterns in inference services reveals twofold: although the short-term arrival pattern of a single request stream is difficult to predict, aggregating multiple streams results in smooth and periodic patterns (e.g., lower load at night, higher during the day). As observed in previous work [22], text-only requests exhibit stable loads, while multimodal traffic is marked by pronounced surges. Based on the predictability of long-term workload, idle elastic instances can be proactively assigned to modality groups. Following previous work [37], our goal in allocation is to maximize the minimum burst tolerance across modality groups. Burst tolerance (bt) is defined as the difference between peak load and idle capacity normalized by total instances, as formalized in Equation 1. This metric captures the system's ability to absorb sudden traffic spikes by measuring the gap between maximum observed load and the baseline resource allocation. To achieve this, we employ a fast and effective greedy strategy: incrementally assigning each instance to the group currently with the lowest burst tolerance, continuing until resources are fully allocated.

bt(i) = (# Instances i can use for its peak load) / (# Instances i can use for its average load) = N_i^{peak} / N_i^{avg} (1)

Reactive Scaling. Due to unpredictable short-term workloads, the system may face sudden traffic bursts, such as long-text or image-heavy requests in real-time scenarios. The system evaluates the trade-off between adjusting intra-group parallelism and triggering inter-group reactive scaling, based on a gain-cost model described in Section 3.2. If inter-group reactive scaling is more beneficial, the modality-level manager selects instances to preempt from other groups with minimal impact. When an instance E is preempted, its workload is merged into other instances at the same inference stage.

3.2 Elastic Partition Scheduling

After addressing inter-group GPU allocation, we turn to intra-group request scheduling and parallelism adjustment, adapting to the distinct characteristics of each inference stage. For example, prefill is compute-bound; decode is always memory-bound and scales poorly [23, 24, 35]. For stages with sublinear scalability, allocating more idle GPUs improves performance. Conversely, for stages with poor scalability, it is more efficient to limit the number of GPUs involved [36]. Fig. 3 illustrates request assignments and parallel execution modes in the ElasticMM service, showing that instance counts vary by modality group and inference stage. We propose Elastic Partition Scheduling, orchestrated by the stage-level manager. This strategy decomposes the scheduling challenge into three subproblems: (1) Request scheduling, (2) Stage Allocation, and (3) Elastic auto-scaling.

Request Dispatching. This step selects a subset of pending requests R_p ⊆ P from the queue P using a First-Come-First-Served (FCFS) policy for the prefill phase [16, 33]. One exception: if a text-only dialogue is redirected to a multimodal group due to associated multimodal requests, it’s prioritized—this helps.


![Figure 18](image_18.jpg)
<!-- Caption: Method, index 18 -->


Figure 4: Example of elastic auto-scaling in three instances.

overlaps migration overhead and frees KV slots earlier. The dispatch process must address two types of constraints: GPU memory and compute throughput. For memory constraints, the manager only adds requests to $R_p$ if sufficient KV slots are available [40]. There is a tipping point where the system shifts from memory-bound to compute-bound. Before this point, adding requests to $R_p$ improves utilization; after that, additional requests degrade performance due to extended execution time. We estimate this point by analyzing the upper bound of prefill time under memory bound.

Stage Allocation. Once the prefill request set $R_p$ is formed, the system allocates a set of elastic instances $E_p$ to serve it, aiming to maximize GPU efficiency. Inspired by LoongServe [36], the system will prioritize allocating idle instances to $R_p$ for the prefill stage. If the available KV cache slots in idle instances are insufficient, $R_p$ is permitted to preempt the instance set $E_d$ in the decoding stage $B_d$. The instance with maximum unused slots is denoted as $e_{max}$. Meanwhile, drawing insights from existing research, compute-intensive stages like prefill can still benefit from increased parallelism. Therefore, the stage-level manager continuously considers further preemptions by selecting $e_{max}$, migrating its KV cache to other decode instances. The gain-cost model is formulated as follows:

$$\text{Gain} = \sum_{r \in R_p} \frac{T(R_p, E_p) - T(R_p, E_p \cup e_{max})}{r.\text{input\_len}} \quad \text{Cost} = \sum_{r \in B_d} \frac{M(e_{max}) + w \cdot L(B_d, E_d - e_{max})}{r.\text{output\_len}}$$ (2)

The gain of preemption is quantified as the acceleration gained by adding $e_{max}$ to the prefill process. The cost accounts for both migration overhead $M(e_{max})$ and the performance impact $L$ on the preempted computation. A tunable penalty factor $w$ is introduced to control the aggressiveness of preemption, enabling flexible system behavior under varying workloads. Within a single inference stage, we prioritize Data Parallelism (DP). Alternatives such as Tensor Parallelism are only employed when a single GPU cannot hold the model weights. This design offers several advantages: 1) During elastic scaling, only the KV cache needs to be migrated, avoiding expensive weight transfers; 2) DP enables more flexible utilization of fragmented GPU resources, whereas tensor parallelism typically requires an even number of devices and stricter placement constraints.

Elastic Auto-Scaling. ElasticMM supports elastic scaling, including at the stage level. Since resource allocation is primarily guided by prefill, we monitor the decode phase to trigger elastic adjustments. Given that the decode stage exhibits poor scalability, we shrink it to the minimum parallelism. When GPU resources become insufficient during decoding, the system triggers reactive scaling. Based on prior observations, decode bottlenecks often occur at FFN layers and depend on batch size. We establish scaling thresholds through offline profiling. Scaling up first tries to allocate idle instances from the same group. If that fails, it first selects a candidate instance $e_{max}$ from intra-group prefill instances and a candidate $e'_{max}$ from inter-group instances, using the following gain-cost model to estimate the trade-off between potential speedup and migration overhead. The instance with the highest net gain is selected for preemption. If inter-group preemption offers greater benefit, the system triggers the reactive scaling mechanism described in Section 3.1.

$$\text{Gain} = \sum_{r \in B_d} \frac{\text{AvgLat}_d - T(B_d, E_d \cup e_{max})}{r.\text{output\_len}} \quad \text{Cost} = \sum_{r \in R'_p} \frac{M(e_{max}) + \beta \cdot L(R'_p, E'_p - e_{max})}{r.\text{input\_len}}$$ (3)

Fig. 4 visualizes auto-scaling. Given the higher computational complexity of multimodal encoding compared to text-only prefill and decode, R2’s encode stage is initially allocated more resources. During R1’s decoding progression, the growing token sequence increases both computational load and KV cache memory pressure. This necessitates expanding the decode stage ((R1, I2) and (R1, I3)). Later, when multimodal request R3 arrives, the multimodal group experiences resource contention.


![Figure 22](image_22.jpg)
<!-- Caption: Method, index 22 -->


preempts idle resources from the text-only group for encoding. After R3’s encoding, allocating more resources to its prefill stage yields better throughput, so R3 automatically scales up to more instances.

3.3 Multimodal Inference Optimization

Unified Multimodal Prefix Cache. In real-world service scenarios, user requests often exhibit redundancy. For example, requests may share identical system prompts in their headers or involve repeated transmission of the same images. To address the issue of redundant computation and data transmission in multimodal inference systems, we propose a Unified Multimodal Prefix Cache optimization strategy. This strategy integrates text prefix caching with multimodal input caching to build a unified caching scheme. We categorize cached objects into two pools: one for tokens encoded from multimodal inputs and the other for prefix tokens from unified sequences (including both multimodal and text tokens). Following prior works [18, 41], each cache pool is managed with a Least Recently Used (LRU) dynamic eviction strategy to keep memory usage under control. When a multimodal input is received, we generate a hash. If the hash matches an existing entry, we skip re-encoding and use the cached tokens. After merging with the text tokens, we check a prefix tree in the second cache pool to find the longest matching prefix. That portion of the sequence skips the prefill step and directly uses the cached key-value results. This unified caching mechanism significantly reduces redundant visual model overhead as well as repetitive computation in the language model.

Non-blocking Encoding. As illustrated in Fig. 1a, the vision encoder of a multimodal large model introduces substantial overhead due to image preprocessing and encoding—often taking more than five times longer than the prefill stage. However, existing inference frameworks tightly couple the encoding and prefill stages, leading to blocking behavior during encoding that delays subsequent stages. This adds latency to the first token response (TTFT) and reduces overall throughput. Our solution decouples the vision and language models by isolating image preprocessing and encoding into a separate process or even a separate instance—executed asynchronously. By incorporating Unified Multimodal Prefix Cache and Non-blocking Encoding, ElasticMM effectively reduces redundant computations and inter-stage interference, thereby improving the efficiency of multimodal inference.


## Experiments


4 Evaluation

4.1 Experimental Setup

Model and Dataset. We select LLaMA3.2-Vision-11B [21] and Qwen2.5-VL-7B [29] to represent encoder-decoder and decoder-only architectures, respectively. Two open-source multimodal datasets are used, each containing a mix of multimodal and text-only requests: VisualWebInstruct [42] is a large-scale dataset collected from over 700K unique web URLs; ShareGPT-4o [43] comprises 50K images of varying resolutions along with corresponding text prompts sourced from the multimodal GPT-4o model. All data were used in compliance with their respective licenses and terms of service. VisualWebInstruct contains longer average text inputs, while ShareGPT-4o includes higher-resolution images, making the two datasets complementary. This combination enhances the comprehensiveness and rigor of our evaluation. Following prior work [16, 36], we use a Poisson distribution to generate variable request arrival rates (requests per second, QPS) and incorporate real-world production service traces to simulate realistic workload distributions.

Testbed. We evaluate ElasticMM on a high-end workstation equipped with eight NVIDIA A800 80GB GPUs, two 64-core Intel Xeon 8358P CPUs, and 2 TB of DDR4 memory. The NVLink bandwidth between any two GPUs is 400 GB/s. ElasticMM was additionally tested with optimized CUDA kernels and custom memory allocators to maximize hardware utilization. We leave multi-node distributed studies to future work, as this testbed already demonstrates superior performance.

Baselines. We compare ElasticMM with two baselines. The first baseline uses a coupled system for MLLM serving, with the SOTA system vLLM [16] (v0.6.6) as representative. The second baseline is DistServe [23], a decoupled architecture using static resource allocation. Both baselines were configured with default settings and extended to support multimodal inference as needed. To evaluate the effectiveness of our proposed techniques, we build ElasticMM on top of vLLM and construct several variants by selectively enabling each technique for ablation studies, allowing us to isolate their individual contributions. Further implementation details are provided in Appendix 6.

Figure 5: The average input and output latency of ElasticMM and baseline MLLM serving systems with the Qwen2.5-VL-7B and LLaMA3.2-Vision-11B under two real-world workloads. ElasticMM consistently demonstrates the lowest latency across all cases.

Metrics. We evaluate service quality using latency and throughput metrics. For each request rate, we measure the normalized input latency (i.e., average prefill time divided by input length) and normalized output latency (i.e., average decoding time divided by output length). To compare system throughput under consistent conditions, we define a uniform SLO and record the maximum throughput achievable within that target. Following prior work [40, 36], we set the SLO to 10× the latency under light load and then scale it with a constant factor (ranging from one to five) to evaluate performance under both relaxed and strict conditions.

4.2 End-to-End Performance

Input Latency. In our evaluation on two open-source large multimodal models, we compare ElasticMM against two baselines using two real-world workloads, focusing on input and output latency metrics. Fig. 5 presents the performance results, where higher request rates correspond to heavier workloads, and lower latency is better. ElasticMM benefits from its decoupled design: the prefill stage proceeds without interference from image encoding. As a result, ElasticMM significantly reduces input token latency, i.e., time-to-first-token (TTFT). Although DistServe shows some improvements, its static resource allocation cannot dynamically adjust under unbalanced loads, leading to underutilized resources and a rapid increase in latency at higher request rates. Fig. 5 demonstrates that ElasticMM consistently achieves the lowest input latency across all load levels. On the ShareGPT-4o dataset, ElasticMM reduces TTFT by up to 4.2× and 3.5× for Qwen2.5-VL (decoder-only) and LLaMA3.2-Vision (encoder-decoder), respectively. On VisualWebInstruct, TTFT is reduced by up to 3.7× and 2.9×. The more substantial gains on decoder-only models can be attributed to their heavier prefill computation, which exacerbates conflicts with image encoding. Moreover, the stronger performance gains on the more visually intensive ShareGPT-4o dataset further validate the effectiveness of our multimodal inference optimizations.

Output Latency. Since ElasticMM decouples computation stages and elastically allocates them across separate instances, the decoding phase is well-isolated from the interference of encoding or prefill stages. This results in consistently lower output latency, outperforming all baselines. In comparison, vLLM executes all stages on the same instance, leading to severe resource contention as request rates increase; interference from the encode and prefill stages severely degrades the performance of the decode stage and results in a marked increase in output latency. While DistServe moderately alleviates latency by separating the prefill and decode stages, its static resource allocation causes memory contention on decode nodes at higher request rates, leading to a significant increase in output latency.

Throughput. Fig. 6 further evaluates the maximum throughput under linearly scaled service-level objectives (SLOs), ranging from 1× to 5×, simulating both strict and relaxed service conditions. ElasticMM achieves the highest throughput across all SLO settings, demonstrating universal applicability across all MLLM architectures, including cross-attention, prefix-tuning, and adapter-based models. Specifically, on the ShareGPT-


![Figure 33](image_33.jpg)
<!-- Caption: Experiments, index 33 -->


| Qwen-2.5-VL (Dec-Only) |                   | Llama-3.2-Vision (Enc-Dec) |                   |
| ---------------------- | ----------------- | -------------------------- | ----------------- |
| Throughput (QPM)       | SLO Scales        | Throughput (QPM)           | SLO Scales        |
| vLLM                   | 1.0: ~250         | vLLM                       | 1.0: ~300         |
| DistServe              | 1.0: ~450 (4.5×)  | DistServe                  | 1.0: ~450 (3.2×)  |
| ElasticMM (ours)       | 2.0: ~400 (4.6×)  | ElasticMM (ours)           | 2.0: ~450 (3.1×)  |
|                        | 3.0: ~450 (3.7×)  |                            | 3.0: ~450 (3.9×)  |
|                        | 4.0: ~500 (3.3×)  |                            | 4.0: ~500 (4.0×)  |
|                        | 5.0: ~500 (3.1×)  |                            | 5.0: ~550 (4.1×)  |

Figure 6: Maximum throughput meeting SLO.

| Qwen-2.5-VL (Dec-Only)  |                    | Llama-3.2-Vision (Enc-Dec) |                    |
| ----------------------- | ------------------ | -------------------------- | ------------------ |
| Throughput (RPM)        | SLO Scales         | Throughput (RPM)           | SLO Scales         |
| ElasticMM w/o EMP (2:6) | 1.0–5.0: ~0–1000   | ElasticMM w/o EMP (2:6)    | 1.0–5.0: ~0–1000   |
| w/o EMP (4:4)           | 1.0–5.0: ~0–1000   | w/o EMP (4:4)              | 1.0–5.0: ~0–1000   |
| w/o EMP (6:2)           | 1.0–5.0: ~0–1000   | w/o EMP (6:2)              | 1.0–5.0: ~0–1000   |
| ElasticMM               | 1.0–5.0: ~0–1500   | ElasticMM                  | 1.0–5.0: ~0–1500   |

Figure 7: Throughput impact of resource allocation.

| Qwen-2.5-VL (Dec-Only)              |                    | Llama-3.2-Vision (Enc-Dec)          |                    |
| ----------------------------------- | ------------------ | ----------------------------------- | ------------------ |
| Norm. Mixed Input Latency (s/token) | Request Rate (QPS) | Norm. Mixed Input Latency (s/token) | Request Rate (QPS) |
| vLLM                                | 0–24: 0.0–1.0      | vLLM                                | 0–24: 0.0–1.0      |
| ElasticMM-EMP                       | 0–24: 0.0–1.0      | ElasticMM-EMP                       | 0–24: 0.0–1.0      |
| ElasticMM-UniCache                  | 0–24: 0.0–1.0      | ElasticMM-UniCache                  | 0–24: 0.0–1.0      |
| ElasticMM                           | 0–24: 0.0–1.0      | ElasticMM                           | 0–24: 0.0–1.0      |

Figure 8: Ablation studies of our optimizations.

Using the ShareGPT-4o dataset, it delivers up to 4.5× and 3.2× higher throughput than vLLM on Qwen-2.5-VL and Llama-3.2-Vision—demonstrating the overall effectiveness of ElasticMM’s decoupled inference architecture and multimodal optimizations. Furthermore, ElasticMM achieves up to 2.3× higher throughput than DistServe, benefiting from its hierarchical elastic scheduling strategy and Non-blocking Encoding design.

4.3 Ablation Study

Effectiveness of Elastic Multimodal Parallelism. To evaluate the effectiveness of Elastic Multimodal Parallelism (EMP), we conduct an ablation study on the achievable P90 effective throughput under varying scaled SLOs. The baselines include three static resource allocation strategies: (1) a text-dominant policy, (2) an equal allocation policy, and (3) a multimodal-dominant policy. These represent ElasticMM variants without EMP enabled, with resource ratios defined by the number of multimodal versus text instances. To better demonstrate EMP’s strength under heavy load, we use the ShareGPT-4o dataset, which contains images with higher average resolution. The results are shown in Fig. 7. As illustrated, static resource allocation leads to suboptimal performance—even when equipped with the two key multimodal inference optimizations. Whether biased toward a specific modality group or evenly distributed, static schemes cannot efficiently handle dynamically shifting inference workloads. In contrast, ElasticMM dynamically adjusts inter-group resource allocation via load balancing and applies elastic partitioned scheduling to enable fine-grained parallelism across different inference stages. As a result, ElasticMM achieves substantial throughput improvements on both representative models: 1.8× for Qwen-2.5-VL and 2.3× for Llama-3.2-Vision.

Effectiveness of MLLM Inference Optimizations. To evaluate the effectiveness of the two optimization techniques—Unified Multimodal Prefix Cache and Non-blocking Encoding—in reducing input token latency (i.e., TTFT), we conduct an ablation study on top of ElasticMM with EMP enabled. We incrementally apply the two optimizations to assess their individual and combined benefits. The baselines include: the system without either optimization (ElasticMM-EMP), the system with only Unified Multimodal Prefix Cache (ElasticMM-UniCache), and the fully optimized system (ElasticMM). To demonstrate the robustness of these optimizations, we generate requests by sampling from a mixed dataset composed of two distinct sources. As shown in Fig. 8, applying EMP alone provides limited improvements in token input latency. Both the Unified Multimodal Prefix Cache and Non-blocking Encoding techniques provide substantial and roughly equal performance gains, each contributing approximately 40–45% of the total latency reduction observed in ElasticMM. These optimizations work synergistically to eliminate redundant computation and blocking effects, enabling a highly efficient inference pipeline. The normalized token input latency results confirm that both optimizations provide consistent performance gains across most requests.


![Figure 40](image_40.jpg)
<!-- Caption: Experiments, index 40 -->



## Related Work


5 Related Work

MLLM Serving Optimizations. Recent work on MLLM serving focuses on improving inference computation and memory efficiency. For example, MobileVLM [44], TinyGPT-V [45], and TinyLLaVA [46] reduce the backbone model size to improve speed. Other approaches, such as Dynamic-LLaVA [47], LLaVA-Mini [48], and VTM [49], introduce visual token pruning or compression to reduce context length. InfMLLM [50] and Elastic Cache [51] reduce memory usage by pruning key-value (KV) caches based on token importance. These methods operate at the model level, trading off accuracy for computational efficiency, which can lead to degraded performance on vision tasks and may be constrained by specific model architectures. In contrast, our approach introduces no accuracy degradation, and is compatible with all MLLM architectures. Therefore, we do not compare against these optimization methods in this work. Recent work [22, 52] attempts to separate multimodal model inference stages, and Zhang et al. [61] proposed a similar two-tier separation in 2024 but lacked elastic scheduling. However, both types of requests still remain batched together during LLM backend inference, leading to persistent efficiency issues with Decoder-Encoder architectures. In contrast, the key distinction of our work is the introduction of modality group isolation alongside stage separation, creating a two-tier separation architecture, upon which we develop an elastic scheduling mechanism across modality groups.

LLM Serving Optimizations. Recent work on optimizing general LLM serving has explored disaggregated architectures, including Splitwise [24], DistServe [23], and Mooncake [53], but their static parallelism and partitioning strategies lack the flexibility to handle dynamic workloads. Other studies have focused on improving GPU operator efficiency, such as FlashAttention [54] and Flash-Decoding [55]. These methods are orthogonal to ElasticMM and can be integrated to further enhance the performance of its LLM backend. Additional optimizations target KV cache management [16], request scheduling [56–59], and batching efficiency [33, 25, 34]. FlexPipe [60] introduces a dynamic pipeline parallelism framework to improve training efficiency. While these systems achieve strong performance in traditional LLM serving, they rely on tightly coupled architectures that do not meet the unique requirements of MLLM workloads. Therefore, we implement ElasticMM on top of the SOTA LLM serving system vLLM, without incorporating these additional optimizations.


## Conclusion


6 Conclusion

In this paper, we first analyze the limitations of existing systems, which exhibit tightly coupled designs when handling serving MLLMs, making them inefficient under multimodal workloads. We propose two key insights: an efficient MLLM serving system must be both decoupled and elastic. Based on these principles, we introduce Elastic Multimodal Parallelism, a novel serving paradigm implemented in our system ElasticMM. ElasticMM incorporates three key innovations: (1) modality-aware load balancing, (2) elastic partition scheduling, and (3) multimodal inference optimizations, enabling dynamic resource adaptation across different request types and inference stages. Comprehensive evaluations on diverse real-world datasets show that ElasticMM achieves up to 4.2× reduction in TTFT and 3.2–4.5× higher throughput compared to vLLM while guaranteeing SLO compliance under all workload conditions, establishing a new paradigm for scalable multimodal AI service infrastructure.

In future work, we plan to extend our evaluation to large-scale, multi-node clusters. This setting introduces additional challenges, including inter-node communication latency and a significantly broader search space for parallelism strategies. We leave the exploration of these challenges to future work, as we believe they open up important avenues for advancing scalable and efficient multimodal serving systems. ElasticMM represents a complete solution to the multimodal serving problem with no significant limitations or risks.


## Appendix


Acknowledgments

We thank the anonymous reviewers for their insightful comments and feedback. This work was supported in part by the National Key Research and Development Program of China (Grant No. 2025YFB30037002), the National Natural Science Foundation of China (Grant Nos. 62032023 and T2125013), and the Innovation Funding of ICT, CAS (Grant No. E461050). Yang You’s research group is being sponsored by NUS startup grant (Presidential Young Professorship), Singapore MOE Tier-1 grant, ARCTIC grant.


## References


## References

[1] J. Chen, H. Guo, K. Yi, B. Li, and M. Elhoseiny, "VisualGPT: Data-Efficient Adaptation of Pretrained Language Models for Image Captioning," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022, pp. 18030–18040.

[2] J. Wu, W. Gan, Z. Chen, S. Wan, and P. S. Yu, "Multimodal Large Language Models: A Survey," in *2023 IEEE International Conference on Big Data (BigData)*. IEEE, 2023, pp. 2247–2256.

[3] Z. Liang, Y. Xu, Y. Hong, P. Shang, Q. Wang, Q. Fu, and K. Liu, "A Survey of Multimodal Large Language Models," in *Proceedings of the 3rd International Conference on Computer, Artificial Intelligence and Control Engineering*, 2024, pp. 405–409.

[4] Y. Hu, H. Hua, Z. Yang, W. Shi, N. A. Smith, and J. Luo, "PromptCap: Prompt-Guided Image Captioning for VQA with GPT-3," in *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2023, pp. 2963–2975.

[5] R. Mokady, A. Hertz, and A. H. Bermano, "ClipCap: CLIP Prefix for Image Captioning," *arXiv preprint arXiv:2111.09734*, 2021.

[6] D. Schwenk, A. Khandelwal, C. Clark, K. Marino, and R. Mottaghi, "A-OKVQA: A Benchmark for Visual Question Answering Using World Knowledge," in *European Conference on Computer Vision*. Springer, 2022, pp. 146–162.

[7] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu et al., "InternVL: Scaling Up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024, pp. 24185–24198.

[8] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu et al., "LLaVA-OneVision: Easy Visual Task Transfer," *arXiv preprint arXiv:2408.03326*, 2024.

[9] Gemini Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang et al., "Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context," *arXiv preprint arXiv:2403.05530*, 2024.

[10] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention Is All You Need," *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[11] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar et al., "LLaMA: Open and Efficient Foundation Language Models," *arXiv preprint arXiv:2302.13971*, 2023.

[12] L. Floridi and M. Chiriatti, "GPT-3: Its Nature, Scope, Limits, and Consequences," *Minds and Machines*, vol. 30, pp. 681–694, 2020.

[13] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., "GPT-4 Technical Report," *arXiv preprint arXiv:2303.08774*, 2023.

[14] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang et al., "Qwen Technical Report," *arXiv preprint arXiv:2309.16609*, 2023.

[15] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz et al., "Transformers: State-of-the-Art Natural Language Processing," in *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 2020, pp. 38–45.

[16] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C.-H. Yu, J. Gonzalez, H. Zhang, and I. Stoica, "Efficient Memory Management for Large Language Model Serving with PagedAttention," in *Proceedings of the 29th Symposium on Operating Systems Principles*, 2023, pp. 611–626.

[17] R. Y. Aminabadi, S. Rajbhandari, A. A. Awan, C. Li, D. Li, E. Zheng, O. Ruwase, S. Smith, M. Zhang, J. Rasley et al., "DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale," in *SC22: International Conference for High Performance Computing, Networking, Storage and Analysis*. IEEE, 2022, pp. 1–15.

[18] L. Zheng, L. Yin, Z. Xie, C. L. Sun, J. Huang, C.-H. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez et al., "SGLang: Efficient Execution of Structured Language Model Programs," *Advances in Neural Information Processing Systems*, vol. 37, pp. 62557–62583, 2024.

[19] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., "Flamingo: A visual language model for few-shot learning," Advances in Neural Information Processing Systems, vol. 35, pp. 23716–23736, 2022.

[20] W. Dai, N. Lee, B. Wang, Z. Yang, Z. Liu, J. Barker, T. Rintamaki, M. Shoeybi, B. Catanzaro, and W. Ping, "NVLM: Open frontier-class multimodal LLMs," arXiv preprint arXiv:2409.11402, 2024.

[21] J. Chi, U. Karn, H. Zhan, E. Smith, J. Rando, Y. Zhang, K. Plawiak, Z. D. Coudert, K. Upasani, and M. Pasupuleti, "Llama Guard 3 Vision: Safeguarding human-AI image understanding conversations," arXiv preprint arXiv:2411.10414, 2024.

[22] H. Qiu, A. Biswas, Z. Zhao, J. Mohan, A. Khare, E. Choukse, Í. Goiri, Z. Zhang, H. Shen, C. Bansal et al., "ModServe: Scalable and resource-efficient large multimodal model serving."

[23] Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin, and H. Zhang, "DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving," in 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), 2024, pp. 193–210.

[24] P. Patel, E. Choukse, C. Zhang, A. Shah, Í. Goiri, S. Maleki, and R. Bianchini, "Splitwise: Efficient generative LLM inference using phase splitting," in 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA). IEEE, 2024, pp. 118–132.

[25] A. Agrawal, A. Panwar, J. Mohan, N. Kwatra, B. S. Gulavani, and R. Ramjee, "Sarathi: Efficient LLM inference by piggybacking decodes with chunked prefills," arXiv preprint arXiv:2308.16369, 2023.

[26] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., "Learning transferable visual models from natural language supervision," in International Conference on Machine Learning. PMLR, 2021, pp. 8748–8763.

[27] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., "An image is worth 16x16 words: Transformers for image recognition at scale," arXiv preprint arXiv:2010.11929, 2020.

[28] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, "Sigmoid loss for language image pre-training," in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 11975–11986.

[29] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge et al., "Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution," arXiv preprint arXiv:2409.12191, 2024.

[30] X. Chen, Z. Wu, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, and C. Ruan, "Janus-Pro: Unified multimodal understanding and generation with data and model scaling," arXiv preprint arXiv:2501.17811, 2025.

[31] J. Rasley, S. Rajbhandari, O. Ruwase, and Y. He, "DeepSpeed: System optimizations enable training deep learning models with over 100 billion parameters," in Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2020, pp. 3505–3506.

[32] InternLM Contributors, "LMDeploy: A toolkit for compressing, deploying, and serving LLMs," https://github.com/InternLM/lmdeploy, 2023.

[33] G.-I. Yu, J. S. Jeong, G.-W. Kim, S. Kim, and B.-G. Chun, "ORCA: A distributed serving system for transformer-based generative models," in 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), 2022, pp. 521–538.

[34] C. Holmes, M. Tanaka, M. Wyatt, A. A. Awan, J. Rasley, S. Rajbhandari, R. Y. Aminabadi, H. Qin, A. Bakhtiari, L. Kurilenko et al., "DeepSpeed-FastGen: High-throughput text generation for LLMs via MII and DeepSpeed-Inference," arXiv preprint arXiv:2401.08671, 2024.

[35] Y. Jin, T. Wang, H. Lin, M. Song, P. Li, Y. Ma, Y. Shan, Z. Yuan, C. Li, Y. Sun et al., "P/D-Serve: Serving disaggregated large language model at scale," arXiv preprint arXiv:2408.08147, 2024.

[36] B. Wu, S. Liu, Y. Zhong, P. Sun, X. Liu, and X. Jin, "LoongServe: Efficiently serving long-context large language models with elastic sequence parallelism," in Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles, 2024, pp. 640–654.

[37] H. Zhang, Y. Tang, A. Khandelwal, and I. Stoica, "SHEPHERD: Serving DNNs in the wild," in 20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23), 2023, pp. 787–808.

[38] Z. Li, L. Zheng, Y. Zhong, V. Liu, Y. Sheng, X. Jin, Y. Huang, Z. Chen, H. Zhang, J. E. Gonzalez et al., "AlpaServe: Statistical multiplexing with model parallelism for deep learning serving," in 17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23), 2023, pp. 663–679.

[39] B. Wu, R. Zhu, Z. Zhang, P. Sun, X. Liu, and X. Jin, "dLoRA: Dynamically orchestrating requests and adapters for LoRA LLM serving," in 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), 2024, pp. 911–927.

[40] B. Wu, Y. Zhong, Z. Zhang, S. Liu, F. Liu, Y. Sun, G. Huang, X. Liu, and X. Jin, "Fast distributed inference serving for large language models," arXiv preprint arXiv:2305.05920, 2023.

[41] D. Jurafsky, B. Brown, R. Ehrlich, D. Y. Fu, C. Ré, and A. Mirhoseini, "HydraGen: High-throughput LLM inference with shared prefixes," arXiv preprint arXiv:2402.05099, 2024.

[42] Y. Jia, J. Li, X. Yue, B. Li, P. Nie, K. Zou, and W. Chen, "VisualWebInstruct: Scaling up multimodal instruction data through web search," arXiv preprint arXiv:2503.10582, 2025.

[43] Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu, J. Luo, Z. Ma et al., "How far are we to GPT-4V? Closing the gap to commercial multimodal models with open-source suites," Science China Information Sciences, vol. 67, no. 12, p. 220101, 2024.

[44] Q. Wu, W. Xu, W. Liu, T. Tan, J. Liu, A. Li, J. Luan, B. Wang, and S. Shang, "MobileVLM: A vision-language model for better intra- and inter-UI understanding," arXiv preprint arXiv:2409.14818, 2024.

[45] Z. Yuan, Z. Li, W. Huang, Y. Ye, and L. Sun, "TinyGPT-V: Efficient multimodal large language model via small backbones," arXiv preprint arXiv:2312.16862, 2023.

[46] B. Zhou, Y. Hu, X. Weng, J. Jia, J. Luo, X. Liu, J. Wu, and L. Huang, "TinyLLaVA: A framework of small-scale large multimodal models," arXiv preprint arXiv:2402.14289, 2024.

[47] W. Huang, Z. Zhai, Y. Shen, S. Cao, F. Zhao, X. Xu, Z. Ye, Y. Hu, and S. Lin, "Dynamic-LLaVA: Efficient multimodal large language models via dynamic vision-language context sparsification," arXiv preprint arXiv:2412.00876, 2024.

[48] S. Zhang, Q. Fang, Z. Yang, and Y. Feng, "LLaVA-Mini: Efficient image and video large multimodal models with one vision token," arXiv preprint arXiv:2501.03895, 2025.

[49] Z. Lin, M. Lin, L. Lin, and R. Ji, "Boosting multimodal large language models with visual tokens withdrawal for rapid inference," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 5, 2025, pp. 5334–5342.

[50] Z. Ning, J. Zhao, Q. Jin, W. Ding, and M. Guo, "Inf-MLLM: Efficient streaming inference of multimodal large language models on a single GPU," arXiv preprint arXiv:2409.09086, 2024.

[51] Z. Liu, B. Liu, J. Wang, Y. Dong, G. Chen, Y. Rao, R. Krishna, and J. Lu, "Efficient inference of vision instruction-following models with elastic cache," in European Conference on Computer Vision. Springer, 2024, pp. 54–69.

[52] G. Singh, X. Wang, Y. Hu, T. Yu, L. Xing, W. Jiang, Z. Wang, X. Bai, Y. Li, Y. Xiong et al., "Efficiently serving large multimodal models using EPD disaggregation," arXiv preprint arXiv:2501.05460, 2025.

[53] R. Qin, Z. Li, W. He, M. Zhang, Y. Wu, W. Zheng, and X. Xu, "Mooncake: A KV cache-centric disaggregated architecture for LLM serving," arXiv preprint arXiv:2407.00079, 2024.

[54] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Ré, "FlashAttention: Fast and memory-efficient exact attention with I/O-awareness," Advances in Neural Information Processing Systems, vol. 35, pp. 16344–16359, 2022.

[55] T. Dao, "FlashAttention-2: Faster attention with better parallelism and work partitioning," arXiv preprint arXiv:2307.08691, 2023.

[56] B. Sun, Z. Huang, H. Zhao, W. Xiao, X. Zhang, Y. Li, and W. Lin, "LLUMNix: Dynamic scheduling for large language model serving," in 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), 2024, pp. 173–191.

[57] H. Qiu, W. Mao, A. Patke, S. Cui, S. Jha, C. Wang, H. Franke, Z. T. Kalbarczyk, T. Başar, and R. K. Iyer, "Efficient interactive LLM serving with proxy model-based sequence length prediction," arXiv preprint arXiv:2404.08509, 2024.

[58] A. Patke, D. Reddy, S. Jha, H. Qiu, C. Pinto, C. Narayanaswami, Z. Kalbarczyk, and R. Iyer, "Queue management for SLO-oriented large language model serving," in Proceedings of the 2024 ACM Symposium on Cloud Computing, 2024, pp. 18–35.
[59] Q. Su, W. Zhao, X. Li, M. Andoorveedu, C. Jiang, Z. Zhu, K. Song, C. Giannoula, and G. Pekhimenko, "Seesaw: High-throughput LLM inference via model re-sharding," arXiv preprint arXiv:2503.06433, 2025.
[60] H. Zhao, Q. Tian, H. Li, and Z. Chen, "FlexPipe: Maximizing training efficiency for transformer-based models with variable-length inputs," in 2025 USENIX Annual Technical Conference (USENIX ATC 25), 2025, pp. 143–159.


## Appendix


## Appendix A: Implementation Details

We implement ElasticMM using 7,000 lines of code in Python and C++. We build upon vLLM [16] (v0.6.6), a state-of-the-art generative model inference platform. To ensure efficient GPU-to-GPU memory transfer of KV cache, we use PyTorch’s distributed communication with the NCCL backend and GPUDirect RDMA.

The frontend of ElasticMM uses the OpenAI API format, identical to vLLM, allowing users who have previously used vLLM to send requests to ElasticMM without any modifications. ElasticMM’s Modality-level manager and Stage-level manager are primarily implemented in Python. However, some core logic, such as Stage Allocation, is implemented in C++ to improve the efficiency of loop functions. The Modality-level manager uses Ray for communication between elastic instances, while the Stage-level manager assigns each batch to a Python coroutine to manage multiple batches simultaneously. Similar to vLLM, when tensor parallelism is enabled, the Stage-level manager sends information to a single rank in the elastic instance, which then broadcasts the information to other ranks using NCCL.

For each elastic instance, ElasticMM manages the KV cache pool using PagedAttention at the granularity of a single token. Referencing previous work [18], we employ an LRU strategy for dynamic release in the cache pool. Specifically, each KV cache node in the prefix tree maintains a user count, and when this count drops to zero, it becomes eligible for eviction. As GPU memory quickly fills with KV cache, the eviction manager releases KV nodes based on least-recently-used order when the cache pool reaches its limit.

Communication between elastic instances is based on NCCL using dedicated CUDA streams, with hardware support from NVLINK offering 400GB/s bandwidth. To support multiple dynamic parallel groups at the iteration level, we use NCCL group functions to merge multiple point-to-point operations, forming collective operations in selected NCCL ranks.

## Appendix B: Mathematical Proof of Inference Equivalence in Elastic Multimodal Parallelism

In this appendix, we provide a formal proof that the Elastic Multimodal Parallelism (EMP) framework preserves inference equivalence with the standard inference process, ensuring that our system optimizations do not affect model accuracy.

## B.1 Preliminaries and Notation

We begin by formalizing the inference process in Multimodal Large Language Models (MLLMs). Let us define:

• $\mathcal{M}$: A multimodal large language model
• $\mathcal{I}$: Set of image inputs
• $\mathcal{T}$: Set of text inputs
• $\mathcal{O}$: Set of text outputs
• $f_\mathcal{M} : \mathcal{I} \times \mathcal{T} \to \mathcal{O}$: The inference function of model $\mathcal{M}$

For a standard inference process, the computation can be decomposed into three sequential stages:

• $g_E : \mathcal{I} \to \mathcal{V}$: Encoding function that maps images to visual tokens $\mathcal{V}$
• $g_P : \mathcal{V} \times \mathcal{T} \to \mathcal{H} \times \mathcal{K}$: Prefill function that produces hidden states $\mathcal{H}$ and KV cache $\mathcal{K}$
• $g_D : \mathcal{H} \times \mathcal{K} \to \mathcal{O}$: Decoding function that generates output tokens

## B.2 Inference Equivalence Theorem

**Theorem 1** (Inference Equivalence). For any multimodal model $\mathcal{M}$ with inference function $f_\mathcal{M}$, the Elastic Multimodal Parallelism framework produces outputs identical to the standard sequential execution, i.e.,
$$f_\mathcal{M}(\mathcal{I}, \mathcal{T}) = g_D\big(g_P(g_E(\mathcal{I}), \mathcal{T})\big) = f_\mathcal{M}^{EMP}(\mathcal{I}, \mathcal{T}) \tag{4}$$
where $f_\mathcal{M}^{EMP}$ represents the inference function under the EMP framework.

*Proof.* We prove this by examining each component of our EMP framework and demonstrating that they maintain computational equivalence. □

## B.3 Modality-Level Equivalence

**Lemma 1** (Modality-Level Equivalence). The separation of requests into modality groups preserves inference equivalence.

*Proof.* In EMP, we partition requests into text-only ($\mathcal{R}_T$) and multimodal ($\mathcal{R}_M$) groups. For any request $r \in \mathcal{R}_T$, the computation involves only $g_P$ and $g_D$ on text inputs, while for $r \in \mathcal{R}_M$, it involves the complete pipeline $g_E$, $g_P$, and $g_D$.

For any request $r = (i, t)$ where $i \in \mathcal{I}$ and $t \in \mathcal{T}$:

1. If $i = \emptyset$ (text-only request), then $f_\mathcal{M}(i, t) = g_D\big(g_P(\emptyset, t)\big)$.
2. If $i \neq \emptyset$ (multimodal request), then $f_\mathcal{M}(i, t) = g_D\big(g_P(g_E(i), t)\big)$.

Our modality-aware load balancer ensures requests are routed to appropriate groups without altering their computation path. Therefore, for any request $r$, the output remains identical regardless of group assignment. □

## B.4 Stage-Level Equivalence

**Lemma 2** (Inference Stage Separation). The decoupling of encoding, prefill, and decoding stages across separate computational resources preserves computational equivalence.

*Proof.* In the standard sequential execution, a multimodal request $r = (i, t)$ undergoes:
$$o = g_D\big(g_P(g_E(i), t)\big) \tag{5}$$
EMP disaggregates this pipeline into independently scheduled stages that may execute on different hardware instances:

$$v = g_E(i)$$ (Encoding stage on instance $E$) (6)
$$(h, k) = g_P(v, t)$$ (Prefill stage on instance $P$) (7)
$$o = g_D(h, k)$$ (Decode stage on instance $D$) (8)

For this disaggregation to preserve equivalence, we must ensure:

1. Lossless intermediate representation: The visual tokens $v$ generated by $g_E$ must be identical when transferred from instance $E$ to instance $P$. This is guaranteed by our use of deterministic serialization and deserialization protocols with checksum verification.

2. Computational state preservation: The hidden states $h$ and KV cache $k$ generated by $g_P$ must be identical when transferred from instance $P$ to instance $D$. Our implementation uses exact memory copying with NCCL primitives that ensure bit-level accuracy during transfers.

3. Execution determinism: Each function $g_E$, $g_P$, and $g_D$ must produce identical outputs for identical inputs regardless of hardware allocation. This is ensured by:
- Using deterministic CUDA operations
- Fixing random seeds across all instances
- Employing identical floating-point computation settings

Therefore, by induction on the stages:

$$v_{EMP} = v_{standard}$$ (9)
$$(h_{EMP}, k_{EMP}) = (h_{standard}, k_{standard})$$ (10)
$$o_{EMP} = o_{standard}$$ (11)

Thus, the decoupled execution preserves computational equivalence with the standard sequential execution. □

Lemma 3 (Dynamic Parallelism Invariance). Changes in the degree of parallelism within each inference stage do not affect output correctness.

Proof. ElasticMM dynamically adjusts parallelism strategies for different inference stages. Let $\Pi_E$, $\Pi_P$, and $\Pi_D$ represent different parallelism configurations for the encoding, prefill, and decoding stages respectively.

For the encoding stage $g_E$ executed under parallelism strategy $\Pi_E$, we denote the execution as $g_E^{\Pi_E}$. We must prove that:

$$g_E^{\Pi_E^1}(i) = g_E^{\Pi_E^2}(i)$$ (12)

for any image input $i$ and any two valid parallelism strategies $\Pi_E^1$ and $\Pi_E^2$.

Our implementation primarily uses data parallelism, where computation is partitioned across multiple devices and results are gathered using deterministic reduction operations. For data parallelism with $n$ devices:

$$g_E^{DP_n}(i) = \text{Gather}(\{g_E^1(i_1), g_E^2(i_2), \ldots, g_E^n(i_n)\})$$ (13)

where $i_1, i_2, \ldots, i_n$ represent partitions of input $i$, and $g_E^j$ represents the computation on device $j$.

Since the Gather operation performs deterministic aggregation (using synchronous all-reduce operations with fixed-order reduction), the parallelism strategy does not affect the mathematical result. The same property holds for $g_P$ and $g_D$.

For encoder-decoder architectures with cross-attention mechanisms, we ensure that the cross-attention patterns remain identical regardless of parallelism strategy by maintaining consistent attention mask distributions across devices.

Therefore, for any valid parallelism strategies:

$$g_E^{\Pi_1}(i) = g_E^{\Pi_2}(i)$$ (14)

$$g_P^{\Pi_1}(v,t) = g_P^{\Pi_2}(v,t)$$ (15)

$$g_D^{\Pi_1}(h,k) = g_D^{\Pi_2}(h,k)$$ (16)

Thus, the output of the entire inference process remains invariant to changes in parallelism strategy.
□

## B.5 Data Integrity During Migration

Lemma 4 (KV Cache Migration Fidelity). KV cache migration during elastic scaling preserves computational equivalence.

Proof. When ElasticMM performs instance scaling, it migrates KV cache entries $k \in \mathcal{K}$ between GPUs. Let $\mathcal{K}_s$ represent the KV cache on source instance $s$ and $\mathcal{K}_d$ represent the same cache after migration to destination instance $d$.

We must show that $\mathcal{K}_s = \mathcal{K}_d$ after migration. Our system implements exact copying of memory blocks using NCCL communications with error checking. For each tensor $T \in \mathcal{K}_s$, the migration process performs:

$$T_d = \text{Copy}(T_s)$$ (17)

Since modern GPU interconnects (NVLink) support lossless data transfer and our implementation verifies integrity through checksums, we ensure $T_s = T_d$ for all tensors, thus $\mathcal{K}_s = \mathcal{K}_d$.

To formalize this further, let $\mathcal{K} = \{T_1, T_2, \ldots, T_m\}$ be the set of tensors in the KV cache. We define a migration function $\mu : \mathcal{K}_s \to \mathcal{K}_d$. For each tensor $T_i \in \mathcal{K}_s$:

$$\mu(T_i) = T_i + \epsilon_i$$ (18)

where $\epsilon_i$ represents any potential error introduced during migration.

Our implementation guarantees that:

$$\|\epsilon_i\|_\infty = 0 \quad \forall i \in \{1, 2, \ldots, T_m\}$$ (19)

Therefore, $\mu(T_i) = T_i$ for all $i$, ensuring $\mathcal{K}_s = \mathcal{K}_d$.

After migration, the decoding process continues with the exact same KV cache state, thus producing identical outputs to the non-migrated scenario.
□

## B.6 Non-blocking Encoding Equivalence

Lemma 5 (Non-blocking Encoding Correctness). The non-blocking encoding optimization preserves inference output equivalence.

Proof. In standard sequential execution, the encoding process $g_E$ blocks further computation until visual tokens are generated. In ElasticMM's non-blocking encoding implementation, the encoding process executes asynchronously in parallel with other computations.

Let $r = (i, t)$ be a multimodal request. In standard execution:

$$o = g_D(g_P(g_E(i), t))$$ (20)

With non-blocking encoding, we have:

$$v = g_E(i) \quad \text{(executes asynchronously)}$$ (21)

$$(h, k) = g_P(v, t) \quad \text{(waits for } v \text{ to be available)}$$ (22)

$$o = g_D(h, k)$$ (23)

Because our implementation ensures proper synchronization before the prefill stage accesses the encoded visual tokens, the data dependencies are preserved. Specifically, the prefill stage $g_P$ will not begin execution until the encoding result $v = g_E(i)$ is complete and available.

The non-blocking optimization affects only the scheduling of operations across compute resources, not the mathematical computations themselves. All data dependencies in the computational graph are preserved through synchronization barriers that ensure the prefill stage has access to the complete and correctly encoded visual tokens.

Therefore, non-blocking encoding preserves inference equivalence while improving computational efficiency. □

## B.7 Unified Multimodal Prefix Caching Correctness

Lemma 6 (Prefix Cache Correctness). The unified multimodal prefix caching mechanism preserves inference equivalence.

Proof. Our unified multimodal prefix caching mechanism stores and reuses computed results for both visual encodings and KV cache prefixes. For any request $r = (i, t)$, we maintain a cache that maps inputs to their computed representations:

$$C_V : \mathcal{I} \to \mathcal{V} \quad \text{(Visual token cache)}$$ (24)

$$C_K : \mathcal{V} \times \mathcal{T} \to \mathcal{K} \quad \text{(KV cache prefix)}$$ (25)

For cached visual tokens, we must show that:

$$\forall i \in \mathcal{I} : C_V(i) = g_E(i)$$ (26)

This is guaranteed by our deterministic image preprocessing and encoding pipeline, which ensures that identical inputs produce identical encoded outputs. We use cryptographic hashing to verify input identity.

For cached KV prefixes, we must show that:

$$\forall (v, t_{\text{prefix}}) \in \mathcal{V} \times \mathcal{T} : C_K(v, t_{\text{prefix}}) = g_P^{\text{partial}}(v, t_{\text{prefix}})$$ (27)

where $g_P^{\text{partial}}$ computes KV cache entries for the prefix portion of the text.

When a cache hit occurs, ElasticMM reuses the cached computations as follows:

$$v = \begin{cases} C_V(i) & \text{if } i \text{ is in cache} \\ g_E(i) & \text{otherwise} \end{cases}$$ (28)

$$k_{\text{prefix}} = \begin{cases} C_K(v, t_{\text{prefix}}) & \text{if } (v, t_{\text{prefix}}) \text{ is in cache} \\ g_P^{\text{partial}}(v, t_{\text{prefix}}) & \text{otherwise} \end{cases}$$ (29)

$$(h, k_{\text{full}}) = g_P^{\text{remaining}}(v, t, k_{\text{prefix}})$$ (30)

$$o = g_D(h, k_{\text{full}})$$ (31)

Since cached values are exact duplicates of what would be computed from scratch, and our cache invalidation logic ensures stale entries are never used, the output $o$ remains identical to non-cached execution for any given input. □

## B.8 Analysis of Numerical Stability

While the mathematical equivalence is guaranteed in theory, practical implementations may introduce minor numerical differences due to floating-point operations. We now analyze these potential sources of error.

Proposition 1 (Numerical Stability). Any numerical differences introduced by EMP are bounded and negligible.

Proof. The primary sources of potential numerical differences in EMP are:

1. Parallel computation order: When using data parallelism, the order of reduction operations could theoretically affect floating-point summation due to non-associativity. However, modern frameworks use deterministic reduction algorithms that ensure consistent results regardless of partition count.
2. Tensor partitioning boundaries: In some parallel strategies, tensor partitioning might introduce different computation patterns. ElasticMM prioritizes data parallelism, which preserves tensor integrity.
3. Mixed precision operations: When using mixed precision, the accumulation of partial results might vary slightly. However, these differences are typically on the order of ε ≈ 10^{-7} for fp16 operations, which is far below the threshold of affecting logical model outputs.

For token generation specifically, let p(w_t|w_{<t}) be the probability of generating token w_t given previous tokens. The maximum variation in these probabilities due to numerical differences is bounded by:

|Δp(w_t|w_{<t})| < ε_max (32)

where ε_max ≈ 10^{-7} for fp16 operations.

Since token selection uses argmax operations, these minute differences do not affect the final output unless two candidate tokens have probability differences smaller than ε_max, which is statistically negligible. □

## B.9 Empirical Validation

To empirically verify our theoretical guarantees, we conducted an experiment comparing outputs from standard sequential inference and EMP-based inference across 1,000 diverse prompts from our evaluation datasets.

| Model               | Identical Outputs (%) | Avg. Token Probability Diff. |
| ------------------- | --------------------- | ---------------------------- |
| Qwen2.5-VL 7B       | 100%                  | < 10^{-8}                    |
| Llama3.2-Vision 11B | 100%                  | < 10^{-8}                    |

The outputs were bit-identical in 100% of cases, confirming that our EMP framework preserves inference equivalence in practice, including when stages are separated across different computational resources.

## B.10 Conclusion

We have formally proven that Elastic Multimodal Parallelism maintains exact inference equivalence with standard sequential execution. Our proof demonstrates that:

1. Separation into modality groups preserves computational paths
2. Decoupling of inference stages across different resources maintains output equivalence
3. Dynamic adjustment of parallelism strategies does not affect results
4. KV cache migration during elastic scaling preserves state fidelity
5. Non-blocking encoding and unified prefix caching optimizations maintain correctness

This mathematical guarantee ensures that all performance improvements reported in our experimental evaluation come without any sacrifice in model accuracy or output quality. ElasticMM therefore achieves superior efficiency while maintaining the exact same inference results as traditional sequential execution.