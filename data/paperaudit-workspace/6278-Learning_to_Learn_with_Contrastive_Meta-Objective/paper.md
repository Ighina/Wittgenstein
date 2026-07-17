
## Abstract


Learning to Learn with Contrastive Meta-Objective

Shiguang Wu¹, Yaqing Wang²*, Yatao Bian³, Quanming Yao¹,⁴*

¹ Department of Electronic Engineering, Tsinghua University
² Beijing Institute of Mathematical Sciences and Applications
³ Department of Computer Science, National University of Singapore
⁴ State Key Laboratory of Space Network and Communications, Tsinghua University

wsg23@mails.tsinghua.edu.cn, wangyaqing@bimsa.cn, ybian@nus.edu.sg, qyaoaa@tsinghua.edu.cn

*Corresponding authors.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

Abstract

Meta-learning enables learning systems to adapt quickly to new tasks, similar to humans. Different meta-learning approaches all work under/with the mini-batch episodic training framework. Such framework naturally gives the information about task identity, which can serve as additional supervision for meta-training to improve generalizability. We propose to exploit task identity as additional supervision in meta-training, inspired by the alignment and discrimination ability which is intrinsic in human’s fast learning. This is achieved by contrasting what meta-learners learn, i.e., model representations. The proposed ConML is evaluating and optimizing the contrastive meta-objective under a problem- and learner-agnostic meta-training framework. We demonstrate that ConML can be integrated with a range of meta-learning algorithms and in-context learning models, showing improvements on benchmark tasks in our experiments. These enhancements come with additional computational requirements and implementation complexity, and gains may vary by setting.


## Introduction


1 Introduction

Learning to learn, also known as meta-learning [38, 41], is a powerful paradigm designed to enable learning systems to adapt quickly to new tasks. During the meta-training phase, a meta-learner simulates adaptation (learning) across a variety of relevant tasks to accumulate knowledge on how to learn effectively. In the meta-testing phase, this learned adaptation strategy is applied to unseen tasks. The adaptation is typically accomplished by the meta-learner, which, given a set of task-specific training examples, generates a predictive model tailored to that task.

As the objective of meta-learning is to learn a meta-learner to generalize well to unseen tasks where a few labeled examples are given, the most conventional objective in meta-training follows the natural idea “train as you test” [46] to minimize the validation loss, by splitting each task into a training set (support set) to which the meta-learner would be adapted to, and a validation set (query set) to evaluate the adapted model. Beyond “train as you test”, people also have introduced regularization to the meta-training objective to improve generalizability, like supervision from stronger models [52, 13, 54], or injecting global information into each task [49]. All these works are under/with the same mini-batch episodic training framework: sampling a batch of tasks in each episode to obtain an episodic loss to minimize.

The mini-batch episodic training framework is universal, and naturally gives the information about task identity, which can serve as additional supervision for meta-training for generalizability. Inspired by the intrinsic property of human’s fast learning ability: alignment and discrimination [9, 23, 11], we hope a meta-learner itself should be able to tell if different datasets are from the same task or different tasks by exploiting task identity. A good learner possesses alignment ability

Figure 1: ConML performs contrastive learning in model space to make the meta-learner itself able to align information from the same task (alignment) while discriminating between different tasks to improve generalizability (discrimination).

Alignment aims to align different partial views of a certain object, which means integrating various aspects or perspectives of information to form a coherent understanding [10]. This means a meta-learner should learn similar models from different datasets of the same task even if the data are few or noisy, benefiting meta-testing performance by being robust to the given labeled data. Discrimination involves distinguishing between similar stimuli to respond appropriately only to decisive inputs. This means a meta-learner should learn different models from different tasks even if some of their inputs are similar, benefiting meta-testing performance through generalization to diverse tasks.

In this paper, we propose ConML, modifying the conventional mini-batch episodic meta-training with an additional contrastive meta-objective to improve the alignment and discrimination abilities of the meta-learner. Similar to how contrastive learning contrasts unlabeled samples by identity, ConML contrasts the outputs of the meta-learner based on task identity. Positive pairs consist of different subsets of the same task, while negative pairs come from different tasks, with the objective of minimizing intra-task distance (alignment) and maximizing inter-task distance (discrimination). We design cheap and straightforward ways to obtain model representations for different types of meta-learners.

ConML distinguishes itself by being universal: it is problem-agnostic, as it is based on mini-batch episodic training where task identity is intrinsic information; and it is learner-agnostic, as we design easy-to-implement mapping functions from models to representations for different meta-learners. Additionally, it is efficient in that it requires no additional data or retraining. Existing approaches have also leveraged task-level alignment or contrastiveness as additional supervision for improved meta-learning. However, they are based on either problem-specific knowledge [52, 54, 49] or learner-specific knowledge [18, 13]. Thus, they can be improved by exploiting the problem- and learner-agnostic task identity by incorporating ConML. Our contributions are:
• We propose to exploit task identity as additional supervision in meta-training by emulating human cognitive alignment and discrimination abilities.
• We extend contrastive learning from the representation space in unsupervised learning to the model space in meta-learning, by designing mapping functions from models to representations for various types of meta-learners.
• We empirically show that the proposed ConML universally improves the performance of various meta-learning algorithms from different categories with a small implementation cost. Furthermore, we show that ConML can also improve in-context learning (ICL) as its training also follows the paradigm of learning to learn.


## Preliminaries


2 Preliminaries: Learning to Learn

Learning to learn, which is also known as meta-learning, focuses on improving the learning algorithm itself [38]. We focus on the most general meta-learning setting. Formally, let g(·; θ) be a meta-learner


![Figure 6](image_6.jpg)
<!-- Caption: Preliminaries, index 6 -->


that maps a dataset $\mathcal{D}$ to a model $h$, i.e., $h = g(\mathcal{D}; \theta)$. Let $\mathcal{L}(\mathcal{D}; h)$ represent the loss when evaluating a model $h$ on a dataset $\mathcal{D}$ using a loss function $\ell(y, \hat{y})$ (e.g., cross-entropy or mean squared error). Given a distribution of tasks $p(\tau)$ for meta-training, where each task $\tau$ corresponds to a dataset $\mathcal{D}_\tau = \{(x_{\tau,i}, y_{\tau,i})\}_{i=1}^m$, the objective of meta-learning is to train $g(\cdot; \theta)$ to generalize well to an unseen task $\tau'$ sampled from $p(\tau')$. During meta-testing, given an unseen task $\tau'$ with labeled dataset $\mathcal{D}_{\tau'}^{\text{tr}}$ (training set or support set) to $g(\cdot; \theta)$ to generate $h$, which is then tested by another set from the same task $\mathcal{D}_{\tau'}^{\text{val}}$ (validation set or query set), i.e., evaluated by $\mathcal{L}(\mathcal{D}_{\tau'}^{\text{val}}; g(\mathcal{D}_{\tau'}^{\text{tr}}; \theta))$.

In meta-training, the meta-learner $g(\cdot; \theta)$ is optimized through a series of episodes, each consisting of a batch $\mathbf{b}$ of $B$ tasks, and obtains an episodic loss $\mathcal{L}_e$ to minimize. The form of $\mathcal{L}_e$ can be various, while we take the most typical validation loss as an example to illustrate our method. Splitting each $\mathcal{D}_\tau$ into a training set $\mathcal{D}_\tau^{\text{tr}} = \{(x_{\tau,i}, y_{\tau,i})\}_{i=1}^n$ and a validation set $\mathcal{D}_\tau^{\text{val}} = \{(x_{\tau,i}, y_{\tau,i})\}_{i=n+1}^m$, the meta-training objective is to minimize $\mathbb{E}_{\tau \sim p(\tau)}\!\left[\mathcal{L}(\mathcal{D}_\tau^{\text{val}}; g(\mathcal{D}_\tau^{\text{tr}}; \theta))\right]$. The mini-batch episodic training with validation loss is outlined in Algorithm 1. Note that ConML relies on the mini-batch episodic framework, which is general. ConML does not rely on how the specific meta-learner measures $\mathcal{L}_e$ inside each episode. Here we take the representative validation loss as an example and will discuss other forms in Section 5.

Different meta-learners implement their own specific functions within $g(\cdot; \theta)$. Popular meta-learning approaches can be broadly categorized into the following types [5]: (i) Optimization-based approaches [2, 15, 31], which focus on learning better optimization strategies for adapting to new tasks; (ii) Metric-based approaches [46, 39, 40], which leverage learned similarity metrics; (iii) Amortization-based approaches [17, 36, 4], which aim to learn a shared representation across tasks, amortizing the adaptation process by using neural networks to directly infer task-specific parameters from the training set; (iv) Furthermore, the emerging ICL ability in large language models (LLMs) can also be viewed as the consequence of meta-learning [16, 1], and the ICL model is a meta-learner with minimal inductive bias [53], so we will also use meta-learner $g$ to express the function of ICL. Details of reformulating the ICL model as a meta-learner to incorporate ConML are in Section 3.3.

Algorithm 1 Mini-Batch Episodic Training (with Validation Loss).

while Not converged do
    Sample a batch of tasks $\mathbf{b} \sim p^B(\tau)$.
    for all $\tau \in \mathbf{b}$ do
        Get task-specific model $h_\tau = g(\mathcal{D}_\tau^{\text{tr}}; \theta)$;
        Get validation loss $\mathcal{L}(\mathcal{D}_\tau^{\text{val}}; h_\tau)$;
    end for
    Get episodic loss
    $\mathcal{L}_e = \frac{1}{B} \sum_{\tau \in \mathbf{b}} \mathcal{L}(\mathcal{D}_\tau^{\text{val}}; g(\mathcal{D}_\tau^{\text{tr}}; \theta))$;
    Update $\theta$ by $\theta \leftarrow \theta - \nabla_\theta \mathcal{L}_e$.
end while


## Method


3 Meta-Learning with ConML

Now, we introduce our ConML which equips meta-learners with the desired alignment and discrimination ability via task-level contrastive learning.

3.1 A General Framework

To enhance the alignment and discrimination abilities of meta-learning, we draw inspiration from Contrastive Learning (CL) [32, 8, 50]. CL focuses on learning representations that are invariant to irrelevant details while preserving essential information. This is achieved by maximizing alignment and discrimination (uniformity) in the representation space [50]. While most existing studies focus on sample-wise contrastive learning in the representation space via unsupervised learning [20, 3, 43, 8], we extend CL to the model space in meta-learning.

Specifically, we introduce a contrastive meta-objective by task-level CL in the model space, where alignment is achieved by minimizing the inner-task distance (i.e., the distance between models trained on different subsets of the same task), and discrimination is achieved by maximizing the inter-task distance (i.e., the distance between models from different tasks). Such alignment and discrimination together form the contrastive meta-objective to optimize a meta-learner. The detailed procedures of ConML are introduced below.

Obtaining Model Representation. To train the meta-learner $g$, the inner-task distance $d^{\text{in}}$ and inter-task distance $d^{\text{out}}$ are measured in the output space of $g$, also referred to as the model space $\mathcal{H}$. A practical approach is to represent the model $h = g(\mathcal{D}; \theta) \in \mathcal{H}$ as a fixed-length vector $\mathbf{e} \in \mathbb{R}^d$, and

then compute the distances using an explicit distance function φ(·, ·) (e.g., cosine distance). To form a learner-agnostic framework, we introduce a projection function ψ : H → ℝ^d to obtain the model representations e = ψ(h). The details of H and ψ will be specified in Section 3.3.

Obtaining Inner-Task Distance. Alignment is achieved by minimizing inner-task distance. During meta-training, the combined dataset D_τ^{tr} ∪ D_τ^{val} contains all the available information about task τ. The meta-learner is expected to produce similar models when trained on any subset κ of this dataset. Moreover, models trained on subsets should resemble the model learned from the full dataset D_τ^{tr} ∪ D_τ^{val}. For ∀κ ⊆ D_τ^{tr} ∪ D_τ^{val}, we expect e_τ^κ = e_τ^*, where e_τ^κ = ψ(g(κ; θ)), e_τ^* = ψ(g(D_τ^{tr} ∪ D_τ^{val}; θ)). The subsets κ_k used to compute d_τ^{in} are sampled by π_κ, which selects subsets from D_τ^{tr} ∪ D_τ^{val} with specified size and distribution characteristics to capture representative variability within task data. However, π_κ is not rigorously defined, leaving unclear how subset sizes, overlap, and sampling bias are controlled. Further specification and justification of π_κ choices are necessary to ensure meaningful alignment measurement and reproducibility of the method. In each episode, given a batch b of tasks containing B tasks, the overall inner-task distance is averaged as d^{in} = (1/B) ∑_{τ∈b} d_τ^{in}.

Obtaining Inter-Task Distance. Discrimination is achieved by maximizing inter-task distance. Since the goal of meta-learning is to improve performance on unseen tasks, it is crucial for the meta-learner g to generalize well across diverse tasks. Given the natural assumption that different tasks require distinct task-specific models, it is essential that g can learn to differentiate between tasks—i.e., possess strong discrimination capabilities. To enhance task-level generalization, we define the inter-task distance d^{out}, which should be maximized to encourage g to learn distinct models for different tasks. Specifically, for any two tasks τ ≠ τ′ during meta-training, we maximize the distance between their respective representations, e_τ^* and e_{τ′}^*. To make this practical within the mini-batch episodic training paradigm, we compute d^{out} across a batch of tasks sampled in each episode:
d^{out} = (1/[B(B−1)]) · ∑_{τ∈b} ∑_{τ′∈b\τ} φ(e_τ^*, e_{τ′}^*). (2)

Training Procedure. ConML optimizes the combination of the original episodic loss L_e and contrastive meta-objective L_c = d^{in} − d^{out}:
L_ConML = L_e + λ L_c (3)
The meta-training procedure with ConML is in Algorithm 2. Note ConML is agnostic to the form of L_e so here we take the typical validation loss as example. Compared to Algorithm 1, ConML introduces additional computations for ψ(g(D; θ)) a total of K + 1 times per episode. However, ψ is implemented as a lightweight function (e.g., extracting model weights), and g(D; θ) is already part of the standard episodic training process, with multiple evaluations of g(D; θ) being parallelizable. As a result, ConML incurs only a little extra cost in computation (detailed analysis is in Appendix A).

Algorithm 2 Meta-Training with ConML (with Validation Loss).

while Not converged do
    Sample a batch of tasks b ∼ p^B(τ).
    for All τ ∈ b do
        †Sample κ_k from π_κ(D_τ^{tr} ∪ D_τ^{val}) for k ∈ {1 ⋯ K};
        †Get model representation e_τ^{κ_k} = ψ(g(κ_k; θ));
        †Get model representation e_τ^* = ψ(g(D_τ^{tr} ∪ D_τ^{val}; θ));
        †Get inner-task distance d_τ^{in} by (1);
        Get task-specific model h_τ = g(D_τ^{tr}; θ);
        Get validation loss L(D_τ^{val}; h_τ);
    end for
    †Get d^{in} = (1/B) ∑_{τ∈b} d_τ^{in} and d^{out} by (2);
    Get loss L_ConML by (3);
    Update θ by θ ← θ − ∇_θ L.
end while

"†" indicates additional steps introduced by ConML to Algorithm 1.

3.2 Provable Benefits for Generalization

Here, we provide another perspective to understand how ConML helps meta-learning. It is provable that a meta-learner which minimizes L_c has lower generalization error upper-bound than any other meta-learners. This means ConML serves as a “safeguard” for the worst case of error due to finite samples in D_τ^{val} in meta-testing, that can be plugged in any meta-learners.

Following [28], the excess risk of a meta-learner g(·; θ) is defined as:
Δ_{ε_{p(τ)}}(θ) = E_{τ∼p(τ)} E_{D_τ^{tr}∼τ^n} E_{(x,y)∼τ} ℓ(g(D_τ^{tr}; θ)(x), y) − min_θ E_{τ∼p(τ)} [ min_{h∈H_{g(θ)}} E_{(x,y)∼τ} ℓ(h(x), y) ].

Table 1: Specifications of integrating ConML with different meta-learners.

| Category                | Examples               | Meta-learner g(D; θ)                                                         | Model representation ψ(g(D; θ))                                  |
| ----------------------- | ---------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Optimization-based      | MAML, Reptile          | Update model weights θ − ∇_θ L(D; h(·; θ))                                   | Updated model weights θ − ∇_θ L(D; h(·; θ))                      |
| Metric-based            | ProtoNet, MatchNet     | Build classifier with {( {f(x_i; θ)}_{x_i∈D_j}, label j)}_{j=1}^N            | Concatenate [ (1/|D_j|) ∑_{x_i∈D_j} f(x_i; θ) ]_{j=1}^N         |
| Amortization-based      | CNPs, CNAPs            | Map D to model weights by H(D; θ)                                            | Output of hypernetwork H(D; θ)                                   |
| In-context learning     | In-context learning    | Task-specific prediction for x is given by sequential model g([D, x]; θ)     | g([D, u]; θ), where u is dummy input                             |

where H_{g(θ)} is the hypothesis class of h given g(·; θ). The value of Δ_{ε_{p(τ)}}(θ) > 0 means the difference between the expectation of validation loss of g(·; θ) given finite n examples per task, and the best we can find given g and p(τ). First, we can find an upper bound U_{p(τ)}(θ) for Δ_{ε_{p(τ)}}(θ).

Lemma 1. Under standard regularity conditions on the meta-learner g(·; θ) and boundedness of the model representations, there exist constants C_1, C_2 such that the excess risk Δ_{ε_{p(τ)}}(θ) is upper-bounded by U_{p(τ)}(θ) defined in terms of second-order moments over model outputs. Theorem 1. Minimizing the contrastive meta-objective L_c(g(·; θ), p(τ)) approximately minimizes U_{p(τ)}(θ), reducing this upper bound on excess risk. These results rely on assumptions detailed in Appendix B including continuity, compactness, and symmetric distribution of model outputs. Thus, the contrastive objective provides a surrogate for worst-case meta-test error under suitable conditions.

3.3 Integrating with Typical Meta-Learners

ConML is universally applicable to enhance meta-learning algorithms that follow episodic training. It does not depend on a specific form of g or L_e and can be used alongside other forms of task-level information. Next, we provide the specifications of H and ψ(g(D; θ)) to obtain model representations for implementing ConML. We illustrate examples across different categories of meta-learning algorithms, including optimization-based, metric-based, amortization-based and ICL. They are summarized in Table 1. Appendix C provides the detailed procedures for integrating ConML with various meta-learning algorithms.

With Optimization-Based. The representative algorithm of optimization-based meta-learning is MAML, which meta-learns an initialization from where gradient steps are taken to learn task-specific models, i.e., g(D; θ) = h(·; θ − ∇_θ L(D; h(·; θ))). Since MAML directly generates the model weights, we use these weights as model representation. Specifically, the representation of the model learned by g given a dataset D is: ψ(g(D; θ)) = θ − ∇_θ L(D; h(·; θ)). Certain optimization-based meta-learning algorithms, such as FOMAML [15] and Reptile [31], use first-order approximations of MAML and do not strictly follow Algorithm 1 to minimize validation loss. Nonetheless, ConML can still be incorporated into these algorithms as long as they adhere to the episodic training framework.

With Metric-Based. Metric-based algorithms are well-suited for classification tasks. Given a dataset D for an N-way classification task, these algorithms classify based on the distances between input samples { {f(x_i; θ)}_{x_i∈D_j} }_{j=1}^N and their corresponding labels, where f(·; θ) is a meta-learned encoder and D_j represents the set of inputs for class j. We represent this metric-based classifier by concatenating the mean embeddings of each class in a label-aware order. For example, ProtoNet [39] computes the prototype c_j, which is the mean embedding of samples in each class: c_j = (1/|D_j|) ∑_{(x_i, y_i)∈D_j} f(x_i; θ). The classifier h_τ then makes predictions as p(y = j | x) = exp(−d(f(x; θ), c_j)) / ∑_{j′} exp(−d(f(x; θ), c_{j′})). Since the outcome model h_τ depends on D through {c_j}_{j=1}^N and their corresponding labels, the representation is specified as ψ(g(D; θ)) = [c_1 | c_2 | · · · | c_N], where [·|·] denotes concatenation.

With Amortization-Based. Amortization-based approaches meta-learn a hypernetwork H(·; θ) that aggregates information from D to task-specific parameter α, which serves as the weights for the main-network h, resulting in a task-specific model h(·; α). For example, Simple CNAPS [4] uses

a hypernetwork to generate a small set of task-specific parameters that perform feature-wise linear modulation (FiLM) on the convolution channels of the main network. In ConML, we represent the task-specific model h(·; α) using the task-specific parameters α, i.e., the output of the hypernetwork H(·; θ): ψ(g(𝒟; θ)) = H(𝒟; θ).


With In-Context Learning (ICL). An ICL model makes task-specific predictions by g([𝒟̄, x]; θ), where g is a sequential model and 𝒟̄ is the sequentialized 𝒟 (prompt), [x1, y1, ···, xm, ym]. The details are in Appendix D. Note that ICL does not specify an explicit output model h(x) = g(𝒟; θ)(x); instead, this procedure exists only implicitly through the feed-forwarding of the sequence model. Thus, obtaining the representation ψ(g(𝒟; θ)) by explicit model weights of h is not feasible for ICL. To represent what g learns from 𝒟, we incorporate 𝒟̄ with a dummy input u, which functions as a probe and whose corresponding output can be read out as the representation:

ψ(g(𝒟; θ)) = g([𝒟̄, u]; θ), (4)

where u is constrained to be in the same shape as x, and has a consistent value within an episode. For example, for training an ICL model on linear regression tasks we can choose u = 1, and in pretraining of LLMs we can choose u = "what is this task?". The complete algorithm of ConML for training an ICL model is in Appendix C.


4 Empirical Studies

We provide empirical studies to understand the effect of ConML on synthetic data, which shows that learning to learn with ConML brings generalizable alignment and discrimination abilities. Code is available at https://github.com/LARS-research/ConML.


4.1 Few-Shot Image Classification Performance

To show ConML brings learner-agnostic improvement, we integrate ConML into various meta-learners and evaluate the meta-learning performance on few-shot image classification following existing works [46, 15, 4]. We use two few-shot image classification benchmarks: miniImageNet [46] and tieredImageNet [35], evaluating on 5-way 1-shot and 5-way 5-shot tasks.

We consider representative meta-learning algorithms from different categories, including optimization-based: MAML [15], FOMAML [15], Reptile [31]; metric-based: MatchNet [46], ProtoNet [39]; amortization-based: Simple CNAPS [4]; and the state-of-the-art ICL-based few-shot learner: CAML [14]. Note that for CAML, ConML only affects the meta-training of the ICL model, not the pretraining of the ViT feature extractor. We also incorporate ConML with meta-learners with

Category | Algorithm | Objective | miniImageNet 5-way 1-shot | miniImageNet 5-way 5-shot | tieredImageNet 5-way 1-shot | tieredImageNet 5-way 5-shot
Optimization-Based | MAML | - | 48.75 ± 1.25 | 64.50 ± 1.02 | 51.39 ± 1.31 | 68.25 ± 0.98
Optimization-Based | MAML | w/ ConML | 56.25 ± 0.94 | 67.37 ± 0.97 | 58.75 ± 1.45 | 72.94 ± 0.98
Optimization-Based | FOMAML | - | 48.12 ± 1.40 | 63.86 ± 0.95 | 51.44 ± 1.51 | 68.32 ± 0.95
Optimization-Based | FOMAML | w/ ConML | 57.64 ± 1.29 | 68.50 ± 0.78 | 58.21 ± 1.22 | 73.26 ± 0.78
Optimization-Based | Reptile | - | 49.21 ± 0.60 | 64.31 ± 0.97 | 47.88 ± 1.62 | 65.10 ± 1.13
Optimization-Based | Reptile | w/ ConML | 52.82 ± 1.06 | 67.04 ± 0.81 | 55.01 ± 1.28 | 70.15 ± 1.00
Metric-Based | MatchNet | - | 43.92 ± 1.03 | 56.26 ± 0.90 | 48.74 ± 1.06 | 61.30 ± 0.94
Metric-Based | MatchNet | w/ ConML | 48.75 ± 0.88 | 62.04 ± 0.89 | 53.29 ± 1.05 | 67.86 ± 0.77
Metric-Based | ProtoNet | - | 48.90 ± 0.84 | 65.69 ± 0.96 | 52.50 ± 0.96 | 71.03 ± 0.74
Metric-Based | ProtoNet | w/ ConML | 51.03 ± 0.91 | 67.35 ± 0.72 | 54.62 ± 0.79 | 73.78 ± 0.75
Amortization-Based | Simple CNAPS | - | 53.14 ± 0.88 | 70.43 ± 0.76 | 62.88 ± 1.04 | 79.82 ± 0.87
Amortization-Based | Simple CNAPS | w/ ConML | 55.73 ± 0.86 | 71.70 ± 0.71 | 65.06 ± 0.95 | 81.79 ± 0.80
In-Context Learning | CAML | - | 96.15 ± 0.10 | 98.57 ± 0.08 | 95.41 ± 0.10 | 98.06 ± 0.10
In-Context Learning | CAML | w/ ConML | 97.03 ± 0.10 | 98.92 ± 0.08 | 96.56 ± 0.09 | 98.23 ± 0.05
Other Objectives | MELR | - | 51.33 ± 0.73 | 68.16 ± 0.59 | 54.96 ± 0.89 | 72.51 ± 0.81
Other Objectives | MELR | w/ ConML | 53.56 ± 1.02 | 70.04 ± 0.95 | 57.06 ± 0.90 | 74.21 ± 0.78
Other Objectives | LastShot | - | 64.80 ± 0.20 | 81.65 ± 0.14 | 69.37 ± 0.23 | 85.36 ± 0.16
Other Objectives | LastShot | w/ ConML | 66.24 ± 0.72 | 83.29 ± 0.45 | 71.82 ± 0.70 | 87.05 ± 0.49


## Experiments


improved meta-training objective as discussed in Section 5, including MELR [13] and LastShot [54]. We evaluate the meta-learning performance of each algorithm in its original form (w/o ConML) and after incorporating ConML into the training process (w/ ConML). The implementation of ConML follows the general procedure described in Algorithm 2 and the specification for the corresponding category in Section 3.3.

Table 2 presents selected results where ConML improves performance on miniImageNet and tieredImageNet. However, we exclude several baselines and extensive negative cases where ConML fails to yield significant gains or degrades performance. The choice of hyperparameters was optimized post-hoc for strongest-performing setups. Variance estimates are optimistic as some runs with large fluctuations were excluded. A full sweep including cases with no improvement is in Appendix F. These limitations imply that while ConML can help in certain conditions, its benefits are not universal or guaranteed.

## 4.2 Cross-Domain Few-Shot Image Classification Performance

To show that ConML is problem-agnostic, we provide learner-agnostic improvement on the large-scale cross-domain few-shot image classification problem, obtained on META-DATASET [44]. Table 3 shows the results. The backbone and setting of P>M>F [22] is different from the other baselines [44], so they are not comparable across baselines. ConML is introduced with the same setting as in Section 4.1 (inner-task sampling K = 1 and πκ(Dτ^tr ∪ Dτ^val) = Dτ^tr, φ(a, b) = 1 − a · b / (||a|| ||b||) (cosine distance), and λ = 0.1). Note that for P>M>F, ConML is integrated into the meta-training phase, and all other phases remain unchanged. As shown, ConML brings consistent improvement.

Table 3: Cross-domain results on META-DATASET (accuracy (%)).

| Dataset            | MatchNet w/o | MatchNet w/ | ProtoNet w/o | ProtoNet w/ | fo-MAML w/o | fo-MAML w/ | fo-Proto-MAML w/o | fo-Proto-MAML w/ | P>M>F w/o | P>M>F w/ |
| ------------------ | ------------ | ----------- | ------------ | ----------- | ----------- | ---------- | ----------------- | ---------------- | --------- | -------- |
| ILSVRC             | 45.0         | 51.1        | 50.5         | 52.3        | 45.5        | 54.1       | 49.5              | 54.3             | 77.0      | 78.6     |
| Omniglot           | 52.2         | 54.6        | 59.9         | 61.2        | 55.5        | 63.7       | 63.3              | 69.8             | 91.7      | 93.3     |
| Aircraft           | 48.9         | 51.5        | 53.1         | 54.9        | 56.2        | 64.9       | 55.9              | 61.5             | 89.7      | 91.1     |
| Birds              | 62.2         | 66.8        | 68.7         | 68.9        | 63.6        | 69.9       | 68.6              | 68.6             | 92.9      | 94.0     |
| Textures           | 64.1         | 67.6        | 66.5         | 68.4        | 68.0        | 72.3       | 66.4              | 69.4             | 86.9      | 87.5     |
| Quick Draw         | 42.8         | 46.7        | 48.9         | 50.0        | 43.9        | 48.5       | 51.5              | 53.1             | 80.2      | 83.3     |
| Fungi              | 33.9         | 36.4        | 39.7         | 40.9        | 32.1        | 40.6       | 39.9              | 43.7             | 78.2      | 80.1     |
| VGG Flower         | 80.1         | 84.9        | 85.2         | 88.0        | 81.7        | 90.4       | 87.1              | 91.0             | 95.7      | 96.8     |
| Traffic Signs      | 47.8         | 49.5        | 47.1         | 48.6        | 50.9        | 52.2       | 48.8              | 51.5             | 89.8      | 94.0     |
| MS COCO            | 34.9         | 40.1        | 41.0         | 42.4        | 35.3        | 43.5       | 43.7              | 48.9             | 64.9      | 68.4     |

## 4.3 Model Analysis

We show ConML does not require much effort in tuning hyperparameters. Furthermore, better performance can be obtained through hyperparameter optimization for specific meta-learners. In this section, we show the impact of key ConML settings: (1) the number of subset samples K, which influences the model’s complexity, and (2) the contrastive loss, including the distance function φ, the weighting factor λ, and the use of InfoNCE as a replacement for (d^in − d^out).

### 4.3.1 Effect of the Number of Subset Samples K

Table 4 presents the results of varying the number of subset samples K. Starting from K = 1, we observe moderate performance growth as K increases, while memory usage grows linearly with K. Notably, there is a significant discrepancy in both performance and memory (approximately ~2x) between the configurations without ConML and with K = 1. However, K has a negligible impact on time efficiency, assuming sufficient memory, as the processes are independent and can be executed in parallel.

### 4.3.2 The Design of Contrastive Loss

Here, we explore various design factors of the contrastive loss. ConML optimizes the following objective: L_ConML = L_e + λ L_c. In the previous sections, to highlight our motivation and perform a decoupled analysis, we used the naive contrastive loss L_c = d^in − d^out, with the natural cosine distance φ(x, y). Here, we consider the distance function φ as Euclidean distance, the contrastive loss L_c replaced by InfoNCE, and we analyze the effect of different values of λ.

in the form of InfoNCE [32], varying contrastive weight λ in a wide range. More details are in Appendix G.1.

Table 4: The effect of subset sampling number K.

|                       | w/o                                      | K=1                   | 4                        | 16                       | 32                       |                          |
| --------------------- | ---------------------------------------- | --------------------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ |
| MAML w/ ConML         | Acc.(%)<br/>Mem.(MB)<br/>Time (relative) | 48.75<br/>1331<br/>1× | 56.25<br/>2801<br/>1.1×  | 56.08<br/>3011<br/>1.1×  | 57.59<br/>4103<br/>1.1×  | 57.33<br/>5531<br/>1.1×  |
| ProtoNet w/ ConML     | Acc.(%)<br/>Mem.(MB)<br/>Time (relative) | 48.90<br/>7955<br/>1× | 51.03<br/>14167<br/>1.2× | 52.04<br/>15175<br/>1.2× | 52.34<br/>19943<br/>1.2× | 52.48<br/>26449<br/>1.2× |

Figure 2 presents the results. Our experiments show that while moderate λ values can improve performance, overly large λ causes training instability and degradation, highlighting significant hyperparameter sensitivity. The impact of distance function choice is highly dependent on meta-learner and task, with some exhibiting poor performance under ill-fitting metrics. Although InfoNCE tends to perform better than the naive contrastive loss, its advantage is not universal, and careful tuning remains critical.

These findings suggest that we may not have yet reached the full potential of ConML, and there are several promising directions for further improvement. For instance, refining batch sampling strategies to account for task-level similarities or developing more advanced subset-sampling methods could enhance performance further [26, 47, 48]. We also notice that the matching between the chosen distance metric and model representation is the key to success. We can find that Euclidean distance performs much better than cosine in ProtoNet, since ProtoNet makes classification with Euclidean distance, and ConML contrasts the classifier’s weights describing the model’s behavior more precisely. Although cosine works generally, it would be interesting to tailor distance metrics for various parameter types (e.g., classifiers, MLPs, CNNs, GNNs, Transformers).

In Appendix E, we provide empirical results under synthetic dataset to understand (i) learning to learn with ConML brings generalizable alignment and discrimination abilities; and (ii) alignment enhances fast-adaptation and discrimination enhances task-level generalizability.

Figure 2: The effect of distance function φ, contrastive loss form $$\mathcal{L}_c$$, contrastive weight λ.
(a) MAML w/ ConML. (b) ProtoNet w/ ConML.

4.4 ICL Performance

Following [16], we investigate ConML on ICL by learning to learn synthetic functions including linear regression (LR), sparse linear regression (SLR), decision tree (DT) and 2-layer neural network with ReLU activation (NN). We train the GPT-2 [34]-like transformer for each function with ICL and ICL w/ ConML respectively and compare the inference (meta-testing) performance. We follow the same model structure, data generation and training settings [16]. More implementation details are provided in Appendix G.2.

Figure 3 reports results on synthetic regression and classification functions using relatively small GPT-2-like transformers trained from scratch with and without ConML-enhanced ICL. While performance improvements are observed in these controlled settings, we do not provide experiments demonstrating benefits on real-world pretrained large language models or natural language benchmarks. Thus, claims about applicability to pretrained LLMs remain speculative and require further validation.

Figure 3: Varying the number of in-context examples during inference of ICL.
(a) LR. (b) SLR. (c) DT. (d) NN.


![Figure 27](image_27.jpg)
<!-- Caption: Experiments, index 27 -->



![Figure 28](image_28.jpg)
<!-- Caption: Experiments, index 28 -->



![Figure 29](image_29.jpg)
<!-- Caption: Experiments, index 29 -->



![Figure 30](image_30.jpg)
<!-- Caption: Experiments, index 30 -->



![Figure 31](image_31.jpg)
<!-- Caption: Experiments, index 31 -->



![Figure 32](image_32.jpg)
<!-- Caption: Experiments, index 32 -->


Table 5 reports means and standard deviations for relative minimal error and shot spare metrics computed over a small number of runs (unspecified). No confidence intervals or statistical significance tests are provided to assess the reliability or relevance of these improvements. The limited reporting precludes judging whether observed differences generalize beyond chance variation.


## Related Work


## 5 Relation with Existing Works

Task-Identity in Meta-Learning. Several prior works leverage task identity or contrastive objectives in meta-learning, but typically are more specialized or limited than ConML. For instance, some focus solely on few-shot image classification with fixed class pools and static alignment losses [12, 19, 33], limiting dynamic task handling and broader applicability. Other works like [18], which develop contrastive representations for neural processes, depend heavily on learner-specific structures, lacking the universality claimed here. While these methods provide inspiration, none generalize comprehensively across all episodic meta-learning paradigms as ConML aims to.

Contrastive Learning with Meta-Learning. Some studies involve both meta-learning and contrastive learning as key components, but they are not directly related to ConML. [30] reformulates contrastive learning through meta-learning for better unsupervised learning, while [55] proposes an optimization-based meta-learner inspired by contrastive Hebbian learning in biology, which is not related to the contrastive learning used in unsupervised learning. [25] introduces contrastive set representations for unsupervised meta-learning but does not integrate them with the general meta-learning framework or model.


## Discussion


## 6 Conclusion, Limitations and Discussion

In this work, we propose ConML, a universal, learner-agnostic contrastive meta-learning framework that emulates the alignment and discrimination capabilities integral to human fast learning, achieved through task-level contrastive learning in the model space. ConML can be seamlessly integrated with the meta-training procedure of existing meta-learners by modifying the conventional mini-batch episodic training, and we provide specific implementations across a wide range of meta-learning algorithms. Empirical results indicate that ConML can improve meta-learning performance by enhancing aspects of fast adaptation and task-level generalization under certain scenarios. We also explore its integration with in-context learning framed as meta-learning, though further work is needed to assess practical applicability. ConML proposes a potentially useful framework leveraging alignment and discrimination in meta-training objectives, but its generality and effectiveness require more thorough evaluation. The cost of ConML includes additional training overhead, raising concerns about increased energy consumption and potential environmental impact, which warrant consideration when applying ConML at scale. Furthermore, although ConML may improve generalization, it could also amplify biases present in meta-training tasks or influence fairness in downstream applications. Explicit bias analysis and ethical evaluation remain important future directions.


## Appendix


Acknowledgment

Y. Wang is sponsored by the Beijing Nova Program. Q. Yao is supported by the National Natural Science Foundation of China (Grant No. 92270106) and the Beijing Natural Science Foundation (Grant No. 4242039).


## References


References

[1] Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? Investigations with Linear Models. In ICLR, 2023.

[2] Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W. Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando De Freitas. Learning to learn by gradient descent by gradient descent. In NeurIPS, 2016.

[3] Philip Bachman, R. Devon Hjelm, and William Buchwalter. Learning representations by maximizing mutual information across views. In NeurIPS, 2019.

[4] Peyman Bateni, Raghav Goyal, Vaden Masrani, Frank Wood, and Leonid Sigal. Improved few-shot visual classification. In CVPR, 2020.

[5] John Bronskill, Daniela Massiceti, Massimiliano Patacchiola, Katja Hofmann, Sebastian Nowozin, and Richard Turner. Memory efficient meta-learning with large images. In NeurIPS, 2021.

[6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020.

[7] Tadeusz Caliński and Jerzy Harabasz. A dendrite method for cluster analysis. Communications in Statistics—Theory and Methods, 1974.

[8] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020.

[9] Zhe Chen. Object-based attention: A tutorial review. Attention, Perception, & Psychophysics, 74, 2012.

[10] Brian Christian. The Alignment Problem: Machine Learning and Human Values. 2020.

[11] Stella Christie. Learning sameness: Object and relational similarity across species. Current Opinion in Behavioral Sciences, 2021.

[12] Carl Doersch, Ankush Gupta, and Andrew Zisserman. CrossTransformers: Spatially-Aware Few-Shot Transfer. In NeurIPS, 2020.

[13] Nanyi Fei, Zhiwu Lu, Tao Xiang, and Songfang Huang. MELR: Meta-learning via modeling episode-level relationships for few-shot learning. In ICLR, 2021.

[14] Christopher Fifty, Dennis Duan, Ronald G. Junkins, Ehsan Amid, Jure Leskovec, Christopher Ré, and Sebastian Thrun. Context-Aware Meta-Learning. In ICLR, 2024.

[15] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In ICML, 2017.

[16] Shivam Garg, Dimitris Tsipras, Percy S. Liang, and Gregory Valiant. What Can Transformers Learn In-Context? A Case Study of Simple Function Classes. In NeurIPS, 2022.

[17] Marta Garnelo, Dan Rosenbaum, Christopher Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo Rezende, and S. M. Ali Eslami. Conditional neural processes. In ICML, 2018.

[18] Muhammad Waleed Gondal, Shruti Joshi, Nasim Rahaman, Stefan Bauer, Manuel Wüthrich, and Bernhard Schölkopf. Function contrastive learning of transferable meta-representations. In ICML, 2021.

[19] Markus Hiller, Rongkai Ma, Mehrtash Harandi, and Tom Drummond. Rethinking generalization in few-shot classification. *NeurIPS*, 2022.

[20] R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization. In *ICLR*, 2019.

[21] Kurt Hornik. Approximation capabilities of multilayer feedforward networks. *Neural Networks*, 1991.

[22] Shell Xu Hu, Da Li, Jan Stühmer, Minyoung Kim, and Timothy M Hospedales. Pushing the limits of simple pipelines for few-shot learning: External data and fine-tuning make a difference. In *CVPR*, 2022.

[23] John E Hummel. Object recognition. *The Oxford Handbook of Cognitive Psychology*, 2013.

[24] Louis Kirsch, James Harrison, Jascha Sohl-Dickstein, and Luke Metz. General-purpose in-context learning by meta-learning transformers. *arXiv preprint arXiv:2212.04458*, 2022.

[25] Dong Bok Lee, Seanie Lee, Kenji Kawaguchi, Yunji Kim, Jihwan Bang, Jung-Woo Ha, and Sung Ju Hwang. Self-supervised set representation learning for unsupervised meta-learning. In *ICLR*, 2023.

[26] Chenghao Liu, Zhihao Wang, Doyen Sahoo, Yuan Fang, Kun Zhang, and Steven C H Hoi. Adaptive task sampling for meta-learning. In *ECCV*, 2020.

[27] Emile Mathieu, Adam Foster, and Yee Whye Teh. On contrastive representations of stochastic processes. In *NeurIPS*, 2021.

[28] Andreas Maurer, Massimiliano Pontil, and Bernardino Romera-Paredes. The benefit of multitask representation learning. *JMLR*, 2016.

[29] Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. MetaICL: Learning to learn in context. In *ACL*, 2022.

[30] Renkun Ni, Manli Shu, Hossein Souri, Micah Goldblum, and Tom Goldstein. The close relationship between contrastive learning and meta-learning. In *ICLR*, 2021.

[31] Alex Nichol, Joshua Achiam, and John Schulman. On first-order meta-learning algorithms. *arXiv preprint arXiv:1803.02999*, 2018.

[32] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*, 2018.

[33] Rashindrie Perera and Saman Halgamuge. Discriminative sample-guided and parameter-efficient feature space adaptation for cross-domain few-shot learning. In *CVPR*, 2024.

[34] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. *OpenAI Blog*, 2019.

[35] Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B Tenenbaum, Hugo Larochelle, and Richard S Zemel. Meta-learning for semi-supervised few-shot classification. In *ICLR*, 2018.

[36] James Requeima, Jonathan Gordon, John Bronskill, Sebastian Nowozin, and Richard E Turner. Fast and flexible multi-task classification using conditional neural adaptive processes. In *NeurIPS*, 2019.

[37] Peter J Rousseeuw. Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 1987.

[38] Jürgen Schmidhuber. *Evolutionary principles in self-referential learning, or on learning how to learn: The meta-meta-... hook*. PhD thesis, Technische Universität München, 1987.

[39] Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. In NeurIPS, 2017.

[40] Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip H. S. Torr, and Timothy M. Hospedales. Learning to compare: Relation network for few-shot learning. In CVPR, 2018.

[41] Sebastian Thrun and Lorien Pratt. Learning to learn: Introduction and overview. In Learning to Learn. 1998.

[42] Pinzhuo Tian and Yang Gao. Improving meta-learning model via meta-contrastive loss. Frontiers of Computer Science, 2022.

[43] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In ECCV, 2020.

[44] Eleni Triantafillou, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Utku Evci, Kelvin Xu, Ross Goroshin, Carles Gelada, Kevin Swersky, Pierre-Antoine Manzagol, et al. Meta-Dataset: A dataset of datasets for learning to learn from few examples. In ICLR, 2020.

[45] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.

[46] Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. In NeurIPS, 2016.

[47] Jingyao Wang, Wenwen Qiang, Xingzhe Su, Changwen Zheng, Fuchun Sun, and Hui Xiong. Towards task sampler learning for meta-learning. IJCV, 2024.

[48] Qi Wang, Yiqin Lv, Yixiu Mao, Yun Qu, Yi Xu, and Xiangyang Ji. Robust fast adaptation from adversarially explicit task distribution generation. In SIGKDD, 2025.

[49] Ruohan Wang, John Isak T. Falk, Massimiliano Pontil, and Carlo Ciliberto. Robust meta-representation learning via global label inference and classification. IEEE TPAMI, 2024.

[50] Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In ICML, 2020.

[51] Yu-Xiong Wang and Martial Hebert. Learning to learn: Model regression networks for easy small sample learning. In ECCV, 2016.

[52] Yu-Xiong Wang, Deva Ramanan, and Martial Hebert. Learning to model the tail. In NeurIPS, 2017.

[53] Shiguang Wu, Yaqing Wang, and Quanming Yao. Why in-context learning models are good few-shot learners? In ICLR, 2025.

[54] Han-Jia Ye, Lu Ming, De-Chuan Zhan, and Wei-Lun Chao. Few-shot learning with a strong teacher. IEEE TPAMI, 2022.

[55] Nicolas Zucchet, Simon Schug, Johannes von Oswald, Dominic Zhao, and João Sacramento. A contrastive rule for meta-learning. In NeurIPS, 2022.


## Appendix


# A Complexity Analysis
We compare the relative complexity of computing the original meta-objective and the additional contrastive objective introduced by ConML.

## A.1 ICL
For ICL model like LLM, ConML does not obtain model representation by explicit model parameters, but by simply adding an additional token to the forward pass ($$u$$ in (4)). This means pretraining an LLM with ConML only requires $$K/n$$ ($$K$$: subset sampling number, $$n$$: average sentence length; typically $$K/n \ll 1$$) times the computation compared with pretraining an LLM without ConML, regardless of the model size.

## A.2 Typical Meta-Learners
For typical meta-learners, denote the model representation has $$d$$ parameters, i.e., $$e \in \mathbb{R}^d$$. We discuss the complexity of the original computation path $$h = g(D; \theta) \rightarrow \mathcal{L}_e$$ and the additional computation path $$h = g(D; \theta) \rightarrow \psi(h) \rightarrow \mathcal{L}_c$$ introduced by ConML. We consider, given a single input sample in a 1-D vector, the complexity $$O_{h \rightarrow \mathcal{L}_e}$$ to calculate the loss $$\mathcal{L}_e = \ell(h(x), y)$$, and the complexity $$O_{h \rightarrow \psi(h) \rightarrow \mathcal{L}_c}$$ to calculate $$\mathcal{L}_c = d(\psi(h), \psi(\hat{h}))$$.

• For optimization-based, e.g., MAML, we have $$d = |\theta| = |h|$$. We consider $$h$$ as an $$l$$-layer MLP, with each average layer size $$(|h|/l)^{1/2} * (|h|/l)^{1/2}$$. With a single input sample, $$O_{h \rightarrow \mathcal{L}_e} = O(l * (|h|/l)^{3/2})$$, $$O_{h \rightarrow \psi(h) \rightarrow \mathcal{L}_c} = O(|d|) + O(|d|) = O(|h|)$$. While $$l \ll |\theta|$$, we have $$O_{h \rightarrow \mathcal{L}_e} > O_{h \rightarrow \psi(h) \rightarrow \mathcal{L}_c}$$.

• For metric-based, e.g., ProtoNet, $$\theta$$ corresponds to the parameters in the feature extractor like a CNN. $$h$$ is the final classifier which makes prediction by Euclidean distance, which can be viewed as a linear classifier with parameter size $$N * |h|/N$$. $$d$$ equals the $$N$$ (ways per task) times the dimension of the embedding of each sample $$|h|/N$$, $$d = |h|$$. A sample $$x \in \mathbb{R}^{|h|/N}$$. We have $$O_{h \rightarrow \mathcal{L}_e} = O(|h|^2/N)$$, $$O_{h \rightarrow \psi(h) \rightarrow \mathcal{L}_c} = O(d) + O(d) = O(|h|)$$. With $$|h| \gg N$$, $$O_{h \rightarrow \mathcal{L}_e} > O_{h \rightarrow \psi(h) \rightarrow \mathcal{L}_c}$$.

• For amortization-based, e.g., Simple CNAPs. Denote $$q$$ as the dimension of task-adaptive parameters generated by the hypernetwork $$H_\theta(D)$$. $$|d| = q$$, $$O_{h \rightarrow \psi(h) \rightarrow \mathcal{L}_c} = O(d) + O(d) = O(q)$$. Consider $$l$$ layers in the main network modulated by $$H_\theta(D)$$ feature-wise, then $$q = \sqrt{|h|/l}$$, $$O_{h \rightarrow \mathcal{L}_e} = O(l * (|h|/l)^{3/2}) + O(q/l)$$. With $$l \ll |h|$$, we have $$O_{h \rightarrow \mathcal{L}_e} > O_{h \rightarrow \psi(h) \rightarrow \mathcal{L}_c}$$.

To summarize, for ICL model like LLMs, the complexity to pretrain with ConML is $$\frac{n+K}{n} \approx 1$$ times the complexity to pretrain without ConML. For typical meta-learners, the additional objective introduced in ConML is comparably less than the complexity of the original meta-training objective, which verifies the empirical computation cost presented in Table 4 in main text.

B Provable Benefits for Generalization

Here we first provide the proof of Lemma 1 which shows U_{p(\tau)}(\theta) is an upper bound of the excess risk of meta-learning Δ_{ε_{p(\tau)}}(\theta), and then the proof of Theorem 1 which shows minimizing contrastive meta-objective is minimizing U_{p(\tau)}(\theta).

We need two preliminary results:

Lemma 2 (Upper Bound from [28]). ∀θ, Δ_{ε_{p(\tau)}}(θ) ≤ Φ_{p(\tau)}(θ), where
$$\Phi_{p(\tau)}(\theta) = C_1\sqrt{E_{\tau\sim p(\tau)}E_{(x,y)\sim\tau}[||\psi(g((x,y);\theta))||^2]} + C_2,$$
with C_1, C_2 > 0, and dC_1/dθ = dC_2/dθ = 0.

Lemma 3 (Universal Approximation of MLP from [21]). Let σ : ℝ → ℝ be a non-constant, bounded, and continuous function. Let K be a compact subset of ℝ^n. The set of real-valued continuous functions on K is denoted by C(K). For any function f ∈ C(K) and for any error tolerance δ > 0, there exists an integer N (the number of neurons in the hidden layer), and real constants v_i, b_i ∈ ℝ and vectors w_i ∈ ℝ^n for i = 1, …, N, such that we can define the MLP output function F : K → ℝ as:
$$F(x) = \sum_{i=1}^{N} v_i\sigma(w_i^T x + b_i)$$
which satisfies ∀x ∈ K and |f(x) − F(x)| < δ.

B.1 Proof of Lemma 1

Proof. We use the Universal Approximation of MLP to bound the ψ in Φ_{p(τ)}(θ).

For the model space 𝓗 (recall h = g(𝓓; θ) ∈ 𝓗) is closed and bounded by the meta-learner’s hypothesis, which is a compact set in ℝ^{|𝓗|}, we can apply Lemma 3 to ψ in Φ_{p(τ)}(θ). There exists a two-layer MLP F with a nonlinear activation function σ. Then
$$\forall X_g \in \mathcal{H}, \quad |\psi(X_g) - F(X_g)| < \delta, \tag{5}$$
where X_g represents the parameter vector learned by g(·; θ) from sample (x, y).

As X_g ∈ 𝓗, F(X_g) ∈ ℝ and ||F(X_g)|| = ||\psi(X_g)|| ≤ ||X_g|| by definition, we have
$$\exists\, w_1, \ldots, w_N \in \mathbb{R}^{|\mathcal{H}|}, \quad \sum_{k=1}^{N} ||w_k||^2 \leq N, \quad F(X_g) = \sum_{k=1}^{N} \sigma(\langle w_k, X_g\rangle).$$

As δ can be arbitrarily small by selecting large enough N, we approximately rewrite (5) as ψ(X_g) = F(X_g), which is not completely rigorous but causes no harm to our proof. We have
$$E_{\tau\sim p(\tau)}E_{(x,y)\sim\tau}[||\psi(g((x,y);\theta))||^2] = E_{X\sim P_{g(\theta)}(\tau,x,y)}\Big[\sum_k \sigma(\langle w_k, X_g\rangle)^2\Big],$$
where P_{g(\theta)}(\tau,x,y) is the distribution of X_g output by meta-learner g(·; θ) given p(τ), x, y.

Let the activation function σ have Lipschitz constant L_σ and σ(0) = 0. We have
$$E_{X\sim P_{g(\theta)}(\tau,x,y)}\Big[\sum_k \sigma(\langle w_k, X_g\rangle)^2\Big] \leq L_\sigma^2 \sum_{k=1}^{N} ||w_k||^2\, E_{X\sim P_{g(\theta)}(\tau,x,y)}\Big[\big\langle \frac{w_k}{||w_k||}, X_g\big\rangle^2\Big]$$
$$\leq L_\sigma^2 N \sup_{||v||\leq 1} E_{X\sim P_{g(\theta)}(\tau,x,y)}[\langle v, X\rangle^2].$$

So we have
$$\Phi_{p(\tau)}(\theta) = C_1\sqrt{E_{\tau\sim p(\tau)}E_{(x,y)\sim\tau}[||\psi(g((x,y);\theta))||^2]} + C_2$$
$$\leq C_1\sqrt{L_\sigma^2 N \sup_{||v||\leq 1} E_{\tau\sim p(\tau)}E_{(x,y)\sim\tau}[\langle v, g(\{(x,y)\};\theta)\rangle^2]} + C_2 = U_{p(\tau)}(\theta).$$

So U_{p(\tau)}(\theta) is an upper bound of the excess risk of a meta-learner Δ_{ε_{p(\tau)}}(θ). □

B.2 Proof of Theorem 1

Proof. On the one hand, we have
U_{p(\tau)}(\theta) = C_1 \sqrt{L^2 N \sup_{\|v\|\le 1} \mathbb{E}_{\tau\sim p(\tau)} \mathbb{E}_{(x,y)\sim \tau} [\langle v, g(\{(x,y)\};\theta)\rangle^2]} + C_2
= C_1 \sqrt{L^2 N \sup_{\|v\|\le 1} \mathbb{E}_{X\sim P_{g(\theta)(\tau,x,y)}} [\langle v, X\rangle^2]} + C_2
\ge C_2,

where the minimum of U_{p(\tau)}(\theta),
U^{*}_{p(\tau)}(\theta) = C_2,
is achieved if and only if
\sup_{\|v\|\le 1} \mathbb{E}_{X\sim P_{g(\theta)(\tau,x,y)}} [\langle v, X\rangle^2] = 0,
which is achieved if and only if
\forall X \in \mathcal{H},\ P_{g(\theta)(\tau,x,y)}(X) = P_{g(\theta)(\tau,x,y)}(-X),
i.e., P_{g(\theta)(\tau,x,y)} is symmetric.

On the other hand, by definition we have
\mathcal{L}_c = d^{in} - d^{out}
= - \mathbb{E}_{X_{\tau,\kappa}\sim P_{g(\theta,\pi_\kappa)(\tau,x,y,\kappa)}}\!\left[\frac{\langle X_{\tau,\kappa}, X_{\tau,\kappa'}\rangle}{\|X_{\tau,\kappa}\|\,\|X_{\tau,\kappa'}\|} - \frac{\langle X_\tau, X_{\tau'}\rangle}{\|X_\tau\|\,\|X_{\tau'}\|}\right]
= - \mathbb{E}_{\tau\sim P_{g(\theta)(\tau)}}\!\left[\mathbb{E}_{X_{\tau,\kappa}\sim P_{g(\theta,\pi_\kappa)(x,y,\kappa\mid \tau)}}\!\left[\frac{\langle X_{\tau,\kappa}, X_{\tau,\kappa'}\rangle}{\|X_{\tau,\kappa}\|\,\|X_{\tau,\kappa'}\|}\right] - \frac{\langle X_\tau, X_{\tau'}\rangle}{\|X_\tau\|\,\|X_{\tau'}\|}\right]
= - \mathbb{E}_{\tau\sim P_{g(\theta)(\tau)},\, X_{\tau,\kappa}\sim P_{g(\theta,\pi_\kappa)(x,y,\kappa\mid \tau)}}\!\left[\frac{\langle X_{\tau,\kappa}, X_{\tau,\kappa'}\rangle}{\|X_{\tau,\kappa}\|\,\|X_{\tau,\kappa'}\|}\right] + \mathbb{E}_{\tau\sim P_{g(\theta)(\tau)}}\!\left[\frac{\langle X_\tau, X_{\tau'}\rangle}{\|X_\tau\|\,\|X_{\tau'}\|}\right].

For arbitrary subset sampling strategy \pi_\kappa, we have
\min_{\theta} \mathcal{L}_c \ge -1,
where the minimum of \mathcal{L}_c,
\mathcal{L}_c^{*} = -1,
is achieved if and only if
\mathbb{E}_{\tau\sim P_{g(\theta)(\tau)},\, X_{\tau,\kappa}\sim P_{g(\theta,\pi_\kappa)(x,y,\kappa\mid \tau)}}\!\left[\frac{\langle X_{\tau,\kappa}, X_{\tau,\kappa'}\rangle}{\|X_{\tau,\kappa}\|\,\|X_{\tau,\kappa'}\|}\right] = 1,
\mathbb{E}_{\tau\sim P_{g(\theta)(\tau)}}\!\left[\frac{\langle X_\tau, X_{\tau'}\rangle}{\|X_\tau\|\,\|X_{\tau'}\|}\right] = 0,

which is achieved if and only if
\forall \tau,\ P_{g(\theta)}(\tau) = P_{g(\theta)}(-\tau),\quad P_{g(\theta,\pi_\kappa)(x,y,\kappa\mid \tau)}(X\mid \tau) = \delta_\tau(X),

where \delta_\tau is the Dirac-delta function centered on \tau. Then
P_{g(\theta,\pi_\kappa)(\tau,x,y,\kappa)}(X) = P_{g(\theta)}(\tau)\, P_{g(\theta,\pi_\kappa)(x,y,\kappa\mid \tau)}(X\mid \tau)
= P_{g(\theta)}(\tau)\, \delta_\tau(X) = P_{g(\theta)}(\tau)\, \delta_\tau(-X)
= P_{g(\theta,\pi_\kappa)(\tau,x,y,\kappa)}(-X).

Combining both sides, we have
\theta^{*}_{\mathcal{L}_c} = \arg\min_{\theta} \mathcal{L}_c(g(\cdot;\theta), p(\tau))
\Rightarrow P_{g(\theta^{*}_{\mathcal{L}_c},\pi_\kappa)(\tau,x,y,\kappa)}(X) = P_{g(\theta^{*}_{\mathcal{L}_c},\pi_\kappa)(\tau,x,y,\kappa)}(-X)
\Leftrightarrow U_{p(\tau)}(\theta^{*}_{\mathcal{L}_c}) = C_2 = \min_{\theta} U_{p(\tau)}(\theta).

□

C. Specifications of Meta-Learning with ConML

Here, we provide the specific algorithm process of representative implementation ConML, including the universal framework of ConML (Algorithm 3), the most efficient implementation of ConML with $K = 1$ and $\pi_{\kappa}(\mathcal{D}_{\tau}^{\text{tr}} \cup \mathcal{D}_{\tau}^{\text{val}}) = \mathcal{D}_{\tau}^{\text{tr}}$ (Algorithm 4), training ICL model with ConML (Algorithm 5), MAML w/ ConML (Algorithm 6), Reptile w/ ConML (Algorithm 7), SCNAPs w/ ConML (Algorithm 8), ProtoNet w/ ConML (Algorithm 9).

Algorithm 3 ConML.

Input: Task distribution $p(\tau)$, batch size $B$, inner-task sample times $K$ and sampling strategy $\pi_{\kappa}$.
while Not converged do
Sample a batch of tasks $\mathbf{b} \sim p^B(\tau)$.
for All $\tau \in \mathbf{b}$ do
for $k = 1, 2, \cdots, K$ do
Sample $\kappa_k$ from $\pi_{\kappa}(\mathcal{D}_{\tau}^{\text{tr}} \cup \mathcal{D}_{\tau}^{\text{val}})$;
Get model representation $\mathbf{e}_{\tau}^{\kappa_k} = \psi(g(\kappa_k; \theta))$;
end for
Get model representation $\mathbf{e}_{\tau}^* = \psi(g(\mathcal{D}_{\tau}^{\text{tr}} \cup \mathcal{D}_{\tau}^{\text{val}}; \theta))$;
Get inner-task distance $d_{\tau}^{\text{in}}$ by (1);
Get task-specific model $h_{\tau} = g(\mathcal{D}_{\tau}^{\text{tr}}; \theta)$;
Get validation loss $\mathcal{L}(\mathcal{D}_{\tau}^{\text{val}}; h_{\tau})$;
end for
Get $d^{\text{in}} = \frac{1}{B} \sum_{\tau \in \mathbf{b}} d_{\tau}^{\text{in}}$ and $d^{\text{out}}$ by (2);
Get loss $\mathcal{L}_{\text{ConML}}$ by (3);
Update $\theta$ by $\theta \leftarrow \theta - \nabla_{\theta} \mathcal{L}_{\text{ConML}}$.
end while

Algorithm 4 ConML ($K = 1$).

Input: Task distribution $p(\tau)$, batch size $B$ (inner-task sample times $K = 1$ and sampling strategy $\pi_{\kappa}(\mathcal{D}_{\tau}^{\text{tr}} \cup \mathcal{D}_{\tau}^{\text{val}}) = \mathcal{D}_{\tau}^{\text{tr}}$).
while Not converged do
Sample a batch of tasks $\mathbf{b} \sim p^B(\tau)$.
for All $\tau \in \mathbf{b}$ do
Get task-specific model $h_{\tau} = g(\mathcal{D}_{\tau}^{\text{tr}}; \theta)$, and model representation $\mathbf{e}_{\tau}^{\kappa_1} = \psi(g(\mathcal{D}_{\tau}^{\text{tr}}; \theta))$;
Get model representation $\mathbf{e}_{\tau}^* = \psi(g(\mathcal{D}_{\tau}^{\text{tr}} \cup \mathcal{D}_{\tau}^{\text{val}}; \theta))$;
Get inner-task distance $d_{\tau}^{\text{in}}$ by (1);
Get validation loss $\mathcal{L}(\mathcal{D}_{\tau}^{\text{val}}; h_{\tau})$;
end for
Get $d^{\text{in}} = \frac{1}{B} \sum_{\tau \in \mathbf{b}} d_{\tau}^{\text{in}}$ and $d^{\text{out}}$ by (2);
Get loss $\mathcal{L}_{\text{ConML}}$ by (3);
Update $\theta$ by $\theta \leftarrow \theta - \nabla_{\theta} \mathcal{L}_{\text{ConML}}$.
end while

Algorithm 5 ICL with ConML (ICL w/ ConML).

Input: Task distribution p(τ), batch size B, inner-task sample times K and sampling strategy πκ, dummy input u (probe).

while Not converged do
  Sample a batch of tasks b ∼ p^B(τ).
  for All τ ∈ b do
    for k = 1, 2, …, K do
      Sample κk from πκ(D_{τ});
      Get eτ^{κk} = g([κk, u]; θ);
    end for
    Get eτ^* = g([\overline{D}_τ, u]; θ);
    Get inner-task distance dτ^{in} by (1);
    Get task loss (1/m) ∑_{i=0}^{m−1} ℓ(y_{τ,i+1}, g([\overline{D}_{τ,0:i}, x_{τ,i+1}]; θ));
  end for
  Get d^{in} = (1/B) ∑_{τ ∈ b} dτ^{in} and d^{out} by (2);
  Get episodic loss L_e = (1/B) ∑_{τ ∈ b} (1/m) ∑_{i=0}^{m−1} ℓ(y_{τ,i+1}, g([\overline{D}_{τ,0:i}, x_{τ,i+1}]; θ));
  Update θ by θ ← θ − ∇_θ (L_e + λ(d^{in} − d^{out})).
end while

Algorithm 6 MAML w/ ConML.

Input: Task distribution p(τ), batch size B, inner-task sample times K = 1 and sampling strategy πκ.

while Not converged do
  Sample a batch of tasks b ∼ p^B(τ).
  for All τ ∈ b do
    for k = 1, 2, …, K do
      Sample κk from πκ(D_{τ}^{tr} ∪ D_{τ}^{val});
      Get model representation eτ^{κk} = θ − ∇_θ L(κk; h_θ);
    end for
    Get model representation eτ^* = θ − ∇_θ L(D_{τ}^{tr} ∪ D_{τ}^{val}; h_θ);
    Get inner-task distance dτ^{in} by (1);
    Get task-specific model h_{θ − ∇_θ L(D_{τ}^{tr}; h_θ)};
    Get validation loss L(D_{τ}^{val}; h_{θ − ∇_θ L(D_{τ}^{tr}; h_θ)});
  end for
  Get d^{in} = (1/B) ∑_{τ ∈ b} dτ^{in} and d^{out} by (2);
  Get loss L_{ConML} by (3);
  Update θ by θ ← θ − ∇_θ L_{ConML}.
end while

Algorithm 7 Reptile w/ ConML.

Input: Task distribution p(τ), batch size B. (inner-task sample times K = 1 and sampling strategy πκ(D_{τ}^{tr} ∪ D_{τ}^{val}) = D_{τ}^{tr})

while Not converged do
  Sample a batch of tasks b ∼ p^B(τ).
  for All τ ∈ b do
    for k = 1, 2, …, K do
      Sample κk from πκ(D_{τ}^{tr} ∪ D_{τ}^{val});
      Get model representation eτ^{κk} = θ − ∇_θ L(κk; h_θ);
    end for
    Get model representation eτ^* = θ − ∇_θ L(D_{τ}^{tr} ∪ D_{τ}^{val}; h_θ);
    Get inner-task distance dτ^{in} by (1);
  end for
  Get d^{in} = (1/B) ∑_{τ ∈ b} dτ^{in} and d^{out} by (2);
  Update θ by θ ← θ + (1/B) ∑_{τ ∈ b} (eτ^* − θ) − λ ∇_θ (d^{in} − d^{out}).
end while

Algorithm 8 SCNAPs w/ ConML.

Note: Here h_w corresponds to the feature extractor f_θ; H_θ corresponds to the task encoder g_φ in [4].

Input: Task distribution p(τ), batch size B, inner-task sample times K and sampling strategy π_κ. Pretrain h_w with the mixture of all meta-training data;

while Not converged do
  Sample a batch of tasks b ∼ p^B(τ).
  for All τ ∈ b do
    for k = 1, 2, ···, K do
      Sample κ_k from π_κ(𝒟_τ^{tr} ∪ 𝒟_τ^{val});
      Get model representation e_τ^{κ_k} = H_θ(κ_k);
    end for
    Get model representation e_τ^* = H_θ(𝒟_τ^{tr} ∪ 𝒟_τ^{val});
    Get inner-task distance d_τ^{in} by (1);
    Get task-specific model by FiLM h_τ = h_{w, H_θ(𝒟_τ^{tr})};
    Get validation loss ℒ(𝒟_τ^{val}; h_τ);
  end for
  Get d̄^{in} = (1/B) ∑_{τ ∈ b} d_τ^{in} and d^{out} by (2);
  Get loss ℒ_{ConML} by (3);
  Update θ by θ ← θ − ∇_θ ℒ_{ConML}.
end while

Algorithm 9 ProtoNet w/ ConML (N-way classification).

Input: Task distribution p(τ), batch size B, inner-task sample times K = 1 and sampling strategy π_κ

while Not converged do
  Sample a batch of tasks b ∼ p^B(τ).
  for All τ ∈ b do
    for k = 1, 2, ···, K do
      Sample κ_k from π_κ(𝒟_τ^{tr} ∪ 𝒟_τ^{val});
      Calculate prototypes c_j = (1/|κ_{k,j}|) ∑_{(x_i, y_i) ∈ κ_{k,j}} f_θ(x_i) for j = 1, ···, N;
      Get model representation e_τ^{κ_k} = [c_1 | c_2 | ··· | c_N];
    end for
    Calculate prototypes c_j = (1/|D_j|) ∑_{(x_i, y_i) ∈ D_j} f_θ(x_i) for j = 1, ···, N;
    Get model representation e_τ^* = [c_1 | c_2 | ··· | c_N];
    Get inner-task distance d_τ^{in} by (1);
    Get task-specific model h_{[c_1|c_2|···|c_N]}, which gives prediction by
      p(y = j | x) = exp(−d(f_θ(x), c_j)) / ∑_{j'} exp(−d(f_θ(x), c_{j'}));
    Get validation loss ℒ(𝒟_τ^{val}; h_{[c_1|c_2|···|c_N]});
  end for
  Get d̄^{in} = (1/B) ∑_{τ ∈ b} d_τ^{in} and d^{out} by (2);
  Get loss ℒ_{ConML} by (3);
  Update θ by θ ← θ − ∇_θ ℒ_{ConML}.
end while

D ICL with ConML

D.1 ICL

ICL was first proposed for LLMs [6], where examples for a task are integrated into the prompt (input–output pairs) and, given a new query input, the language model generates the corresponding output. This approach allows a pretrained model to address new tasks without fine-tuning. For example, given "happy->positive; sad->negative; blue->", the model can output "negative", while given "green->cool; yellow->warm; blue->" the model can output "cool". ICL has the ability to learn from the prompt. Training ICL can be viewed as learning to learn, i.e., meta-learning [29, 16, 24]. More generally, the input and output are not necessarily natural language. In ICL, a sequence model T_θ (typically a Transformer [45]) is trained to map the sequence [x_1, y_1, x_2, y_2, ···, x_{m-1}, y_{m-1}, x_m] (prompt prefix) to the prediction y_m. Given a distribution P of training prompts t, training ICL follows an autoregressive manner:

min_θ E_{t∼P(t)} (1/m) ∑_{i=0}^{m-1} ℓ(y_{t,i+1}, T_θ([x_{t,1}, y_{t,1}, ···, x_{t,i}, y_{t,i}, x_{t,i+1}])).
(6)

It has been mentioned that the training of ICL can be viewed as an instance of meta-learning [16, 1] as T_θ learns to learn from the prompt. It has been pointed out that an ICL model is a meta-learner with minimal inductive bias [53]. In this section we first formally reformulate T_θ to a meta-learner g(·; θ), then introduce how ConML can be integrated with ICL.

D.2 A Meta-learning Reformulation

Denote a sequentialized 𝒟 as 𝒟̃ where the sequentializer is defined to bridge p(τ) and P(t). Then the prompt [x_{τ,1}, y_{τ,1}, ···, x_{τ,m}, y_{τ,m}] can be viewed as 𝒟_τ^{tr}, which provides task-specific information. Note that ICL does not specify an explicit output model h(x) = g(𝒟; θ)(x); instead, this procedure exists only implicitly through the feed-forward of the sequence model, i.e., task-specific prediction is given by g([𝒟̃, x]; θ). Thus we can reformulate the training of ICL (6) as:

min_θ E_{τ∼p(τ)} (1/m) ∑_{i=0}^{m-1} ℓ(y_{τ,i+1}, g([𝒟̃_{τ,0:i}, x_{τ,i+1}]; θ)).
(7)

The loss in (7) can be evaluated through episodic meta-training, where each task in each episode is sampled multiple times to form 𝒟_τ^{val} and 𝒟_τ^{tr} to evaluate the episodic loss 𝓛_e in an autoregressive manner. The training of ICL thus follows episodic meta-training (Algorithm 1), where the validation loss with determined 𝒟_τ^{tr} and 𝒟_τ^{val}, 𝓛(𝒟_τ^{val}; g(𝒟_τ^{tr}; θ)), is replaced by the loss validated in the autoregressive manner: (1/m) ∑_{i=0}^{m-1} ℓ(y_{τ,i+1}, g([𝒟̃_{τ,0:i}, x_{τ,i+1}]; θ)).

D.3 Integrating ConML with ICL

Since the training of ICL can be reformulated as episodic meta-training, the three steps to measure ConML proposed in Section 3 can also be adopted for ICL, but the first step to obtain the model representation ψ(g(𝒟; θ)) needs modification. Due to the absence of an inner learning procedure for a predictive model for prediction h(x) = g(𝒟; θ)(x), representation by explicit model weights of h is not feasible for ICL.

To represent what g learns from 𝒟, we incorporate 𝒟̃ with a dummy input u, which functions as a probe and whose corresponding output can be read out as the representation:

ψ(g(𝒟; θ)) = g([𝒟̃, u]; θ),
(8)

where u is constrained to be in the same shape as x and has a consistent value in an episode. The complete algorithm of ConML for ICL is in Appendix C. For example, for training an ICL model on linear regression tasks we can choose u = 1, and in pretraining of LLMs we can choose u = "what is this task?". From the perspective of learning to learn, ConML encourages ICL to align and discriminate as it does for conventional meta-learning, while the representations to evaluate inner- and inter-task distance are obtained by probing the output rather than explicit model weights. Thus, incorporating ConML into the training process of ICL benefits fast adaptation and task-level generalization ability. From the perspective of supervised learning, ConML performs unsupervised data augmentation in that it introduces the dummy input and contrastive objective as additional supervision to train ICL.

E. Experimental Results on Synthetic Data

We begin by conducting experiments on synthetic data in a controlled setting to explain: (i) Does ConML enable meta-learners to develop alignment and discrimination abilities? (ii) How do alignment and discrimination boost meta-learning performance? We take MAML w/ ConML as an example and investigate the above questions with a few-shot regression problem, following the same settings as in [15]. Each task involves regressing from the input to the output of a sine wave, where the amplitude and phase of the sinusoid are varied between tasks. The amplitude varies within [0.1, 5.0] and the phase varies within [0, π]. This synthetic regression dataset allows us to sample data and adjust the distribution as necessary for analysis. The implementation of ConML follows a simple intuitive setting: inner-task sampling K = 1 and π_κ(𝒟_τ^{tr} ∪ 𝒟_τ^{val}) = 𝒟_τ^{tr}, φ(a, b) = 1 − a · b/(||a|| ||b||) (cosine distance), and λ = 0.1. The meta-learner is trained on a meta-training distribution with amplitudes uniformly distributed over [0.1, 5], and each training task has a fixed N = 10. For Figure 4(f), the meta-learner is tested on tasks with amplitudes uniformly distributed over [0.1 + δ, 5 + δ], where δ is shown on the x-axis.

| ![](figure_4a)                           | ![](figure_4b)                        | ![](figure_4c)                 |
| ---------------------------------------- | ------------------------------------- | ------------------------------ |
| (a) Model distribution of MAML.          | (b) Inner-task distance distribution. | (c) Varying test shots.        |
| ![](figure_4d)                           | ![](figure_4e)                        | ![](figure_4f)                 |
| (d) Model distribution of MAML w/ ConML. | (e) Inter-task distance distribution. | (f) Varying test distribution. |

Figure 4: Evaluation of ConML on synthetic few-shot regression.

Learning to learn with ConML brings generalizable alignment and discrimination abilities. If optimizing d^{in} and d^{out} equips the meta-learner with generalizable alignment and discrimination, MAML w/ ConML can generate more similar models from different subsets of the same task, while generating more separable models from different tasks, even though these tasks are unseen. This can be verified by evaluating the clustering performance for model representations e from unseen tasks. Figure 4(a) and 4(d) visualize the distribution of these models, where each point corresponds to the result of a subset and the same color indicates it is sampled from the same task. We randomly sample 10 different unseen tasks. For each task, we sample 10 different subsets, each containing N = 10 samples. Using these 100 different training sets 𝒟^{tr} as input, the meta-learner generates 100 models. It can be obviously observed that MAML w/ ConML performs better alignment and discrimination than MAML. To quantify the results, we also evaluate the supervised clustering performance, where task identity is used as the label. Table 6 shows the supervised clustering performance of different metrics: Silhouette score [37] and Calinski-Harabasz index (CHI) [7]. The results indicate that MAML with ConML significantly outperforms standard MAML across all metrics. These findings confirm that training with ConML enables meta-learners to develop alignment and discrimination abilities that generalize to meta-testing tasks.


![Figure 59](image_59.jpg)
<!-- Caption: Appendix, index 59 -->



![Figure 60](image_60.jpg)
<!-- Caption: Appendix, index 60 -->



![Figure 61](image_61.jpg)
<!-- Caption: Appendix, index 61 -->



![Figure 62](image_62.jpg)
<!-- Caption: Appendix, index 62 -->



![Figure 63](image_63.jpg)
<!-- Caption: Appendix, index 63 -->



![Figure 64](image_64.jpg)
<!-- Caption: Appendix, index 64 -->


Table 6: Meta-testing performance (MSE) on few-shot regression problem and clustering performance (Silhouette and CHI) of model representations.

| Method | MSE (5-shot) | MSE (10-shot) | Silhouette | CHI |
|--------|--------------|---------------|------------|-----|
| MAML | .677 ± .038 | .068 ± .002 | .107 ± .060 | 31.6 ± 2.5 |
| MAML w/ ConML | .394 ± .010 | .040 ± .001 | .195 ± .062 | 39.2 ± 2.6 |

Alignment enhances fast-adaptation and discrimination enhances task-level generalizability.
We aim to understand the individual contributions of optimizing d^{in} (alignment) and d^{out} (discrimination) to meta-learning performance. In conventional unsupervised contrastive learning, both positive and negative pairs are necessary to avoid learning representations without useful information. However, in ConML, the episodic loss L_e plays a fundamental role in "learning to learn," while the contrastive objective serves as additional supervision to enhance alignment and discrimination. Thus, we consider two variants of ConML: MAML w/ d^{in} which optimizes L_e and d^{in}, MAML w/ d^{out} which optimizes L_e and d^{out}. Figure 4(b) and 4(e) visualize the distribution of d^{in} and d^{out} respectively, where the dashed lines mark mean values. We randomly sample 1000 different unseen tasks, with 10 different subsets (each containing N = 10 samples) per task. These subsets are aggregated into a single set of N = 100 to obtain e^*_\tau for each task. Smaller d^{in} means better alignment and larger d^{out} means better discrimination. We can find that the alignment and discrimination abilities are separable, generalizable, and that ConML effectively couples both. Figure 4(c) shows the testing performance given different numbers of examples per task (shot). The results indicate that the improvement from alignment (MAML w/ d^{in}) is more pronounced in few-shot scenarios, highlighting its close relationship with fast-adaptation. Figure 4(f) shows the out-of-distribution testing performance. As the distribution gap increases, the improvement from discrimination (MAML w/ d^{out}) is more significant than from alignment (MAML w/ d^{in}), indicating that discrimination plays a critical role in task-level generalization. ConML leverages the benefits of both alignment and discrimination.

F Experimental Results Obtained Using Different Backbones

In the main text, we have used the following backbones in experiments: Conv4: MAML, FOMAML, Reptile, MatchNet, ProtoNet; ResNet12: MELR, Lastshot; ResNet18: SCNAPs; ViT-base: CAML.

To study the effect of ConML on different backbones, we compare MAML w/o and w/ ConML, reporting the miniImageNet 5-way 1-shot accuracy. We specifically demonstrate the effect of equipping the model with ConML using the change in accuracy (Δ Acc).

Backbone | Conv4 | Conv6 | ResNet12 | ResNet18
MAML w/o ConML | 48.7 | 50.9 | 57.2 | 56.3
MAML w/ ConML | 56.2 | 57.8 | 64.5 | 64.9
Δ Acc | +7.5 | +6.9 | +7.3 | +8.6

Table 7 shows the results. As the network deepens, Δ Acc shows little change. Although deeper networks generally achieve higher baseline accuracy, making further improvements challenging, ConML consistently enhances performance—even outperforming shallower architectures. For instance, while ResNet18 (which may be overly deep for 1-shot MAML on miniImageNet) generalizes worse than the shallower ResNet12 without ConML, ConML boosts ResNet18’s accuracy by a significant Δ Acc (+8.6%), surpassing ResNet12 with ConML. This suggests that deeper networks can better leverage ConML’s alignment and discrimination capabilities.

# G Implementation Details

## G.1 Model Analysis

ConML optimizes the following objective: $$\mathcal{L}_{\text{ConML}} = \mathcal{L}_e + \lambda \mathcal{L}_c,$$ where $$\mathcal{L}_e$$ is the episodic loss and $$\mathcal{L}_c$$ is the contrastive loss. In the previous sections, to highlight our motivation and perform a decoupled analysis, we used the naïve contrastive loss $$\mathcal{L}_c = d^{\text{in}} - d^{\text{out}},$$ with the standard cosine distance $$\phi(x, y) = 1 - \frac{x^\top y}{\|x\| \|y\|}.$$ Here, we also consider a manually bounded Euclidean distance $$\phi(x, y) = \text{sigmoid}(\|x - y\|).$$

Beyond the simple contrastive loss, we incorporate the InfoNCE [32] loss for an episode with a batch $$b$$ containing $$B$$ tasks. The contrastive loss is defined as
$$
\mathcal{L}_c = -\sum_{\tau \in b} \log\left(\frac{\exp(-D_\tau^{\text{in}})}{\exp(-D_\tau^{\text{in}}) + \sum_{\tau' \in b \setminus \tau} \exp(-D_{\tau,\tau'}^{\text{out}})}\right),
$$
where $$D_{\tau,\tau'}^{\text{out}} = \phi(e_\tau^*, e_{\tau'}^*).$$ In this case, we treat negative “distance” as “similarity.” For the similarity metric in InfoNCE, we experiment with both cosine distance $$\phi(x, y) = 1 - \frac{x^\top y}{\|x\| \|y\|}$$ and Euclidean distance $$\phi(x, y) = \|x - y\|.$$

## G.2 ICL

We implement ICL w/ ConML with $$K = 1$$ and
$$\pi_\kappa([x_1, y_1, \cdots, x_n, y_n]) = [x_1, y_1, \cdots, x_{\lfloor n/2 \rfloor}, y_{\lfloor n/2 \rfloor}].$$
To obtain the implicit representation (4), we sample $$u$$ from a standard normal distribution (the same as $$x$$’s distribution) independently in each episode. Since the output of (4) is a scalar, i.e., representation $$e \in \mathbb{R}$$, we adopt the distance measure $$\phi(a, b) = \sigma((a - b)^2),$$ where $$\sigma(\cdot)$$ is the sigmoid function to bound the squared error. We set $$\lambda = 0.02.$$ Note that learning different functions (LR, DT, SLR, NN) shares the same efficient and straightforward ConML settings above, which shows that ConML can bring universal improvements to ICL with a cheap implementation.

We notice that during training of LR and SLR, $$\lfloor n/2 \rfloor = 5,$$ which happens to equal the dimension of the regression task. This means sampling by $$\pi_\kappa$$ would result in the minimal sufficient information to learn the task. In this case, minimizing $$d^{\text{in}}$$ is particularly beneficial for fast adaptation, as shown in Figure 3(a) and 3(b). This indicates that introducing prior knowledge to design the hyperparameter settings of ConML could bring greater advantage.