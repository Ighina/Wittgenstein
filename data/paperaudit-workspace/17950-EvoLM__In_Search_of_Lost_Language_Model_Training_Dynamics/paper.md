
## Introduction


EvoLM: In Search of Lost Language Model Training Dynamics
Zhenting Qi1 Fan Nie2 Alexandre Alahi3 James Zou2 Himabindu Lakkaraju1 Yilun Du1 Eric Xing4 Sham Kakade1 Hanlin Zhang1
1 Harvard 2 Stanford 3 EPFL 4 CMU


## Abstract


Abstract
Modern language model (LM) training has been divided into multiple stages, making it difficult for downstream developers to evaluate the impact of design choices made at each stage. We present EvoLM, a model suite that enables systematic and transparent analysis of LMs’ training dynamics across pre-training, continued pre-training, supervised fine-tuning, and reinforcement learning. By training over 100 LMs with 1B and 4B parameters from scratch, we rigorously evaluate both upstream (language modeling) and downstream (problem-solving) reasoning capabilities, including considerations of both in-domain and out-of-domain generalization. Key insights highlight diminishing returns from excessive pre-training, challenges and partial mitigation of catastrophic forgetting during domain-specific continued pre-training, complex interactions and limited bridging benefits between pre-training and post-training phases, and nuanced trade-offs in configuring supervised fine-tuning and reinforcement learning. To facilitate open research, we release selected pre-trained and post-trained models along with portions of the training datasets and partial evaluation scripts. However, due to proprietary portions of the dataset and internal infrastructure dependencies, full training pipelines and all data cannot be fully released publicly at this time. We note that the datasets used have not undergone formal ethical review or bias audits, and potential risks related to content fairness and misuse remain unexamined.
Model Suite Datasets Code


## Introduction


1 Introduction
Scaling up language models has been a paradigm that enables various downstream applications [8, 52, 32]. One approach to understanding scaling—and enabling more efficient resource allocation—is through scaling laws, which characterize the quantitative relationship between pre-training log-loss and compute [22, 27, 23, 21]. In part due to the vast design space [34] and the complex interactions of several training phases such as pre-training and post-training [13, 70] for open-weight models [17], it remains challenging to clearly identify which decisions consistently lead to reliable downstream performance gains.

Although progress has been made in understanding how models learn during training [59, 53, 43, 16, 66], accurately forecasting downstream problem-solving performance remains challenging due to the training-inference mismatch in auto-regressive generative models [46] and the non-smooth nature of downstream performance improvements [45]. Existing studies often rely on checkpoints with limited transparency regarding training details, which can introduce potential confounding factors [70]. These include: (1) dependence on opaque analyses from post-training studies that utilize off-the-shelf base models, often without strict control over key variables such as model size, pre-training data size, and data components [42, 10, 61], and (2) evaluations based on intermediate checkpoints [59, 51], which may have sub-optimal downstream performance due to incomplete learning rate decay [48, 24, 54, 67, 30], thereby complicating fair comparisons.

In this work, we establish an end-to-end development pipeline using open toolkits [1, 71, 49] and open data sources [38, 63, 55, 31] to systematically and transparently investigate language models’ reasoning capabilities throughout their lifecycle, covering phases of pre-training, continued pre-training, supervised fine-tuning, and reinforcement learning.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

pretraining, supervised fine-tuning, and reinforcement learning. We introduce EvoLM, a model suite comprising 100+ decoder-only autoregressive LMs with 1B and 4B parameters, each trained from scratch with complete learning rate decay across various configurations of model size and dataset scale. Pre-trained on publicly available corpora FineWeb [38] only, our base models achieve competitive performance on English-only language modeling tasks compared with other open-weight models with significantly more pretraining compute (Table 4). For example, our 1B and 4B models, pre-trained on 320B tokens, perform competitively with TinyLlama-1B and Qwen1.5-4B, respectively, despite their significantly more pretraining data (2T and 3T tokens). We evaluate both upstream language modeling performance (measured by perplexity) and downstream practical problem-solving capabilities (assessed through generative rollout performance) on both in-domain (ID) math reasoning and out-of-domain (OOD) general reasoning tasks. Through extensive controlled and transparent experiments, our study addresses several critical gaps in understanding LM training dynamics, provides novel insights into model behaviors, and identifies open research directions in recent literature. In summary, our contributions include:
• Systematic analyses of language model capabilities across their entire lifecycle—from pre-training to RL post-training—with evaluation on reasoning-intensive upstream cloze tasks and downstream generative tasks, considering both in-domain and out-of-domain generalization.
• Open-sourcing 100+ LMs trained from scratch with 1B and 4B parameters and their training data for all stages, enabling the research community to build upon our findings.
• Open-sourcing a comprehensive, transparent, and reproducible training pipeline and evaluation framework, facilitating further research into scaling laws, training dynamics, and evaluating upstream and downstream capabilities of language models.


## Method


2 Experimental Settings

2.1 Training Setup

We initialize all models using the LLaMA-2 [56] architecture with 1B and 4B parameters. Our training pipeline consists of four sequential stages:
• Pre-training: Conducted exclusively on FineWeb-Edu [38]. Following the Chinchilla scaling law [23], models were pretrained under strictly cumulative token budgets: from 20× model size (compute-optimal) to 320B tokens. Budgets are non-overlapping and strictly sequential across stages.
• Continued Pre-training (CPT): Performed on FineMath [2], exclusively after pre-training, with token budgets ranging from 2B to 42B. No token overlap or mixing occurs; when replay data is used, it strictly supplements FineMath tokens without double counting.
• Supervised Fine-Tuning (SFT): Conducted on aggregated, filtered QA datasets from GSM8K [12], MATH [20], MetaMathQA [63], OpenMathInstruct2 [55], and NuminaMath [31], with explicit disjoint samples from CPT data.
• Reinforcement Learning (RL): Built on PPO [47], using data separate from SFT with non-overlapping examples.

Model naming strictly reflects sequential and exclusive token counts per stage. For instance, 1B-160BT-8+42BT-100Kep1-100Kep16 indicates: a 1B-parameter model pretrained on 160B tokens; continued pretrained on 8B tokens of replay FineWeb-Edu data plus 42B tokens FineMath data; fine-tuned with 100K SFT examples for 1 epoch; then RL tuned on 100K examples for 16 epochs. Token counts are additive but explicitly non-overlapping to avoid confusion.

For all configurations, models are analyzed using both final checkpoints and representative intermediate checkpoints throughout training to better capture training dynamics, stabilizing effects, and trade-offs associated with various stopping points. This approach also aids in quantifying variance across runs.

## 2.2 Evaluation Protocol

Upstream Cloze Tasks: These tasks assess models’ language modeling capabilities via next-token prediction, without requiring conversational abilities. We evaluate pretrained and continued-pretrained models on the following datasets, reporting average zero-shot accuracy across them: HellaSwag [65], Winogrande [44], PIQA [6], OBQA [36], ARC-Easy/Challenge [11].

Downstream Generative Tasks: These tasks evaluate models’ problem-solving abilities in a generative, conversational setting. We test supervised fine-tuned and RL-finetuned models on: 1) In-Domain Tasks (math reasoning): GSM8K-Platinum [57] (a revised version of the full GSM8K [12] test set that minimizes label noise) and MATH [20]. 2) Out-of-Domain Tasks: CRUXEval [19] (code reasoning), BGQA [28] (logical reasoning), TabMWP [35] (table reasoning), and StrategyQA [18] (commonsense reasoning). We evaluate models in a zero-shot manner by prompting them to generate full solutions in response to problems and report average performance for ID and OOD tasks. More evaluation details including dataset descriptions, sampling parameters, and standard errors are reported in Appendix B.3. Evaluation metrics include:

• Accuracy: We measure accuracy under four prompting schemes: 1) Pass@1: Temperature = 0. A single deterministic response is generated. The problem is marked correct if this response is correct. 2) Maj@16: Temperature = 1. Sixteen responses are sampled, and the majority answer is evaluated for correctness. 3) RM@16: Temperature = 1. Sixteen responses are sampled; the one with the highest ORM score is evaluated for correctness. 4) Pass@16: Temperature = 1. Sixteen responses are sampled; the problem is marked solved if any one of the responses is correct. For all these settings, final answers are extracted from model outputs and compared against ground-truth solutions to determine correctness. We additionally report Correct Ratio: In the response groups that have at least one correct solution, we compute the ratio of the number of correct solutions to the total number of solutions (16).

• ORM Score: Computed as the mean scalar output from the Skywork-Reward-Llama-3.1-8B-v0.2 [33] reward model applied to generated solutions, normalized by response length and calibrated using a held-out validation set. Exact computation details and scaling parameters are provided in Appendix B.3 for reproducibility.


## Experiments


## 3 Scaling Studies Across Three Training Stages

### 3.1 Scaling Up Pre-training Compute

To quantify how varying the total amount of pre-training compute affects language modeling performance, we pre-train 0.5B, 1B, and 4B models on token budgets ranging from 10B up to 320B tokens. Figure 1 shows a general upward trend in upstream task performance with pre-training tokens but notable run-to-run variability is not shown. Confidence intervals would reveal considerable fluctuations around 80× to 160× model size, suggesting the apparent saturation may partly reflect noise rather than definitive plateauing. The larger 4B model continues to benefit slightly longer but also plateaus by 320BT.

We further assess how these pre-training budgets translate to downstream capabilities for both SFT and SFT+RL models. Figure 2 shows all six metrics on ID and OOD downstream tasks from 20BT to 320BT pre-training budgets for 1B models. Both SFT and SFT+RL variants exhibit strong initial gains up to 80BT, but performance saturates thereafter: For instance, ID Maj@16 accuracy of the SFT model rises sharply from 8% at 20BT to 15% at 80BT, yet only inches up to 17% at 320BT. RL yields a consistent uplift over pure SFT, but likewise shows negligible benefit from over-training beyond 80BT.

Figure 1: Upstream task performance vs. pre-training tokens on models {0.5B, 1B, 4B}-{10BT, 20BT, 40BT, 80BT, 160BT, 320BT}.


![Figure 11](image_11.jpg)
<!-- Caption: Experiments, index 11 -->


Figure 2: Downstream task performance vs. number of pretraining tokens on models:
- SFT: 1B-{20BT, 40BT, 80BT, 160BT, 320BT}-8+42BT-100Kep1
- SFT+RL: 1B-{20BT, 40BT, 80BT, 160BT, 320BT}-8+42BT-100Kep1-100Kep8.


Table 1 presents a filtered comparison favoring smaller models, with 1B-320BT models reported to outperform 4B-80BT models based solely on token counts, ignoring the inherently higher FLOPs and parameter updates of the 4B models. The compute budget equivalence is thus misleading because training for more tokens on smaller models does not equate to the same compute expended on larger models with fewer tokens. Effective compute normalization accounting for FLOPs and step counts is absent, resulting in incomplete and cherry-picked evidence that exaggerates smaller models’ competitive edge while underreporting larger models’ potential.


Moreover, Maj@16, RM@16, and Pass@16 accuracies on OOD tasks decrease after the 160BT budget, and such degradation is also amplified by a drop in ORM score, showing the overall generation quality decreases to a certain degree. These patterns reveal that excessively large pre-training budgets lead to diminishing returns on downstream performance and might even cause degradation. This finding is consistent with previous work [51], which points out that scaling up pre-training does not always improve or can even hurt LMs’ performance after SFT, and we further complete the studies by showing that 1) such performance gain stagnation is also reflected on downstream generative reasoning tasks and 2) RL finetuning is also constrained by overtraining.


➠ Takeaway 1. Excessive pre-training can lead not only to diminishing returns but also to degradation in downstream generalization, particularly on OOD tasks, underscoring the need for careful early stopping and regularization when scaling compute beyond 80× to 160× model size.


We further look into how model size interplays with scaling up pre-training. As Table 1 illustrates, under a fixed pre-training compute budget (1B–320BT vs. 4B–80BT), the smaller 1B model even outperforms the 4B model across both SFT and SFT+RL settings. When matching on pre-training tokens, we see the same trend at lower budgets: at 80B tokens the 1B–80BT and 4B–80BT models perform comparably, with the smaller model slightly ahead. However, once the budget rises to 160B tokens, the 4B–160BT model “unlocks” its scale: for example, the 4B SFT model jumps to an ID Maj@16 of 26.4% (vs. 14.2% of the 1B counterpart) and the 4B SFT+RL model jumps to 34.8% (vs. 22.5% of the 1B counterpart), demonstrating that only after reaching the saturation regime of pre-training does model size translate into substantial gains in post-training performance.


➠ Takeaway 2. Under limited pre-training budgets, smaller post-trained models can even outperform larger counterparts. Conversely, once pre-training tokens reach the saturation regime, increasing model size enables clear improvements in both in-domain performance and OOD generalization.


![Figure 13](image_13.jpg)
<!-- Caption: Experiments, index 13 -->


## 3.2 Scaling Up Continued Pre-training Compute

We investigate the impact of continued pre-training (CPT) compute by varying the total CPT tokens from 0 (no CPT) to 50 BT, using the 1B-160BT pretrained model as the base. As shown in Figure 3, increasing CPT compute gradually degrades upstream task performance, indicating catastrophic forgetting [15]. To mitigate this issue, we adopt a simple “replay” strategy [25] by randomly interleaving a small amount of pre-training data during CPT. Figure 3 demonstrates that the model with 8 BT replay consistently maintains higher upstream accuracy than the no-replay baseline across all CPT budgets. We then apply SFT to the CPT models on 100K examples for one epoch to investigate the impact of replay on downstream performance. Table 2 reports Pass@1 accuracy on GSM8K-Platinum for each CPT mix. Pure FineMath CPT (50 BT) achieves 19.27%, whereas a mix of 8 BT FineWeb replay with 42 BT FineMath tokens even yields a better result at 21.01%. Configurations with either too little (1.6+48.4 BT) or too much (16+34 BT) replay perform worse, highlighting that a modest replay budget (around 5%) optimally balances retention of general-domain knowledge with adaptation to downstream generative tasks.

| CPT Config                      | Acc.  |
| ------------------------------- | ----- |
| No CPT                          | 6.04  |
| FineMath 50BT                   | 19.27 |
| FineWeb 1.6BT + FineMath 48.4BT | 16.21 |
| FineWeb 8BT + FineMath 42BT     | 21.01 |
| FineWeb 16BT + FineMath 34BT    | 15.22 |

Table 2: GSM8K-Platinum performance (Pass@1 accuracy) of pretrained model 1B-160BT continued pretrained with various configurations and then fine-tuned using 100K SFT examples with 1 epoch.

➠ Takeaway 3. CPT on domain-specific data induces catastrophic forgetting of pre-trained knowledge which could harm both upstream and downstream performance, while incorporating a small replay budget (e.g., 5%) could effectively mitigate this degradation.

In Figure 4, we plot downstream performance of both SFT and SFT+RL models as a function of CPT budget (with a fixed 8 BT replay budget). All variants improve steadily with more domain-specific tokens up to around 32 BT and then plateau by 42 BT. For instance, the ID-greedy accuracy of the SFT model rises from about 5% at 2 BT to 12% at 32 BT before leveling off. Such a trend is also observed in OOD metrics. Across the CPT range, RL fine-tuning consistently outperforms pure SFT; notably, without CPT, RL can even underperform SFT (as seen in Maj@16, RM@16, and Pass@16), yet the gain brought by RL tends to strengthen as CPT tokens increase.

Figure 3: Upstream task performance vs. CPT tokens on models 1B-160BT, 1B-160BT-8+{2BT, ..., 42BT}, 1B-160BT-0+{10BT, ..., 50BT}, 1B-160BT-{1.6+48.4BT, 16+34BT}.

Figure 4: Downstream task performance vs. continued pre-training tokens on models:
- SFT: 1B-160BT-100Kep1, 1B-160BT-8+{2BT, ..., 42BT}-100Kep1
- SFT+RL: 1B-160BT-100Kep1-100Kep8, 1B-160BT-8+{2BT, ..., 42BT}-100Kep1-100Kep8.


![Figure 15](image_15.jpg)
<!-- Caption: Experiments, index 15 -->



![Figure 16](image_16.jpg)
<!-- Caption: Experiments, index 16 -->


➠ Takeaway 4. Domain-specific post-training should be supported by adequate domain-specific CPT data: without it, SFT performance remains suboptimal and RL can even degrade such performance.

➠ Takeaway 5. As domain-specific CPT data increase, in-domain downstream performance steadily improves and SFT models can benefit more from RL fine-tuning.

➠ Takeaway 6. With sufficient domain-specific CPT data, post-training on in-domain tasks not only improves in-domain performance but also generalizes effectively to OOD tasks.

3.3 Scaling Up SFT Compute

|        |                             | SFT                         |                             | SFT+RL                      |                                        |                                          |                                 |
| :----: | --------------------------- | --------------------------- | --------------------------- | --------------------------- | -------------------------------------- | ---------------------------------------- | ------------------------------- |
|        | Greedy                      | Maj@16                      | RM@16                       | Pass@16                     | ORM (avg@16)                           | Correct Ratio@16                          |                                 |
|   ID   | 0.5<br/>0.4<br/>0.3<br/>0.2 | 0.5<br/>0.4<br/>0.3<br/>0.2 | 0.5<br/>0.4<br/>0.3<br/>0.2 | 0.5<br/>0.4<br/>0.3<br/>0.2 | 5.0<br/>7.5<br/>10.0<br/>12.5<br/>15.0 | 0.35<br/>0.30<br/>0.25<br/>0.20<br/>0.15 |                                 |
|        | OOD                         | 0.6<br/>0.5<br/>0.4<br/>0.3 | 0.6<br/>0.5<br/>0.4<br/>0.3 | 0.6<br/>0.5<br/>0.4<br/>0.3 | 0.6<br/>0.5<br/>0.4<br/>0.3            | 18<br/>20<br/>22<br/>24                  | 0.35<br/>0.30<br/>0.25<br/>0.20 |
|        |                             | 4 8 16 32                   | 4 8 16 32                   | 4 8 16 32                   | 4 8 16 32                              | 4 8 16 32                                | 4 8 16 32                       |
| Epochs |                             |                             |                             |                             |                                        |                                          |                                 |

Figure 5: Downstream task performance vs. number of SFT epochs for models:
- SFT: 1B-160BT-8+42BT-100K ep{1,2,4,8,16,32}
- SFT+RL: 1B-160BT-8+42BT-100K ep{1,2,4,8,16,32}-100K ep8.

|              |                                     | SFT                                 |                                     | SFT+RL                              |                                            |                                                   |                                 |
| :----------: | ----------------------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------- | ------------------------------------------ | ------------------------------------------------- | ------------------------------- |
|              | Greedy                              | Maj@16                              | RM@16                               | Pass@16                             | ORM (avg@16)                                | Correct Ratio@16                                   |                                 |
|      ID      | 0.5<br/>0.4<br/>0.3<br/>0.2<br/>0.1 | 0.5<br/>0.4<br/>0.3<br/>0.2<br/>0.1 | 0.5<br/>0.4<br/>0.3<br/>0.2<br/>0.1 | 0.5<br/>0.4<br/>0.3<br/>0.2<br/>0.1 | 6<br/>8<br/>10<br/>12<br/>14<br/>16<br/>18 | 0.35<br/>0.30<br/>0.25<br/>0.20<br/>0.15<br/>0.40 |                                 |
|              | OOD                                 | 0.6<br/>0.5<br/>0.4<br/>0.3         | 0.6<br/>0.5<br/>0.4<br/>0.3         | 0.6<br/>0.5<br/>0.4<br/>0.3         | 0.6<br/>0.5<br/>0.4<br/>0.3                | 18<br/>20<br/>22<br/>24                           | 0.35<br/>0.30<br/>0.25<br/>0.20 |
|              |                                     | 100 200 300 400                     | 100 200 300 400                     | 100 200 300 400                     | 100 200 300 400                            | 100 200 300 400                                   | 100 200 300 400                 |
| Examples (k) |                                     |                                     |                                     |                                     |                                            |                                                   |                                 |

Figure 6: Downstream task performance vs. number of SFT examples for models:
- SFT: 1B-160BT-8+42BT-{50K, 100K, 150K, ..., 400K} ep1
- SFT+RL: 1B-160BT-8+42BT-{50K, 100K, 150K, ..., 400K} ep1-100K ep8.

To evaluate how downstream performance responds to increased SFT compute, we conduct two complementary studies using 1B-160BT-8+42BT as the base model.

Varying SFT epochs. Holding SFT examples fixed at 100K, we fine-tune the base model for {1, 2, 4, 8, 16, 32} epochs. As shown in Figure 5, ID metrics increase steadily with more epochs and saturate at around 8 epochs, reflecting increased memorization of solving in-domain problems. In contrast, OOD performance peaks at 2–4 epochs before declining, indicating that over-specialization hinders generalization. These findings also validate the commonly chosen SFT hyperparameter of approximately 3 epochs. Moreover, the marginal gains from downstream RL fine-tuning shrink on over-trained SFT models: once the model has excessively memorized the supervised data, there is little room for RL to improve.


![Figure 19](image_19.jpg)
<!-- Caption: Experiments, index 19 -->



![Figure 20](image_20.jpg)
<!-- Caption: Experiments, index 20 -->


Varying SFT dataset size. As proposed by a previous study [42], post-training performance for downstream tasks follows a power-law relationship with SFT dataset size, but the conclusion is drawn from experiments conducted on up to 10K examples. We further scale that budget by varying the number of SFT examples from 50K to 400K, holding epochs fixed at one to minimize memorization. As illustrated in Figure 6, ID performance improves monotonically with more examples, confirming that additional SFT compute consistently improves performance on in-domain tasks. However, OOD metrics fluctuate and can even decline with larger datasets. Similar to scaling up epochs, the incremental benefit from RL diminishes as the model learns more SFT examples.

➠ Takeaway 7. Excessive SFT improves ID performance with diminishing returns but does not necessarily improve and can even degrade OOD performance.

➠ Takeaway 8. Excessive SFT, especially overly large epochs, could limit further RL improvements.

3.4 Scaling Up RL Compute

Similarly, to evaluate how downstream performance responds to increased RL compute, we also vary either epochs or dataset size, using a 1B-160BT-8+42BT-100K-ep1 base model. We additionally incorporate 0 epochs/examples to indicate the SFT baseline. More experimental results and findings regarding RL can be found in Appendix A.2.

(a) Performance vs. number of RL epochs for models 1B-160BT-8+42BT-100K-ep1-100K-ep{0, 1, 2, 4, 8, 16, 32}. (b) Performance vs. number of RL examples for models 1B-160BT-8+42BT-100K-ep1-{0, 50K, 100K, 150K, ..., 400K}-ep1.

Figure 7: Downstream task performance under different RL scales.

Varying RL epochs. We apply RL across another 100K examples (disjoint from the SFT dataset) for {0, 1, 2, 4, 8, 16, 32} epochs. As shown in Figure 7a, for both ID and OOD tasks, greedy, Maj@16, and RM@16 accuracies peak at around 8–16 epochs and then saturate thereafter. We also notice that while the correct ratio keeps increasing, Pass@16 accuracy greatly degrades beyond 4 epochs, indicating that RL primarily sharpens confidence in already-correct outputs rather than effectively expanding the set of solvable samples. This is also reflected by results in Table 1: For 1B and 4B SFT models, Maj@16 accuracy could sometimes underperform greedy accuracy, indicating that low-quality solutions take the majority. However, after RL is conducted on the SFT models, all Maj@16 accuracies are higher than greedy accuracies.

Varying RL dataset size. Given a fixed epoch of 8, we vary the RL dataset size from 0 to 400K examples. Figure 7b shows that for both ID and OOD metrics, greedy, Maj@16, and RM@16 accuracies continue to increase from more data up to around 150–200K examples, after which gains flatten and fluctuate. In contrast, Pass@16 saturates much earlier and starts to degrade, while the correct ratio keeps increasing, similar to the finding in scaling up RL epochs. This finding is in line with observations by concurrent work [64] that similarly concludes that RL mainly boosts the confidence of existing correct outputs rather than enhancing the fundamental reasoning capabilities of LMs. The drastic performance drop observed at 350K and 400K RL examples appears correlated with unreported training instabilities, such as excessive sequence lengths exceeding model context windows and insufficient truncation policies. Additional training diagnostics are needed to confirm whether these drops reflect model failures or experimental artifacts.


![Figure 25](image_25.jpg)
<!-- Caption: Experiments, index 25 -->



![Figure 26](image_26.jpg)
<!-- Caption: Experiments, index 26 -->


Training results show that during the final RL steps, both models learn to greatly increase response length, and their generations often exceed their predefined context-window lengths, causing the performance drop. However, RL with overly large epochs is much more stable, and such collapse caused by response-length scaling is not observed.

Takeaway 9. RL with excessive epochs or examples improves downstream performance on both ID and OOD tasks but with diminishing returns (saturation happens at 4–8 epochs or 50–100K examples in our study).

Takeaway 10. Beyond the saturation regime, RL primarily increases the probability of sampling high-quality rollouts but does not necessarily improve models’ fundamental reasoning capabilities.

To further investigate how to configure SFT and RL data allocation in data-constrained scenarios, we subsample 100K examples from the entire 500K dataset and evaluate five SFT/RL splits: (10/90, 30/70, 50/50, 70/30, 90/10) K, and conduct either SFT or RL for 4 epochs. We choose 100K because it is around the saturation regime of both ID and OOD performance (Figure 7b). As shown in Figure 8, ID accuracy (greedy and Pass@16) increases with the proportion of SFT data, plateauing beyond around 70K, whereas OOD metrics are driven by RL allocation, peaking at 10K SFT (i.e., 90K RL). These trends hold across both the 1B and 4B models.

Figure 8: Downstream task performance for {1B, 4B}-160BT-8+42BT-{10K,…,90K} ep4-{90K,…,10K} ep4. Darker green/blue denotes more data allocation to SFT/RL. The total number of examples is fixed at 100K.

Takeaway 11. Under a constrained downstream data budget, allocating more examples to SFT maximizes in-domain gains at the expense of weaker OOD generalization, while allocating more to RL improves OOD performance.

4 Additional Studies and Discussions

Given that we find post-training interacts non-trivially with pre-training—necessitating a sophisticated training recipe—does downstream performance scale smoothly or predictably? This section provides one example illustrating why our comprehensive study is essential to fully grasp how training dynamics shape downstream performance in LMs, and another example where a metric could correlate with downstream problem-solving performance.

4.1 Intermediate Checkpoints May Not Be Reliable Surrogates

While Section 4.1 highlights limitations of intermediate checkpoints as proxies, Section 3 experiments rely extensively on such checkpoints for performance evaluation across token budgets. This conflicting usage suggests a methodological inconsistency that undermines conclusions drawn from intermediate checkpoint analysis. After each model sees 20B or 40B tokens, we further apply a single epoch of SFT on 100K examples to deliver a basic conversational grounding, and evaluate the models on the two easiest subsets of the MATH dataset.

Model results:
- 20BT full: Upstream 46.43; Math Level 1 (Greedy / Pass@16) 2.75 / 17.85; Math Level 2 (Greedy / Pass@16) 3.36 / 15.10.
- 20BT int.: Upstream 46.07; Math Level 1 2.52 / 11.44; Math Level 2 1.90 / 12.64.
- 40BT full: Upstream 49.38; Math Level 1 2.97 / 17.96; Math Level 2 3.36 / 14.88.
- 40BT int.: Upstream 49.06; Math Level 1 1.37 / 9.38; Math Level 2 2.68 / 8.72.

Table 3: Performance on upstream tasks and MATH (Level 1 and 2) under different pretraining configurations. “xBT full” refers to a complete pre-training run on xBT, while “xBT int.” refers to an intermediate checkpoint taken during training to 160B tokens, corresponding to xBT seen so far.


![Figure 30](image_30.jpg)
<!-- Caption: Experiments, index 30 -->


As Table 3 shows, the intermediate checkpoints consistently lag behind their dedicated 20B and 40B counterparts on both upstream task accuracy and math reasoning performance. This gap arises because earlier stopping points—captured before learning-rate decay and data repetition—omit the full optimization trajectory that smaller runs complete. In other words, simply slicing out a 40B-token checkpoint from a longer schedule does not reproduce the benefits of training a model exclusively for 40B tokens.

These results caution against using such intermediate checkpoints as proxies for studying and understanding fully trained smaller models. When interpreting training dynamics, it is essential to compare like-for-like runs—each with its own complete schedule—rather than relying on mid-course snapshots that understate true model capability.

## 4.2 Correlating Downstream Task Performance with ORM Score

While perplexity across domains sometimes shows strong correlations, downstream task accuracy may not be consistently correlated, largely because post-trained models are miscalibrated and thus lower validation perplexity does not necessarily indicate better generative performance. In our experiments, we found that the correlation between ORM scores and downstream task accuracy presents a clear relationship. In Figure 9, we plot ORM score (avg@16) versus Maj@16 accuracies for all post-trained model variants starting from base model 1B-160BT-8+42BT and find that ORM scores exhibit consistently strong predictive power, evidenced by high correlation coefficients ranging approximately from 0.62 to 0.84 across both ID and OOD tasks. While we observe that the correlation is low for StrategyQA, this might arise because 1) StrategyQA emphasizes more about commonsense knowledge rather than explicit deductive reasoning, or 2) the reward model used is less suited to the specific problem distribution of this dataset.

Figure 9: Correlation between accuracy and ORM score across different tasks. Each subplot represents one dataset, where each point corresponds to a model variant. A dashed line indicates the linear trend, and the Pearson correlation coefficient is reported in each title.

The non-trivial correlation between ORM scores and downstream accuracies suggests that scores produced by large ORMs can serve as reliable unsupervised proxy metrics for assessing generation quality during post-training phases. For example, ORM scores can be particularly useful in data-constrained scenarios where collecting sufficient high-quality test examples is challenging. ORM scoring is also advantageous when direct testing is impractical, such as in tasks where final answers are inherently difficult to automatically extract and verify. Moreover, the generalizability of ORMs enables practitioners to train them on existing reasoning tasks and apply to other data-constrained reasoning tasks. Under such circumstances, ORM scores enable effective validation and iterative refinement of models without the reliance on extensive labeled evaluation datasets.

➠ Takeaway 12. ORM score could be a more reliable unsupervised validation metric that helps predict downstream task performance during post-training, compared to validation loss. Notably, ORM scores from an 8B reward model correlate well with problem-solving accuracies of 1B models on many downstream reasoning tasks.


## Related Work


# 5 Related Work

Studying Language Models Across Training Stages. Recent research has explored how different training stages shape downstream capabilities of language models. Observations by recent study [16] indicate that extensively pre-trained language models scale reliably on downstream tasks, though their conclusions predominantly address pre-trained models evaluated via top-1 error, leaving open questions regarding models subjected to additional post-training. In contrast, “catastrophic overtraining” is identified by recent work [51]: Prolonged pre-training beyond a certain point actually impairs downstream fine-tuning by increasing sensitivity to parameter updates and exacerbating forgetting. Complementing this, researchers [66] have derived a multiplicative joint scaling law for


![Figure 34](image_34.jpg)
<!-- Caption: Related Work, index 34 -->


Fine-tuning shows performance gains depend more on scaling model size than pre-training data, with optimal approaches depending critically on task and data regimes.

Pre-training Drives Post-training. Prior research has examined how post-training interacts with pre-trained model capabilities. Some studies [26] use observational analyses showing upstream capabilities relate to model compute, affecting specialized reasoning skills. Other works explore RL post-training effects, indicating it tends to amplify existing pretrained behaviors and biases rather than fundamentally enhance reasoning [70, 64]. These studies provide important context for understanding post-training dynamics.

Scaling Laws for Language Models. Early scaling work [23, 27] established fundamental relationships linking training loss to model size, data quantity, and compute. Recent studies have extended this framework in several ways. A dual-axis scaling law has shown reliable loss predictions even in highly over-trained regimes, significantly beyond traditional optimal compute points [16]. Additionally, new quantitative models predict emergent behaviors in model accuracy either through explicit loss thresholds or by probing with targeted finetuning [50, 14]. Cross-distribution transferability has also been modeled, allowing accurate extrapolations of loss curves between different datasets from minimal pilot data [7]. Further refinements address data-limited contexts, deriving optimal epoch allocation when unique training data is scarce [37], and revealing similar scaling patterns for synthetic data with clear diminishing returns [40]. Moreover, scaling laws now capture continual pre-training dynamics, guiding mixing domain-specific and general data, and quantifying forgetting effects during domain adaptation with replay data [41]. Finally, research into compute allocation has developed scaling relationships specifically for distillation, determining precisely when distillation methods surpass direct pre-training efficiency [9].

Post-training for Reasoning. Recent research has investigated the impact of post-training strategies on the reasoning capabilities of LLMs. One study challenges the “Superficial Alignment Hypothesis” [72], demonstrating that SFT post-training performance scales with the number of fine-tuning examples, akin to pre-training scaling laws [42]. Moreover, RL post-training has been shown to amplify behaviors acquired during pre-training, particularly in tasks requiring advanced mathematical reasoning and coding [70]. A comparative study indicates that while SFT tends to memorize training data, RL fosters better generalization [10]. Investigations into the mechanics of reasoning have demystified long chain-of-thought learned through RL, identifying factors that enable the generation of extended reasoning trajectories [61]. Conversely, a critical examination questions whether RL truly incentivizes reasoning capacities beyond what is already learned during pre-training, suggesting that RL may not elicit fundamentally new reasoning patterns [64].


## Conclusion


6 Concluding Remarks

In this work, we systematically studied how factors such as training tokens and model size influence language models’ upstream and downstream performance. Our study found indications of scaling trends and diminishing returns for models up to 4B parameters, alongside tentative evidence of mitigation strategies for forgetting during domain-specific pre-training. While ORM scores appear to have some predictive utility, further work is needed to validate these findings across larger scales and diverse domains. We acknowledge limitations including model scale, dataset scope, and lack of thorough hyperparameter tuning, which constrain the generality of conclusions and warrant cautious interpretation. Second, our focus on reasoning-centric post-training objectives leaves unexplored dynamics for objectives like safety alignment, instruction-following, tool-calling, and coding tasks. Lastly, our RL experiments employed only Proximal Policy Optimization (PPO) with verifiable rewards. Exploring alternative reinforcement learning methods could offer broader insights into their effects on downstream capabilities.

Broadly, we advocate open-source research to enhance transparency, enabling better understanding, controlling, and responsibly managing machine learning models through community efforts.


## References


References

[1] Lightning AI. LitGPT. https://github.com/Lightning-AI/litgpt, 2023.

[2] L. B. Allal, A. Lozhkov, E. Bakouch, G. M. Blázquez, G. Penedo, L. Tunstall, A. Marafioti, H. Kydlíček, A. P. Lajarín, V. Srivastav, J. Lochner, C. Fahlgren, X.-S. Nguyen, C. Fourrier, B. Burtenshaw, H. Larcher, H. Zhao, C. Zakka, M. Morlon, C. Raffel, L. von Werra, and T. Wolf. SmolLM2: When Smol Goes Big – Data-centric Training of a Small Language Model, 2025.

[3] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, et al. Qwen Technical Report. arXiv preprint arXiv:2309.16609, 2023.

[4] L. Bethune, D. Grangier, D. Busbridge, E. Gualdoni, M. Cuturi, and P. Ablin. Scaling Laws for Forgetting During Fine-tuning with Pretraining Data Injection. arXiv preprint arXiv:2502.06042, 2025.

[5] S. Biderman, H. Schoelkopf, Q. G. Anthony, H. Bradley, K. O’Brien, E. Hallahan, M. A. Khan, S. Purohit, U. S. Prashanth, E. Raff, et al. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. In International Conference on Machine Learning, pages 2397–2430. PMLR, 2023.

[6] Y. Bisk, R. Zellers, R. Le Bras, J. Gao, and Y. Choi. PIQA: Reasoning about Physical Commonsense in Natural Language. In Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020.

[7] D. Brandfonbrener, N. Anand, N. Vyas, E. Malach, and S. Kakade. Loss-to-Loss Prediction: Scaling Laws for All Datasets. arXiv preprint arXiv:2411.12925, 2024.

[8] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language Models Are Few-Shot Learners. Advances in Neural Information Processing Systems, 33:1877–1901, 2020.

[9] D. Busbridge, A. Shidani, F. Weers, J. Ramapuram, E. Littwin, and R. Webb. Distillation Scaling Laws. arXiv preprint arXiv:2502.08606, 2025.

[10] T. Chu, Y. Zhai, J. Yang, S. Tong, S. Xie, D. Schuurmans, Q. V. Le, S. Levine, and Y. Ma. SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training. arXiv preprint arXiv:2501.17161, 2025.

[11] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think You Have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv preprint arXiv:1803.05457, 2018.

[12] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training Verifiers to Solve Math Word Problems. arXiv preprint arXiv:2110.14168, 2021.

[13] R. Dominguez-Olmedo, F. E. Dorner, and M. Hardt. Training on the Test Task Confounds Evaluation and Emergence. arXiv preprint arXiv:2407.07890, 2024.

[14] Z. Du, A. Zeng, Y. Dong, and J. Tang. Understanding Emergent Abilities of Language Models from the Loss Perspective. arXiv preprint arXiv:2403.15796, 2024.

[15] R. M. French. Catastrophic Forgetting in Connectionist Networks. Trends in Cognitive Sciences, 3(4):128–135, 1999.

[16] S. Y. Gadre, G. Smyrnis, V. Shankar, S. Gururangan, M. Wortsman, R. Shao, J. Mercat, A. Fang, J. Li, S. Keh, et al. Language Models Scale Reliably with Overtraining and on Downstream Tasks. arXiv preprint arXiv:2403.08540, 2024.

[17] K. Gandhi, A. Chakravarthy, A. Singh, N. Lile, and N. D. Goodman. Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STARs, 2025.

[18] M. Geva, D. Khashabi, E. Segal, T. Khot, D. Roth, and J. Berant. Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies. Transactions of the Association for Computational Linguistics, 9:346–361, 2021. 3, 20

[19] A. Gu, B. Rozière, H. Leather, A. Solar-Lezama, G. Synnaeve, and S. I. Wang. CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution. arXiv preprint arXiv:2401.03065, 2024. 3, 20

[20] D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring Mathematical Problem Solving with the MATH Dataset. arXiv preprint arXiv:2103.03874, 2021. 2, 3, 20

[21] D. Hernandez, T. Brown, T. Conerly, N. DasSarma, D. Drain, S. El-Showk, N. Elhage, Z. Hatfield-Dodds, T. Henighan, T. Hume, et al. Scaling Laws and Interpretability of Learning from Repeated Data. arXiv preprint arXiv:2205.10487, 2022. 1

[22] J. Hestness, S. Narang, N. Ardalani, G. Diamos, H. Jun, H. Kianinejad, M. M. A. Patwary, Y. Yang, and Y. Zhou. Deep Learning Scaling Is Predictable, Empirically. arXiv preprint arXiv:1712.00409, 2017. 1

[23] J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. Training Compute-Optimal Large Language Models. arXiv preprint arXiv:2203.15556, 2022. 1, 2, 10

[24] S. Hu, Y. Tu, X. Han, C. He, G. Cui, X. Long, Z. Zheng, Y. Fang, Y. Huang, W. Zhao, et al. MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies. arXiv preprint arXiv:2404.06395, 2024. 1

[25] A. Ibrahim, B. Thérien, K. Gupta, M. L. Richter, Q. Anthony, T. Lesort, E. Belilovsky, and I. Rish. Simple and Scalable Strategies to Continually Pre-Train Large Language Models. arXiv preprint arXiv:2403.08763, 2024. 2, 5

[26] J. Jin, V. Syrgkanis, S. Kakade, and H. Zhang. Discovering Hierarchical Latent Capabilities of Language Models via Causal Representation Learning, 2025. 10

[27] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling Laws for Neural Language Models. arXiv preprint arXiv:2001.08361, 2020. 1, 10

[28] M. Kazemi, Q. Yuan, D. Bhatia, N. Kim, X. Xu, V. Imbrasaite, and D. Ramachandran. BoardgameQA: A Dataset for Natural Language Reasoning with Contradictory Information. Advances in Neural Information Processing Systems, 36:39052–39074, 2023. 3, 20

[29] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023. 20

[30] H. Li, W. Zheng, J. Hu, Q. Wang, H. Zhang, Z. Wang, S. Xuyang, Y. Fan, S. Zhou, X. Zhang, et al. Predictable Scale: Part I – Optimal Hyperparameter Scaling Law in Large Language Model Pretraining. arXiv preprint arXiv:2503.04715, 2025. 1

[31] J. Li, E. Beeching, L. Tunstall, B. Lipkin, R. Soletskyi, S. C. Huang, K. Rasul, L. Yu, A. Jiang, Z. Shen, Z. Qin, B. Dong, L. Zhou, Y. Fleureau, G. Lample, and S. Polu. NuminaMath. https://huggingface.co/AI-MO/NuminaMath-CoT, 2024. 1, 2, 20

[32] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al. DeepSeek-V3 Technical Report. arXiv preprint arXiv:2412.19437, 2024. 1

[33] C. Y. Liu, L. Zeng, J. Liu, R. Yan, J. He, C. Wang, S. Yan, Y. Liu, and Y. Zhou. Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs. arXiv preprint arXiv:2410.18451, 2024. 3

[34] E. Liu, A. Bertsch, L. Sutawika, L. Tjuatja, P. Fernandes, L. Marinov, M. Chen, S. Singhal, C. Lawrence, A. Raghunathan, et al. Not-Just-Scaling Laws: Towards a Better Understanding of the Downstream Impact of Language Model Design Decisions. arXiv preprint arXiv:2503.03862, 2025. 1
[35] P. Lu, L. Qiu, K.-W. Chang, Y. N. Wu, S.-C. Zhu, T. Rajpurohit, P. Clark, and A. Kalyan. Dynamic Prompt Learning via Policy Gradient for Semi-Structured Mathematical Reasoning. arXiv preprint arXiv:2209.14610, 2022. 3, 21
[36] T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal. Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering. In EMNLP, 2018. 3
[37] N. Muennighoff, A. Rush, B. Barak, T. Le Scao, N. Tazi, A. Piktus, S. Pyysalo, T. Wolf, and C. A. Raffel. Scaling Data-Constrained Language Models. Advances in Neural Information Processing Systems, 36:50358–50376, 2023. 10
[38] G. Penedo, H. Kydlíček, A. Lozhkov, M. Mitchell, C. A. Raffel, L. von Werra, T. Wolf, et al. The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. Advances in Neural Information Processing Systems, 37:30811–30849, 2024. 1, 2, 20
[39] Z. Qi, M. Ma, J. Xu, L. L. Zhang, F. Yang, and M. Yang. Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers. arXiv preprint arXiv:2408.06195, 2024. 2
[40] Z. Qin, Q. Dong, X. Zhang, L. Dong, X. Huang, Z. Yang, M. Khademi, D. Zhang, H. H. Awadalla, Y. R. Fung, et al. Scaling Laws of Synthetic Data for Language Models. arXiv preprint arXiv:2503.19551, 2025. 10
[41] H. Que, J. Liu, G. Zhang, C. Zhang, X. Qu, Y. Ma, F. Duan, Z. Bai, J. Wang, Y. Zhang, et al. D-CPT Law: Domain-Specific Continual Pre-Training Scaling Law for Large Language Models. Advances in Neural Information Processing Systems, 37:90318–90354, 2024. 2, 10
[42] M. Raghavendra, V. Nath, and S. Hendryx. Revisiting the Superficial Alignment Hypothesis. arXiv preprint arXiv:2410.03717, 2024. 1, 7, 10
[43] Y. Ren and D. J. Sutherland. Learning Dynamics of LLM Fine-Tuning. arXiv preprint arXiv:2407.10490, 2024. 1
[44] K. Sakaguchi, R. Le Bras, C. Bhagavatula, and Y. Choi. WinoGrande: An Adversarial Winograd Schema Challenge at Scale. Communications of the ACM, 64(9):99–106, 2021. 3
[45] R. Schaeffer, B. Miranda, and S. Koyejo. Are Emergent Abilities of Large Language Models a Mirage? Advances in Neural Information Processing Systems, 36:55565–55581, 2023. 1
[46] F. Schmidt. Generalization in Generation: A Closer Look at Exposure Bias. arXiv preprint arXiv:1910.00292, 2019. 1
[47] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347, 2017. 2
[48] Y. Shen, M. Stallone, M. Mishra, G. Zhang, S. Tan, A. Prasad, A. M. Soria, D. D. Cox, and R. Panda. Power Scheduler: A Batch Size and Token Number Agnostic Learning Rate Scheduler. arXiv preprint arXiv:2408.13359, 2024. 1
[49] G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, Y. Peng, H. Lin, and C. Wu. HybridFlow: A Flexible and Efficient RLHF Framework. arXiv preprint arXiv:2409.19256, 2024. 1
[50] C. Snell, E. Wallace, D. Klein, and S. Levine. Predicting Emergent Capabilities by Fine-Tuning. arXiv preprint arXiv:2411.16035, 2024. 10
[51] J. M. Springer, S. Goyal, K. Wen, T. Kumar, X. Yue, S. Malladi, G. Neubig, and A. Raghunathan. Overtrained Language Models Are Harder to Fine-Tune. arXiv preprint arXiv:2503.19206, 2025. 1, 4, 9

[52] Gemini Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1

[53] K. Tirumala, A. Markosyan, L. Zettlemoyer, and A. Aghajanyan. Memorization without overfitting: Analyzing the training dynamics of large language models. Advances in Neural Information Processing Systems, 35:38274–38290, 2022. 1

[54] H. Tissue, V. Wang, and L. Wang. Scaling law with learning rate annealing. arXiv preprint arXiv:2408.11029, 2024. 1

[55] S. Toshniwal, W. Du, I. Moshkov, B. Kisacanin, A. Ayrapetyan, and I. Gitman. OpenMathInstruct-2: Accelerating AI for math with massive open-source instruction data. arXiv preprint arXiv:2410.01560, 2024. 1, 2, 20

[56] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2, 17

[57] J. Vendrow, E. Vendrow, S. Beery, and A. Madry. Do large language model benchmarks test reliability? arXiv preprint arXiv:2502.03461, 2025. 3, 20

[58] Y. Wang, Q. Yang, Z. Zeng, L. Ren, L. Liu, B. Peng, H. Cheng, X. He, K. Wang, J. Gao, et al. Reinforcement learning for reasoning in large language models with one training example. arXiv preprint arXiv:2504.20571, 2025. 17

[59] M. Xia, M. Artetxe, C. Zhou, X. V. Lin, R. Pasunuru, D. Chen, L. Zettlemoyer, and V. Stoyanov. Training trajectories of language models across scales. arXiv preprint arXiv:2212.09803, 2022. 1

[60] A. Yang, B. Zhang, B. Hui, B. Gao, B. Yu, C. Li, D. Liu, J. Tu, J. Zhou, J. Lin, et al. Qwen2.5-Math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024. 20

[61] E. Yeo, Y. Tong, M. Niu, G. Neubig, and X. Yue. Demystifying long chain-of-thought reasoning in LLMs. arXiv preprint arXiv:2502.03373, 2025. 1, 10

[62] Ç. Yıldız, N. K. Ravichandran, N. Sharma, M. Bethge, and B. Ermis. Investigating continual pre-training in large language models: Insights and implications. arXiv preprint arXiv:2402.17400, 2024. 2

[63] L. Yu, W. Jiang, H. Shi, J. Yu, Z. Liu, Y. Zhang, J. T. Kwok, Z. Li, A. Weller, and W. Liu. MetaMath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023. 1, 2, 20

[64] Y. Yue, Z. Chen, R. Lu, A. Zhao, Z. Wang, S. Song, and G. Huang. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? arXiv preprint arXiv:2504.13837, 2025. 7, 10

[65] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi. HellaSwag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019. 3

[66] B. Zhang, Z. Liu, C. Cherry, and O. Firat. When scaling meets LLM finetuning: The effect of data, model and finetuning method. arXiv preprint arXiv:2402.17193, 2024. 1, 9

[67] H. Zhang, D. Morwani, N. Vyas, J. Wu, D. Zou, U. Ghai, D. Foster, and S. M. Kakade. How does critical batch size scale in pre-training? In The Thirteenth International Conference on Learning Representations, 2025. 1

[68] P. Zhang, G. Zeng, T. Wang, and W. Lu. TinyLlama: An open-source small language model. arXiv preprint arXiv:2401.02385, 2024. 17

[69] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin, et al. OPT: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022. 17

[70] R. Zhao, A. Meterez, S. Kakade, C. Pehlevan, S. Jelassi, and E. Malach. Echo chamber: RL post-training amplifies behaviors learned in pretraining. arXiv preprint arXiv:2504.07912, 2025. 1, 10

[71] Y. Zheng, R. Zhang, J. Zhang, Y. Ye, Z. Luo, Z. Feng, and Y. Ma. LLaMA-Factory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. 1

[72] C. Zhou, P. Liu, P. Xu, S. Iyer, J. Sun, Y. Mao, X. Ma, A. Efrat, P. Yu, L. Yu, et al. LIMA: Less is more for alignment. Advances in Neural Information Processing Systems, 36:55006–55021, 2023. 10


## Appendix


Appendices

A Additional Experiment Results
A.1 Observational Comparison of Pre-trained Models . . . . . . . . . . . . . . . . . . 17
A.2 Scaling Up RL Compute . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
A.3 Post-trained Models are Miscalibrated for Language Modeling Tasks . . . . . . . . 18

B Reproducibility
B.1 Model Architectures . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
B.2 Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
B.2.1 Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
B.2.2 SFT/RL Template . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
B.2.3 Training Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
B.3 Evaluation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
B.3.1 Benchmarks and Sampling Parameters . . . . . . . . . . . . . . . . . . . . . 20
B.3.2 Statistical Significance . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
B.3.3 Example Model Outputs . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

A Additional Experiment Results

A.1 Observational Comparison of Pre-trained Models

| Model Name   | Tokens | H/S   | W/G   | PIQA  | OBQA  | ARC-E | ARC-C | Avg.  |
| ------------ | ------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| OPT 1.3B     | 300B   | 53.65 | 59.59 | 72.36 | 33.40 | 50.80 | 29.44 | 49.87 |
| Pythia 1B    | 300B   | 47.16 | 53.43 | 69.21 | 31.40 | 48.99 | 27.05 | 46.21 |
| Pythia 1.4B  | 300B   | 52.01 | 57.38 | 70.95 | 33.20 | 54.00 | 28.50 | 49.34 |
| TinyLlama 1B | 2T     | 61.47 | 59.43 | 73.56 | 36.80 | 55.47 | 32.68 | 53.23 |
| Llama 3.2 1B | 9T     | 63.66 | 60.46 | 74.54 | 37.00 | 60.48 | 35.75 | 55.31 |
| Qwen3 1.7B   | 36T    | 60.46 | 61.01 | 72.36 | 36.80 | 69.91 | 43.26 | 57.30 |
| 1B (ours)    | 20B    | 42.25 | 51.30 | 67.85 | 32.80 | 54.80 | 29.61 | 46.44 |
|              | 40B    | 47.53 | 54.62 | 69.59 | 36.20 | 58.08 | 30.29 | 49.38 |
|              | 80B    | 51.05 | 53.59 | 70.78 | 37.20 | 62.71 | 35.92 | 51.88 |
|              | 160B   | 52.30 | 53.99 | 71.71 | 36.60 | 63.09 | 36.09 | 52.30 |
|              | 320B   | 53.86 | 53.51 | 71.93 | 37.20 | 62.29 | 36.18 | 52.49 |
| Pythia 6.9B  | 300B   | 63.89 | 61.17 | 76.39 | 37.20 | 61.07 | 35.15 | 55.81 |
| OPT 6.7B     | 300B   | 67.18 | 65.35 | 76.50 | 37.40 | 60.06 | 34.73 | 56.87 |
| Qwen1.5 4B   | 3T     | 71.45 | 64.09 | 77.10 | 39.60 | 61.41 | 39.51 | 58.86 |
| Qwen2.5 3B   | 18T    | 73.61 | 68.51 | 78.89 | 42.00 | 73.23 | 47.18 | 63.90 |
| Qwen3 4B     | 36T    | 73.71 | 70.64 | 77.75 | 41.00 | 76.22 | 51.88 | 65.20 |
| Llama 3.2 3B | 9T     | 73.63 | 69.69 | 77.53 | 43.20 | 71.76 | 45.90 | 63.62 |
| 4B (ours)    | 80B    | 48.84 | 54.38 | 69.91 | 35.80 | 59.68 | 32.68 | 50.22 |
|              | 160B   | 56.49 | 55.88 | 72.63 | 40.20 | 66.67 | 39.93 | 55.30 |
|              | 320B   | 61.38 | 57.46 | 74.27 | 41.80 | 67.55 | 39.16 | 56.94 |

Table 4 compares EvoLM models to public baselines without harmonizing training hyperparameters or evaluation protocols. Differences in tokenizers, training schedules, and data distribution likely influence reported performances. Thus, claims of state-of-the-art competitiveness should be tempered pending standardized, fair comparisons accounting for these factors.

Table 4 compares our pre-trained models against several open-weight models including OPT [69], Pythia [5], TinyLlama [68], Llama [56], and Qwen [3]. Our models, pretrained on a significantly smaller number of tokens (320B tokens for our 1B and 4B models), demonstrate competitive performance with other state-of-the-art small models such as TinyLlama-1B (trained on 2T tokens) and Qwen1.5-4B (trained on 3T tokens).

Specifically, despite TinyLlama-1B and Qwen1.5-4B models being trained with 6.25x and 9.38x more tokens respectively, our 1B and 4B models achieve similar or slightly better results across standard benchmarks like HellaSwag (H/S), Winogrande (W/G), PIQA, OBQA, ARC-Easy (ARC-E), and ARC-Challenge (ARC-C). This empirical observation is consistent with our experimental findings in Section 3.1, highlighting diminishing returns from excessive pretraining: beyond a certain optimal compute threshold, additional pretraining leads to minimal incremental gains in general domain upstream task performance.

A.2 Scaling Up RL Compute

To further look into effective practice for scaling up RL compute, we plot results in “example-epochs” units (#examples × #epochs, in 10⁵) in Figure 10. We use the same configurations as Section 3.4. Under a fixed compute budget, allocating more epochs on a moderate dataset (e.g., 100K×8 = 800K example-epochs) typically yields higher ID and OOD performance than spreading compute over a larger dataset with fewer epochs, and RL with excessive training examples could sometimes lead to collapsed performance due to overly long and unfinished responses (shown by the crosses in Figure 10 and response length in Figure 11), while we do not observe such problems when conducting RL with excessive training epochs (shown in Figure 12). This demonstrates that deeper policy optimization per sample is more cost-effective than broader data coverage for RL scaling, which is consistent with findings proposed by [58] showing that RL using even only one training example could be effective in incentivizing the mathematical reasoning capabilities of LLMs.

Figure 10: Downstream task performance vs. RL compute. A cross mark indicates models that tend to generate responses longer than their context window limits.

|    | Baseline (No RL)                                  | Varying RL #epochs (fixed 100K examples)                                        |                                                         | Varying RL #examples (fixed 8 epochs)                      |                                                            |                                          |
| -- | ------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------- |
|    | Greedy                                            | Maj@16                                                                          | RM@16                                                   | Pass@16                                                    | ORM (avg@16)                                               | Correct Ratio@16                         |
| ID | 0.18 0.17 0.16 0.15 0.14 0.13                     | 0.225 0.200 0.175 0.150 0.125 0.100                                             | 0.250 0.225 0.200 0.175 0.150 0.125                     | 0.46 0.44 0.42 0.40 0.38 0.36 0.34                         | 10.0 12.5 15.0 17.5 20.0 22.5 25.0                        | 0.35 0.30 0.25 0.20 0.15                 |
|    | OOD                                               | 0.30 0.29 0.28 0.27 0.26 0.25 0.24                                              | 0.30 0.25 0.20 0.15 0.10                                | 0.35 0.30 0.25 0.20 0.15                                  | 0.65 0.60 0.55 0.50 0.45 0.40                              | 18 20 22 24 26                           |
|    |                                                   | 0 4 8 16 32                                                                     | 0 4 8 16 32                                             | 0 4 8 16 32                                               | 0 4 8 16 32                                                | 0 4 8 16 32 Compute (×10³ example-epochs) |

response_length/mean | response_length/mean
1000 | 1000
900 | 900
800 | 800
700 | 700
600 | 600
500 | 500
400 | 400
Step | Step
500 1k 1.5k 2k 2.5k 3k | 500 1k 1.5k 2k 2.5k 3k

Figure 11: Response length versus training step when tuning 1B-160BT-8+42BT-100Kep1-400Kep8.

Figure 12: Response length versus training step when tuning 1B-160BT-8+42BT-100Kep1-100Kep32.

## A.3 Post-trained Models are Miscalibrated for Language Modeling Tasks

Our upstream evaluations indicate that post-trained LMs exhibit significant miscalibration when assessed through validation PPL. We evaluate PPL on the validation set (disjoint from the training set) for each post-trained model. As illustrated in Figure 13, we observe negligible correlations between validation perplexity and downstream task accuracy across various datasets. Specifically, the Pearson correlation coefficients remain close to zero, reinforcing that low perplexity does not reliably predict enhanced generative reasoning performance. This contrasts sharply with the strong predictive capability exhibited by ORM scores, as discussed in Section 4.2. While validation perplexity is conventionally used to monitor model quality, it is insufficient for post-training phases, particularly when evaluating generative reasoning tasks. In practice, relying solely on perplexity as a validation metric could misguide resource allocation decisions during training.

|              | GSM8K-P (r²=0.019) | MATH (r²=0.219) | ID BGQA (r²=0.000) | OOD STGQA (r²=0.009) | Regression TabMWP (r²=0.006) | CRUXEval (r²=0.001) |
| ------------ | ------------------- | ---------------- | ------------------ | --------------------- | ----------------------------- | ------------------- |
| Perplexity   | 20 10               | 20 10            | 20 10              | 20 10                 | 20 10                         | 20 10               |
| Maj@16 Acc.  | 0.2 0.4             | 0.1 0.2          | 0.30 0.35          | 0.475 0.500 0.525 0.550 | 0.10 0.15 0.20               | 0.05 0.10 0.15      |

Figure 13: Correlation between accuracy and validation PPL across different tasks. Each subplot represents one dataset, where each point corresponds to a post-trained model variant. A dashed line indicates the linear trend, and the Pearson correlation coefficient is reported in each title.


![Figure 53](image_53.jpg)
<!-- Caption: Appendix, index 53 -->



![Figure 54](image_54.jpg)
<!-- Caption: Appendix, index 54 -->



![Figure 55](image_55.jpg)
<!-- Caption: Appendix, index 55 -->



![Figure 56](image_56.jpg)
<!-- Caption: Appendix, index 56 -->


B Reproducibility

B.1 Model Architectures

We show model architecture details for 0.5B, 1B and 4B models in Table 5. However, the dissemination of large pretrained models entails risks of misuse and ethical concerns about bias amplification, for which no mitigation strategies were implemented or described.

| Model Size | Hidden Size | Intermediate Size | Vocab Size | Context Length | # Heads | # Layers | # Query Groups |
| ---------- | ----------- | ----------------- | ---------- | -------------- | ------- | -------- | -------------- |
| 0.5B       | 1536        | 3216              | 32000      | 2048           | 32      | 20       | 4              |
| 1B         | 2048        | 4896              | 32000      | 2048           | 32      | 22       | 4              |
| 4B         | 4096        | 7792              | 32000      | 2048           | 32      | 28       | 4              |

Table 5: Model architecture details.

B.2 Training Details

B.2.1 Hyperparameters

Hyperparameters for pretraining/continued pretraining, SFT, and RL are shown in Table 6, Table 7, Table 8, respectively. We use the AdamW optimizer and up to 32 NVIDIA H100 80GB HBM3 GPUs for all training stages. For pretraining, continued pretraining, and SFT, we use a standard warmup-cosine-decay strategy for the learning rate schedule. For RL, we apply a warmup-constant learning rate schedule.

| 0.5B                | 0.5B       | 1B                  | 1B         | 4B                  | 4B         |
| ------------------- | ---------- | ------------------- | ---------- | ------------------- | ---------- |
| precision           | bf16-mixed | precision           | bf16-mixed | precision           | bf16-mixed |
| global\_batch\_size | 512        | global\_batch\_size | 512        | global\_batch\_size | 1024       |
| max\_seq\_length    | 2048       | max\_seq\_length    | 2048       | max\_seq\_length    | 2048       |
| lr\_warmup\_ratio   | 0.1        | lr\_warmup\_ratio   | 0.1        | lr\_warmup\_ratio   | 0.1        |
| max\_norm           | 1          | max\_norm           | 1          | max\_norm           | 1          |
| lr                  | 0.00025    | lr                  | 0.0002     | lr                  | 0.00015    |
| min\_lr             | 0.000025   | min\_lr             | 0.00002    | min\_lr             | 0.000015   |
| weight\_decay       | 0.1        | weight\_decay       | 0.1        | weight\_decay       | 0.1        |
| beta1               | 0.9        | beta1               | 0.9        | beta1               | 0.9        |
| beta2               | 0.95       | beta2               | 0.95       | beta2               | 0.95       |
| epoch               | 1          | epoch               | 1          | epoch               | 1          |

Table 6: Hyperparameters for pretraining/continued pretraining.

| 1B                  | 1B      | 4B                  | 4B        |
| ------------------- | ------- | ------------------- | --------- |
| cutoff\_len         | 2048    | cutoff\_len         | 2048      |
| batch\_size         | 128     | batch\_size         | 256       |
| learning\_rate      | 0.00001 | learning\_rate      | 0.0000075 |
| lr\_scheduler\_type | cosine  | lr\_scheduler\_type | cosine    |
| warmup\_ratio       | 0.1     | warmup\_ratio       | 0.1       |

Table 7: Hyperparameters for supervised finetuning.

B.2.2 SFT/RL Template

We use the following template for SFT and RL tuning:

Human: {prompt}
Assistant: {response}

| 1B                                      |          | 4B                                      |          |
| --------------------------------------- | -------- | --------------------------------------- | -------- |
| actor_lr                                | 2.00E-06 | actor_lr                                | 1.00E-06 |
| critic_lr                               | 2.00E-05 | critic_lr                               | 1.00E-05 |
| kl                                      | 0.0001   | kl                                      | 0.0001   |
| train_batch_size                        | 1024     | train_batch_size                        | 2048     |
| max_prompt_length                       | 1024     | max_prompt_length                       | 1024     |
| max_response_length                     | 1024     | max_response_length                     | 1024     |
| ppo_mini_batch_size                     | 1024     | ppo_mini_batch_size                     | 2048     |
| ppo_micro_batch_size_per_gpu            | 32       | ppo_micro_batch_size_per_gpu            | 16       |
| log_prob_micro_batch_size_per_gpu       | 64       | log_prob_micro_batch_size_per_gpu       | 32       |
| warmup_steps_ratio                      | 0.1      | warmup_steps_ratio                      | 0.1      |

Table 8: Hyperparameters for reinforcement learning (PPO).

## B.2.3 Training Data

FineWeb-Edu [38]: An extensive educational dataset sourced from web content, specifically designed for pretraining language models on high-quality academic and educational text. There are ∼1.3 trillion tokens in total.

FineMath [38]: A curated dataset of mathematical texts, problems, and solutions, intended to enhance language models’ mathematical knowledge. There are ∼50 billion tokens in total.

OpenMathInstruct2 [55], MetaMathQA [63], NuminaMath [31]: Instruction-tuning datasets containing mathematical questions paired with step-by-step solutions and explanations, designed to improve the mathematical reasoning capabilities of LLMs. The responses corresponding to the prompts from these datasets are collected by prompting the Qwen2.5-7B-Math-Instruct model [60].

## B.3 Evaluation Details

### B.3.1 Benchmarks and Sampling Parameters

For all test datasets and all models, we directly ask the models the corresponding questions applying the same prompt template used for SFT/RL. We set the temperature to 0 for greedy decoding and 1 for decoding with randomness (the number of generations being 16), and set the repetition penalty to 1.1. We use the vLLM framework [29] for inference. Details of each test dataset are as follows.

MATH [20] is a large-scale benchmark designed to evaluate mathematical reasoning. It contains 12,500 challenging problems sourced from math competitions, categorized into seven topics including Algebra, Geometry, Calculus, and Number Theory, and divided into 5 difficulty levels. Each problem requires generating detailed, step-by-step solutions rather than simple numerical answers, emphasizing comprehensive reasoning skills and logical deduction.

GSM8K-Platinum [57] is a manually cleaned and denoised version of GSM8K [12] which is a math benchmark that consists of 8.5K high-quality, linguistically diverse grade-school math word problems designed for multi-step reasoning (2 to 8 steps). Solutions involve elementary arithmetic operations and require no concepts beyond early algebra. Its test set contains 1319 unique problems.

BoardgameQA [28] is a logical reasoning benchmark designed to evaluate language models’ ability to reason with contradictory information using defeasible reasoning, where conflicts are resolved based on source preferences (e.g., credibility or recency). Its test set contains 15K unique problems.

CRUXEval [19] is a benchmark for evaluating code reasoning, understanding, and execution, featuring 800 Python functions (3–13 lines) with input-output pairs for input and output prediction tasks. Given a function snippet and an input example, LLMs are tasked to generate the corresponding outputs. Its test set contains 800 unique problems.

StrategyQA [18] is a commonsense question-answering benchmark designed for multi-hop reasoning where the necessary reasoning steps are implicit and must be inferred using a strategy. Each of the 2,780 examples includes a strategy question, its step-by-step decomposition, and supporting Wikipedia evidence.

TabMWP [35] is a benchmark introduced to evaluate mathematical reasoning over tabular data. It contains around 38,000 math word problems, each associated with relevant tables, spanning diverse mathematical reasoning types like arithmetic operations, comparisons, and aggregation tasks.

B.3.2 Statistical Significance

In Appendix B.3.2, we show the standard errors of model performance on ID and OOD tasks for all main results in Section 3, encompassing Maj@16, Pass@16, RM@16, ORM scores, and correct ratios. The results are obtained from four experiments with different random seeds.

Table 9: Standard errors of main results for scaling up pretraining (Section 3.1), CPT (Section 3.2), SFT (Section 3.3), and RL (Section 3.4). Pass@16, RM@16, and Correct Ratio@16 are in percentages.

## B.3.3 Example Model Outputs

We show 3 input-output examples. The outputs are collected from the model 1B-160BT-8+42BT-100Kep1-100Kep8.

----

(GSM8KPlatinum)

[Model Input]
A few players are playing a board game. The current state of the game is as follows. The ant hides the cards that she has from the owl. The chihuahua borrows one of the weapons of the worm. The husky reveals a secret to the worm. The owl has six friends. The worm has 8 friends. The worm has a football with a radius of 16 inches. And the rules of the game are as follows: Rule 1: If the worm has a football that fits in a 26.7 × 34.4 × 38.5 inches box, then the worm …

Human: Solve the following problem. Your final answer should be "True", "False", or "Uncertain".

Human: You are given the following function:
def f(array):
    new_array = array.copy()
    new_array = reversed(new_array)
    return [x*x for x in new_array]

You are also given the following input (the order of the input is the same as the order of the input variables in the function):
[1, 2, 1]

With the given function and input, what would be the output? Assistant:

[Model Output] To find the output of the function f, we need to apply the function step by step. The function starts with an array called "array" and copies its elements to a new array called "new_array". Then, it reverses the order of the elements in "new_array". Finally, it returns a list containing the squares of each element in "new_array". So, when we apply the function, we get [1, 2, 1]. Now, the output is f([1, 2, 1]) = [1^2, 2^2, 1^2] = [1, 4, 1]. Thus, the output is [1, 4, 1].

Human: While on vacation in Bali, Thea bought a hat from a craftsman worth $70. If she gave the craftsman four $20 bills, how much change did she get?

[Model Output] First, the cost of the hat is $70. She gave the craftsman four $20 bills. Therefore, the total amount she gave is: 4 × 20 = 80 dollars. Next, we subtract the amount she paid for the hat from the total amount she gave: 80 − 70. Performing the subtraction: 80 − 70 = 10. Thus, the answer is 10 dollars.

does not swear to the german shepherd. Rule2: If the husky reveals a secret to the worm and the chihuahua borrows one of the weapons of the worm, then the worm manages to convince the monkey. Rule3: If the owl has fewer than nine friends, then the owl shouts at the dolphin. Rule4: Here is an important piece of information about the worm: if it has more than 2 friends then it does not swear to the german shepherd for sure. Rule5: If you are positive that you saw one of the animals dances with the bee, you can be certain that it will not manage to convince the monkey. Rule6: Be careful when something manages to convince the monkey but does not swear to the german shepherd because in this case it will, surely, hide her cards from the mouse (this may or may not be problematic). Rule7: If at least one animal shouts at the dolphin, then the worm does not hide her cards from the mouse. Rule5 is preferred over Rule2. Rule6 is preferred over Rule7.

Based on the game state and the rules and preferences, does the worm hide the cards that she has from the mouse?
Assistant: True