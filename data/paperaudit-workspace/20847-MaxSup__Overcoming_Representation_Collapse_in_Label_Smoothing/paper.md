
## Abstract


MaxSup: Overcoming Representation Collapse in Label Smoothing

Yuxuan Zhou^α,β*, Heng Li^γ,*†, Zhi-Qi Cheng^γ‡, Xudong Yan^γ,†, Yifei Dong^γ, Mario Fritz^β, Margret Keuper^α

^α University of Mannheim
^γ University of Washington
^β CISPA Helmholtz Center for Information Security

*Equal contribution.
†Internship at UW.
‡Corresponding author.

Abstract

Label Smoothing (LS) is widely adopted to reduce overconfidence in neural network predictions and improve generalization. Despite these benefits, recent studies reveal two critical issues with LS. First, LS induces overconfidence in misclassified samples. Second, it compacts feature representations into overly tight clusters, diluting intra-class diversity, although the precise cause of this phenomenon remained elusive. In this paper, we analytically decompose the LS-induced loss, identifying two key terms: (i) a regularization term that dampens overconfidence when correct predictions occur, and (ii) an error-amplification term associated with misclassifications. Empirically, this latter term correlates with increased misclassification confidence and tighter feature clustering, which may contribute to representation collapse alongside other factors. Thus, while the error-amplification term likely plays a role, it is part of a broader set of influences on feature representation characteristics. To address these shortcomings, we propose Max Suppression (MaxSup), which applies uniform regularization to both correct and incorrect predictions by penalizing the top-1 logit rather than the ground-truth logit. Through extensive feature-space analyses, we show that MaxSup helps restore intra-class variation and sharpens inter-class boundaries. Experiments on large-scale image classification and multiple downstream tasks suggest that MaxSup offers a promising alternative to LS with potential robustness benefits in certain settings.^4

^4 https://github.com/ZhouYuxuanYX/Maximum-Suppression-Regularization


## Introduction


1 Introduction

Multi-class classification [19, 26] typically relies on one-hot labels, which implicitly treat different classes as mutually orthogonal. In practice, however, classes often share low-level features [31, 44] or exhibit high-level semantic similarities [3, 24, 42], rendering the one-hot assumption overly restrictive. Such a mismatch can yield over-confident classifiers and ultimately degrade generalization [9].

To moderate overconfidence, Szegedy et al. [34] introduced Label Smoothing (LS), which combines a uniform distribution with the hard ground-truth label, thereby reducing the model’s certainty in the primary class. LS has since become prevalent in image recognition [10, 22, 36, 47] and neural machine translation [1, 6], often boosting accuracy and calibration [23]. Yet subsequent work indicates that LS can overly compress features into tight clusters [15, 28, 41], hindering intra-class variability and transferability [5]. In parallel, Zhu et al. [48] found that LS paradoxically fosters overconfidence in misclassified samples, though the precise mechanism behind this remains uncertain.

In this paper, we reveal that LS’s training objective inherently contains an error amplification term. This term pushes the network to reinforce incorrect predictions with exaggerated certainty, yielding

Cross-Entropy (Feature Space)
Baseline
Label Smoothing: × Reduced intra-class variation
MaxSup: ✓ Preserved intra-class variation

Grad-CAM
African chameleon
Label Smoothing: × Distracted attention
MaxSup: ✓ Precise attention

Figure 1: Comparison of Label Smoothing (LS) and MaxSup. Left: MaxSup mitigates the intra-class compression induced by LS while preserving inter-class separability. Right: Grad-CAM visualizations show that MaxSup more effectively highlights class-discriminative regions than LS.

highly confident misclassifications and further compressing feature clusters (Section 3.1, Table 1). Building on Zhu et al. [48], we characterize “overconfidence” in terms of the model’s top-1 prediction, rather than through conventional calibration metrics. Through our analysis, we further show that punishing the ground-truth logit during misclassification reduces intra-class variation (Table 2), a phenomenon corroborated by Grad-CAM visualizations (Figure 2).

To overcome these shortcomings, we introduce Max Suppression (MaxSup), a method that retains the beneficial regularization effect of LS while eliminating its error amplification. Rather than penalizing the ground-truth logit, MaxSup focuses on the model’s top-1 logit, ensuring a consistent regularization signal regardless of whether the current prediction is correct or misclassified. By preserving the ground-truth logit in misclassifications, MaxSup sustains richer intra-class variability and sharpens inter-class boundaries. As visualized in Figure 1, this approach mitigates the feature collapse and attention drift often induced by LS, ultimately leading to more robust representations. Through comprehensive experiments in both image classification (Section 4.2) and semantic segmentation (Section 4.3), we show that MaxSup not only alleviates severe intra-class collapse but also consistently boosts top-1 accuracy and robustly enhances downstream transfer performance (Section 4.1).

Our contributions are summarized as follows:
• We perform a logit-level analysis of Label Smoothing, revealing how the error amplification term inflates misclassification confidence and compresses features.
• We propose Max Suppression (MaxSup), removing detrimental error amplification while preserving LS’s beneficial regularization. As shown in extensive ablations, MaxSup alleviates intra-class collapse and yields consistent accuracy gains.
• We demonstrate superior performance across tasks and architectures, including ResNet, MobileNetV2, and DeiT-S, where MaxSup significantly boosts accuracy on ImageNet and consistently delivers stronger representations for downstream tasks such as semantic segmentation and robust transfer learning.


## Related Work


2 Related Work

We first outline mainstream regularization techniques in deep learning, then survey recent advances in Label Smoothing (LS), and finally clarify how our MaxSup diverges from prior variants.

2.1 Regularization

Regularization techniques aim to improve the generalization of deep neural networks by constraining model complexity. Classical methods like ℓ2 [18] and ℓ1 [49] impose direct penalties on large or sparse weights, while Dropout [32] randomly deactivates neurons to discourage over-adaptation. In the realm of loss-based strategies, Label Smoothing (LS) [34] redistributes a fraction of the label probability mass away from the ground-truth class, thereby improving accuracy and calibration [23]. Variants such as Online Label Smoothing (OLS) [45] and Zipf Label Smoothing (Zipf-LS) [21] refine LS by dynamically adjusting the smoothed labels based on a model’s evolving predictions. However, they do not fully address the fundamental issue that emerges when the ground-truth logit is not the highest one (see Section 3.1, Table 1). Other loss-based regularizers focus on alternative aspects of the predictive distribution. Confidence Penalty [25] penalizes the model’s confidence directly, while Logit Penalty [4] minimizes the global ℓ2-norm of logits, a technique reported to enhance class separation [15]. Despite these benefits, Logit Penalty can inadvertently shrink intra-class variation,


![Figure 7](image_7.jpg)
<!-- Caption: Related Work, index 7 -->



![Figure 8](image_8.jpg)
<!-- Caption: Related Work, index 8 -->


thereby hampering transfer learning (see Section 4.1). Unlike the aforementioned methods, MaxSup enforces regularization by penalizing only the top-1 logit $z_{max}$ rather than the ground-truth logit $z_{gt}$. In LS-based approaches, suppressing $z_{gt}$ for misclassified samples can worsen errors, whereas MaxSup applies a uniform penalty regardless of whether the model’s prediction is correct. Consequently, MaxSup avoids the error amplification effect, retains richer intra-class diversity (see Table 2), and achieves robust transfer performance across diverse datasets and model families (see Table 3).

## 2.2 Studies on Label Smoothing

Label Smoothing has also been studied extensively under knowledge distillation. For instance, Yuan et al. [43] observed that LS can approximate the effect of a teacher–student framework, while Shen et al. [30] investigated its role in such pipelines more systematically. Additionally, Chandrasegaran et al. [2] demonstrated that a low-temperature, LS-trained teacher can notably improve distillation outcomes. Concurrently, Kornblith et al. [15] showed that LS tightens intra-class clusters in the feature space, diminishing transfer performance. From a Neural Collapse perspective [46, 8], LS nudges the model toward rigid feature clusters, as evidenced by the reduced feature variability measured in Xu and Liu [41]. Our goal is to overcome LS’s inherent error amplification effect. Rather than adjusting how the smoothed label distribution is constructed (as in OLS or Zipf-LS), MaxSup directly penalizes the highest logit $z_{max}$. This design ensures consistent regularization even if $z_{gt}$ is not the top logit, thereby avoiding the degradation in performance typical of misclassified samples under LS (see Section 3.2). Moreover, MaxSup integrates seamlessly into standard training pipelines, introducing negligible computational overhead beyond substituting the LS term.


## Method


## 3 Max Suppression Regularization (MaxSup)

We first partition the training objective into two components: the standard Cross-Entropy (CE) loss and a regularization term introduced by Label Smoothing (LS). By expressing LS in terms of logits (Theorem 3.3), we isolate two key factors: a regularization term that controls overconfidence and an error amplification term that enlarges the gap between the ground-truth logit $z_{gt}$ and any higher logits (Theorem 3.4, Equation (5)), ultimately degrading performance. To address these issues, we propose Max Suppression Regularization (MaxSup), which applies the penalty to the largest logit $z_{max}$ rather than $z_{gt}$ (Equation (8), Section 3.2). This shift delivers consistent regularization for both correct and incorrect predictions, preserves intra-class variation, and bolsters inter-class separability. Consequently, MaxSup mitigates the representation collapse found in LS, attains superior ImageNet-1K accuracy (Table 1), and improves transferability (Table 2, Table 3). The following sections elaborate on MaxSup’s formulation and integration into the training pipeline.

## 3.1 Revisiting Label Smoothing

Label Smoothing (LS) is a regularization technique designed to reduce overconfidence by softening the target distribution. Rather than assigning probability 1 to the ground-truth class and 0 to all others, LS redistributes a fraction $\alpha$ of the probability uniformly across all classes:

**Definition 3.1.** For a standard classification task with $K$ classes, Label Smoothing (LS) converts a one-hot label $\mathbf{y} \in \mathbb{R}^K$ into a softened target label $\mathbf{s} \in \mathbb{R}^K$:
$$s_k = (1 - \alpha) y_k + \frac{\alpha}{K},$$
(1)
where $y_k = \mathbb{1}_{\{k = gt\}}$ denotes the ground-truth class. The smoothing factor $\alpha \in [0, 1]$ reduces the confidence assigned to the ground-truth class and distributes $\frac{\alpha}{K}$ to other classes uniformly, thereby mitigating overfitting, enhancing robustness, and promoting better generalization.

To clarify the effect of LS on model training, we first decompose the Cross-Entropy (CE) loss into a standard CE term and an additional LS-induced regularization term:

**Lemma 3.2.** Decomposition of Cross-Entropy Loss with Soft Labels.
$$H(\mathbf{s}, \mathbf{q}) = H(\mathbf{y}, \mathbf{q}) + L_{LS},$$
(2)
where
$$L_{LS} = \alpha \left( H\left(\frac{1}{K}, \mathbf{q}\right) - H(\mathbf{y}, \mathbf{q}) \right).$$
(3)

Where q is the predicted probability vector, H(·) denotes the Cross-Entropy, and 1/K is the uniform distribution introduced by LS. This shows that LS adds a regularization term, LLS, which smooths the output distribution and helps to reduce overfitting. (See Section A for a formal proof.)

Building on Theorem 3.2, we next explicitly express LLS at the logit level for further analysis.

Theorem 3.3. Logit-Level Formulation of Label Smoothing Loss.

While a direct closed-form expression is complex due to the softmax normalization, the effective Label Smoothing loss can be approximated by:

LLS ≈ α ( zgt − (1/K) ∑_{k=1}^K zk ) − log \sum_{j=1}^K e^{z_j}

Here, zgt is the logit of the ground-truth class, and the log-sum-exp term accounts for normalization. This formulation emphasizes that the smoothing penalty involves both the difference between zgt and the average logit and the normalization constant, reflecting a trade-off in overconfidence reduction. (See supplementary Section B for a detailed derivation.)

The behavior of LLS differs depending on whether zgt is already the maximum logit. Specifically, depending on whether the prediction is correct (zgt = zmax) or incorrect (zgt ≠ zmax), we can decompose LLS into two parts:

Corollary 3.4. Decomposition of Label Smoothing Loss.

LLS = (α/K) ∑_{zm<zgt} (zgt − zm) + (α/K) ∑_{zn>zgt} (zgt − zn), (5)

Regularization Error amplification

where M and N are the numbers of logits below and above zgt, respectively (M + N = K − 1). Note that the error amplification term vanishes when zgt = zmax.

1. Regularization: Penalizes the gap between zgt and any smaller logits, thereby moderating overconfidence.
2. Error amplification: Penalizes the gap between zgt and larger logits, inadvertently increasing overconfidence in incorrect predictions.

Although LS aims to combat overfitting by reducing prediction confidence, its error amplification component can be detrimental for misclassified samples, as it widens the gap between the ground-truth logit zgt and the incorrect top logit. Concretely:

1. Correct Predictions (zgt = zmax): The error amplification term is zero, and the regularization term effectively reduces overconfidence by shrinking the gap between zgt and any smaller logits.
2. Incorrect Predictions (zgt ≠ zmax): LS introduces two potential issues:
- Error amplification: Increases the gap between zgt and larger logits, reinforcing overconfidence in incorrect predictions.
- Inconsistent regularization: The regularization term lowers zgt yet does not penalize zmax, which further impairs learning.

These issues with LS on misclassified samples have also been systematically observed in prior work [39]. By precisely disentangling these two components (regularization vs. error amplification), we can design a more targeted and effective solution.

Ablation Study on LS Components. To isolate the effects of each component in LS, we carefully perform a detailed and systematic ablation study on ImageNet-1K using a DeiT-Small model [36] without Mixup or CutMix. As indicated in Table 1, the performance gains from LS stem solely from the regularization term, whereas the error amplification term degrades accuracy. In contrast, our MaxSup omits the error amplification component and leverages only the beneficial regularization, thereby boosting accuracy beyond that of standard LS. Specifically, Table 1 shows that LS’s overall improvement can be attributed exclusively to its regularization contribution; the error amplification term consistently reduces accuracy (e.g., to 73.63% or 73.69%). Disabling only the error amplification while retaining the regularization yields a slight but measurable improvement (75.98% vs. 75.91%). By fully removing error amplification and faithfully preserving the helpful aspects of LS, our MaxSup achieves 76.12% accuracy, clearly and consistently outperforming LS. This result underscores that MaxSup directly tackles LS’s fundamental shortcoming by maintaining a consistent and meaningful regularization signal—even when the top-1 prediction is incorrect.

Table 1: Ablation on LS components using DeiT-Small on ImageNet-1K (without CutMix or Mixup). “Regularization” denotes penalizing logits smaller than z_{gt}; “error amplification” penalizes logits larger than z_{gt}. MaxSup removes error amplification while retaining regularization.

| Method                | Formulation                                                                                                             | Accuracy |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------- | -------- |
| Baseline              | –                                                                                                                       | 74.21    |
| + Label Smoothing     | $\frac{\alpha}{K} \sum_{z_m < z_{gt}} (z_{gt} - z_m) + \frac{\alpha}{K} \sum_{z_n > z_{gt}} (z_{gt} - z_n)$            | 75.91    |
| + Regularization      | $\frac{\alpha}{M} \sum_{z_m < z_{gt}} (z_{gt} - z_m)$                                                                   | 75.98    |
| + error amplification | $\frac{\alpha}{N} \sum_{z_n > z_{gt}} (z_{gt} - z_n)$                                                                   | 73.63    |
| + error amplification | $\alpha (z_{gt} - z_{max})$                                                                                             | 73.69    |
| + MaxSup              | $\alpha \left( z_{max} - \frac{1}{K} \sum_{k=1}^{K} z_k \right)$                                                        | 76.12    |

3.2 Max Suppression Regularization

Building on our analysis in Section 3.1, we find that Label Smoothing (LS) not only impacts correctly classified samples but also influences misclassifications in unintended and harmful ways. Specifically, LS suffers from two main limitations: inconsistent regularization and error amplification. As illustrated in Table 1, LS penalizes the ground-truth logit $z_{gt}$ even in misclassified examples, needlessly widening the gap between $z_{gt}$ and the erroneous top-1 logit. To resolve these critical shortcomings, we propose Max Suppression Regularization (MaxSup), which explicitly penalizes the largest logit $z_{max}$ rather than $z_{gt}$. This key design choice ensures uniform regularization across both correct and misclassified samples, effectively eliminating the error-amplification issue in LS (Table 1) and preserving the ground-truth logit’s integrity for more stable, robust learning.

Definition 3.5. Max Suppression Regularization

We define the Cross-Entropy loss with MaxSup as follows:

$$\underbrace{H(\mathbf{s}, \mathbf{q})}_{\text{CE with Soft Labels}} = \underbrace{H(\mathbf{y}, \mathbf{q})}_{\text{CE with Hard Labels}} + \underbrace{L_{\text{MaxSup}}}_{\text{Max Suppression Loss}} , \tag{6}$$

where

$$L_{\text{MaxSup}} = \alpha \left( H\!\left(\frac{1}{K}, \mathbf{q}\right) - H(\mathbf{y}', \mathbf{q})\right), \tag{7}$$

and

$$y'_k = \mathbb{1}_{\{k=\arg\max(\mathbf{q})\}},$$

so that $y'_k = 1$ identifies the model’s top-1 prediction and $y'_k = 0$ otherwise. Here, $H\!\left(\frac{1}{K}, \mathbf{q}\right)$ encourages a uniform output distribution to mitigate overconfidence, while $H(\mathbf{y}', \mathbf{q})$ penalizes the current top-1 logit. By shifting the penalty from $z_{gt}$ (the ground-truth logit) to $z_{max}$ (the highest logit), MaxSup avoids unduly suppressing $z_{gt}$ when the model misclassifies, thus overcoming Label Smoothing’s principal shortcoming.

Logit-Level Formulation of MaxSup

Building on the logit-level perspective introduced for LS in Section 3.1, we can express $L_{\text{MaxSup}}$ as:

$$L_{\text{MaxSup}} = \alpha \left( z_{max} - \frac{1}{K} \sum_{k=1}^{K} z_k \right), \tag{8}$$

where $z_{max} = \max_k\{z_k\}$ is the largest (top-1) logit, and $\frac{1}{K} \sum_{k=1}^{K} z_k$ is the mean logit. Unlike LS, which penalizes the ground-truth logit $z_{gt}$ and may worsen errors in misclassified samples, MaxSup shifts the highest logit uniformly, thus providing consistent regularization for both correct and incorrect predictions. As shown in Table 1, this approach eliminates LS’s error-amplification issue while preserving the intended overconfidence suppression.

Comparison with Label Smoothing

MaxSup fundamentally differs from LS in handling correct and incorrect predictions. When $z_{gt} = z_{max}$, both LS and MaxSup similarly reduce overconfidence. However, when $z_{gt} \neq z_{max}$, LS shrinks $z_{gt}$, widening the gap with the incorrect logit, whereas MaxSup penalizes the incorrect maximum logit directly, narrowing the gap and preventing error amplification.

Gradient Analysis.
Taking the gradient of the MaxSup loss in Equation (8), which involves both the max logit and the mean of all logits, yields:
$$
\frac{\partial L_{\text{MaxSup}}}{\partial z_k} = \alpha \left( \mathbf{1}_{k = \arg\max(\mathbf{q})} - \frac{1}{K} \right).
$$

Therefore, the top-1 logit receives a positive gradient scaled by α(1 - 1/K), pushing it down, while all other logits receive a negative gradient of magnitude α/K, pushing them slightly up. This balanced redistribution reflects coupled interactions among logits due to the mean term, better capturing the softmax normalization effects. Such an explicit formulation clarifies optimization dynamics and justifies MaxSup’s improved convergence behavior compared to LS. While [39] conducted a related gradient analysis of the training loss, it focuses specifically on the setting of selective classification, and examines a post hoc logit normalization technique to mitigate confidence calibration issues. However, this approach addresses only the overconfidence problem of label smoothing (LS), without tackling representation collapse. Moreover, our work presents a logit-level reformulation of LS that provides a deeper theoretical understanding of why LS amplifies errors.

MaxSup penalizes z_{max}, preserving z_{gt} from undue suppression. As illustrated in Figure 2, this helps the model recover from mistakes more effectively and avoid reinforcing incorrect predictions.

Behavior Across Different Samples.
MaxSup applies a dynamic penalty based on the model’s current predictions. For high-confidence, correctly classified examples, it behaves similarly to LS by reducing overconfidence, effectively mitigating overfitting. In contrast, for misclassified or uncertain samples, MaxSup aggressively suppresses the incorrect top-1 logit, further safeguarding the ground-truth logit z_{gt}. This selective strategy preserves a faithful and reliable representation of the true class while actively discouraging error propagation. As shown in Section 4.2 and Table 5, this promotes more robust decision boundaries and leads to stronger generalization.

Theoretical Insights and Practical Benefits.
MaxSup provides both theoretical and practical advantages over LS. Whereas LS applies a uniform penalty to the ground-truth logit regardless of correctness, MaxSup penalizes only the most confident logit z_{max}. This dynamic adjustment robustly prevents error accumulation in misclassifications, ensuring more stable convergence. As a result, MaxSup generalizes better and achieves strong performance on challenging datasets. Moreover, as shown in Section 4.1, MaxSup preserves greater intra-class diversity, substantially improving transfer learning (Table 3) and yielding more interpretable activation maps (Figure 2).


## Experiments


4 Experiments
We begin by examining how MaxSup improves feature representations, then evaluate it on large-scale image classification and semantic segmentation tasks. Finally, we visualize class activation maps to illustrate the practical benefits of MaxSup.

4.1 Analysis of MaxSup’s Learning Benefits
Having established how MaxSup addresses Label Smoothing’s (LS) principal shortcomings (Section 3.1), we now demonstrate its impact on inter-class separability and intra-class variation—two properties essential for accurate classification and effective transfer learning.

4.1.1 Intra-Class Variation and Transferability
As noted in Section 3.1, Label Smoothing (LS) primarily curbs overconfidence for correctly classified samples but inadvertently triggers error amplification in misclassifications. This uneven penalization can overly compress intra-class feature representations. By contrast, MaxSup uniformly penalizes the top-1 logit, whether the prediction is correct or incorrect, thereby eliminating LS’s error-amplification effect and preserving finer distinctions within each class.

| Method        | $\bar{d}_{\text{within}}$ ↑<br/>Train | $\bar{d}_{\text{within}}$ ↑<br/>Val | $R^2$ ↑<br/>Train | $R^2$ ↑<br/>Val |
| ------------- | -------------------------------------- | ------------------------------------ | ----------------- | --------------- |
| Baseline      | 0.311                                  | 0.331                                | 0.403             | 0.445           |
| LS            | 0.263                                  | 0.254                                | 0.469             | 0.461           |
| OLS           | 0.271                                  | 0.282                                | 0.594             | 0.571           |
| Zipf-LS       | 0.261                                  | 0.293                                | 0.552             | 0.479           |
| MaxSup        | 0.293                                  | 0.300                                | 0.519             | 0.497           |
| Logit Penalty | 0.284                                  | 0.314                                | 0.645             | 0.602           |

Table 2 compares intra-class variation ($\bar{d}_{\text{within}}$) and inter-class separability ($R^2$) [15] for ResNet-50 trained on ImageNet-1K. Although all regularizers typically decrease $\bar{d}_{\text{within}}$ compared to the baseline, Logit Penalty occasionally displays increased $\bar{d}_{\text{within}}$ and $R^2$ values, inconsistent with prior results. This anomaly likely results from unstable estimation due to limited sample sizes or variation in regularization strength. Therefore, its comparative effectiveness in preserving intra-class diversity merits cautious interpretation. Notably, MaxSup maintains $\bar{d}_{\text{within}}$ reductions similar to other methods, suggesting only marginal benefits in preserving intra-class statistics.

These benefits are further underscored by the linear-probe transfer accuracy on CIFAR-10 (Table 3). While LS and Logit Penalty each boost ImageNet accuracy, both degrade transfer accuracy, likely by suppressing informative and transferable features. By contrast, MaxSup preserves near-baseline performance, implying that it maintains rich discriminative information crucial for downstream tasks. For extended evaluations on diverse datasets, see Table 12 in the appendix.

## 4.1.2 Connection to Logit Penalty

As detailed in Section 3, both Label Smoothing (LS) variants and MaxSup impose penalties directly at the logit level, aligning with the perspective that various regularizers influence a model's representational capacity via distinct logit constraints [15]. Within this family of techniques, Logit Penalty and MaxSup both address the maximum logit, yet diverge fundamentally in their specific methods of regularization.

Logit Penalty minimizes the $\ell_2$-norm of the entire logit vector, inducing a global contraction that can improve class separation but also reduce intra-class diversity, potentially hindering downstream transfer. By contrast, MaxSup focuses exclusively on the top-1 logit, gently nudging it closer to the mean logit. Because only the highest-confidence prediction is penalized, MaxSup avoids the uniform shrinkage observed in Logit Penalty, preserving richer intra-class variation—a property essential for robust transfer. Further insights into this behavior can be found in Section L, where logit-value histograms illustrate how each method affects the logit distribution.

Table 3: Linear-probe transfer accuracy on CIFAR-10 (higher is better).

| Method          | Acc.  |
| --------------- | ----- |
| Baseline        | 0.814 |
| Label Smoothing | 0.746 |
| Logit Penalty   | 0.724 |
| MaxSup          | 0.810 |

## 4.2 Evaluation on ImageNet Classification

Next, we compare MaxSup to standard Label Smoothing (LS) and various LS extensions on the large-scale ImageNet-1K dataset.

### 4.2.1 Experiment Setup

Model Training Configurations. We evaluate convolutional (ResNet [10], MobileNetV2 [27]) and transformer (DeiT-Small [36]) architectures on ImageNet [17]. All models are trained with SGD, momentum 0.9, and weight decay of $1 \times 10^{-4}$, batch size 2048. Initial learning rates and decay schedules vary by model.

For both MaxSup and baselines (including LS, Zipf-LS, OLS), hyperparameters and training settings are re-tuned under each training recipe to ensure optimal performance for each method. Specifically, learning rates, smoothing parameters, and augmentation policies (CutMix, Mixup) are adjusted separately for all methods when evaluating robustness under the shorter training schedule (Appendix E). This re-tuning guarantees that reported performance differences reflect method improvements rather than hyperparameter bias.

All comparisons employ identical hardware and software stacks to control confounding factors, enhancing reproducibility and fairness across experiments.

### 4.2.2 Experiment Results

ConvNet Comparison. Table 4 shows results for MaxSup alongside various label-smoothing and self-distillation methods on both ImageNet and CIFAR-100 benchmarks. Across all convolutional

<sup>5</sup> Additional training hyperparameters follow the FFCV scripts at https://github.com/libffcv/ffcv. See Section E for further details.

architectures tested, MaxSup consistently delivers the highest top-1 accuracy among label-smoothing approaches. By contrast, OLS [45] and Zipf-LS [21] exhibit less stable gains, suggesting their effectiveness may heavily hinge on specific training protocols.

To reproduce OLS and Zipf-LS, we apply the authors' official codebases and hyperparameters but do not replicate their complete training recipes (e.g., OLS trains for 250 epochs with a step-scheduled learning rate of 0.1, and Zipf-LS uses 100 epochs with distinct hyperparameters). Even under these modified settings, MaxSup remains robust, highlighting its effectiveness across a variety of training schedules—unlike the more schedule-sensitive improvements noted for OLS and Zipf-LS.

Table 4: Performance comparison of classical convolutional networks on ImageNet and CIFAR-100. All results are shown as "mean ± std" (percentage). Bold highlights the best performance; underlined marks the second best. (Methods labeled with * indicate code adapted from official repositories; see the text for additional details.)

| Method          | ImageNet       |                |                |                | CIFAR-100      |                |                |                |
| --------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
|                 | ResNet-18      | ResNet-50      | ResNet-101     | MobileNetV2    | ResNet-18      | ResNet-50      | ResNet-101     | MobileNetV2    |
| Baseline        | 69.09±0.12     | 76.41±0.10     | 75.96±0.18     | 71.40±0.12     | 76.16±0.18     | 78.69±0.16     | 79.11±0.21     | 68.06±0.06     |
| Label Smoothing | 69.54±0.15     | 76.91±0.11     | 77.37±0.15     | 71.61±0.09     | 77.05±0.17     | 78.88±0.13     | 79.19±0.25     | 69.65±0.08     |
| Zipf-LS*        | 69.31±0.12     | 76.73±0.17     | 76.91±0.11     | 71.16±0.15     | 76.21±0.12     | 78.75±0.21     | 79.15±0.18     | 69.39±0.08     |
| OLS*            | 69.45±0.15     | 77.23±0.21     | 77.71±0.17     | 71.63±0.11     | 77.33±0.15     | 78.79±0.12     | 79.25±0.15     | 68.91±0.11     |
| MaxSup          | 69.96±0.13     | 77.69±0.07     | 78.18±0.12     | 72.08±0.17     | 77.82±0.15     | 79.15±0.13     | 79.41±0.19     | 69.88±0.07     |
| Logit Penalty   | 68.48±0.10     | 76.73±0.10     | 77.20±0.15     | 71.13±0.10     | 76.41±0.15     | 78.90±0.16     | 78.89±0.21     | 69.46±0.08     |

DeiT Comparison. Table 5 summarizes performance for DeiT-Small on ImageNet across various regularization strategies. Notably, MaxSup attains a top-1 accuracy of 76.49%, surpassing standard Label Smoothing by 0.41%. In contrast, LS variants such as Zipf-LS and OLS offer only minor gains over LS, implying that their heavy reliance on data augmentation may limit their applicability to vision transformers. By outperforming both LS and its variants without additional data manipulations, MaxSup demonstrates robust feature enhancement. These findings underscore MaxSup’s adaptability to different architectures and emphasize its utility in scenarios where conventional label-smoothing methods yield limited benefits.

Table 5: DeiT-Small top-1 accuracy (%), reported as mean ± standard deviation. Values in parentheses indicate absolute improvements over the baseline.

| Method          | Mean              | Std        |
| --------------- | ----------------- | ---------- |
| Baseline        | 74.39             | 0.45       |
| Label Smoothing | 76.08 (+1.69)     | 0.38       |
| Zipf-LS         | 75.89 (+1.50)     | 0.52       |
| OLS             | 76.16 (+1.77)     | 0.41       |
| MaxSup          | 76.49 (+2.10)     | 0.47       |

Fine-Grained Classification. Beyond large-scale benchmarks like ImageNet, we further evaluate MaxSup on two fine-grained visual recognition tasks: CUB-200-2011 [37] and Stanford Cars [16]. These datasets pose unique challenges due to subtle inter-class differences, which often expose the limitations of standard regularization approaches. As shown in Table 6, MaxSup achieves the best performance across both datasets, surpassing LS and its recent variants. This demonstrates that MaxSup encourages the model to learn more discriminative and semantically rich representations that better capture fine-grained attributes, such as textures and part-level details. The consistent improvements on these benchmarks further validate MaxSup’s capacity to generalize across different visual domains and its potential to enhance robustness in recognition scenarios where nuanced feature understanding is critical.

Table 6: Classification on CUB and Cars Datasets.

| Method   | CUB[37]  | Cars[16] |
| -------- | --------- | -------- |
| Baseline | 80.88     | 90.27    |
| LS       | 81.96     | 91.64    |
| OLS      | 82.33     | 91.96    |
| Zipf-LS  | 81.40     | 90.99    |
| MaxSup   | 82.53     | 92.25    |

Long-Tailed Classification. We evaluated MaxSup on the CIFAR-10-LT dataset with imbalance ratios of 50 and 100, following the protocol in [35]. Table 7 summarizes the overall accuracy comparisons of Focal Loss, Focal Loss + LS, and Focal Loss + MaxSup. While MaxSup improves overall accuracy relative to baselines, it fails to improve performance on minority low-shot classes, which remain at or near zero accuracy across all methods. This highlights potential fairness concerns and limitations in handling classes with scarce data, emphasizing the need for further work on equitable learning under severe imbalance. Ethical considerations regarding bias amplification remain unaddressed and warrant explicit discussion.

Table 7: Comparison of classification performance (%) across imbalance levels for different loss strategies (Focal Loss vs Label Smoothing (LS) vs MaxSup) on the long-tailed CIFAR-10 dataset using ResNet-32. Best performances are in bold.

| Dataset     | Split | Imbalance Ratio | Method          | Overall  | Many     | Medium   | Low |
| ----------- | ----- | --------------- | --------------- | -------- | -------- | -------- | --- |
| LT CIFAR-10 | val   | 50              | Focal Loss      | 77.4     | 76.0     | 89.7     | 0.0 |
|             |       |                 | Label Smoothing | 81.2     | 81.6     | 77.0     | 0.0 |
|             |       |                 | MaxSup          | **82.1** | **82.5** | 78.1     | 0.0 |
| LT CIFAR-10 | test  | 50              | Focal Loss      | 76.8     | 75.3     | 90.4     | 0.0 |
|             |       |                 | Label Smoothing | 80.5     | 81.1     | 75.4     | 0.0 |
|             |       |                 | MaxSup          | **81.4** | **82.3** | 73.4     | 0.0 |
| LT CIFAR-10 | val   | 100             | Focal Loss      | 75.1     | 71.8     | 88.3     | 0.0 |
|             |       |                 | Label Smoothing | 76.6     | 80.6     | 60.7     | 0.0 |
|             |       |                 | MaxSup          | **77.1** | 80.1     | **65.1** | 0.0 |
| LT CIFAR-10 | test  | 100             | Focal Loss      | 74.7     | 71.6     | 87.2     | 0.0 |
|             |       |                 | Label Smoothing | **76.4** | **80.8** | 59.0     | 0.0 |
|             |       |                 | MaxSup          | **76.4** | 79.9     | **62.4** | 0.0 |

0.9 percentage points. These results indicate that MaxSup achieves a better trade-off between many- and medium-shot accuracy. While it does not fully resolve the challenge of imbalanced classification—especially for minority classes—it shows positive effects and offers a promising direction for further extension.

Corrupted Image Classification

To evaluate the effectiveness of MaxSup on out-of-distribution (OOD) settings, we conducted experiments on the CIFAR-10-C benchmark [12]. While previous works have explored label smoothing in corruptions [38], standard protocols vary. Here, we follow the commonly adopted experimental setup based on Hendrycks and Dietterich [12]. Table 8 reports MaxSup’s performance compared to Label Smoothing (LS) using ResNet-50 as backbone. LS attains slightly lower NLL, indicating more confident predictions, whereas MaxSup achieves better Expected Calibration Error (ECE), suggesting improved confidence alignment. Results demonstrate that MaxSup offers a promising trade-off in OOD robustness consistent with prior literature.

Table 8: Comparison of MaxSup, Label Smoothing (LS), and standard Cross Entropy (CE) on CIFAR-10-C. Lower is better. Values show mean (std) across three setups.

| Metric       | MaxSup           | LS               | CE               |
| ------------ | ---------------- | ---------------- | ---------------- |
| Error (Corr) | 0.362(0.055)     | 0.359(0.064)     | **0.354(0.015)** |
| NLL (Corr)   | 1.770(0.103)     | **1.476(0.111)** | 1.819(0.158)     |
| ECE (Corr)   | **0.145(0.003)** | 0.158(0.015)     | 0.260(0.015)     |

Ablation on the Weight Schedule.

We also systematically investigate how different α scheduling strategies impact MaxSup’s performance. Empirical results indicate that MaxSup consistently maintains high accuracy across a wide range of schedules, further underscoring its robustness against hyperparameter changes. For additional details and discussions, refer to Section F.

4.3 Evaluation on Semantic Segmentation

We further investigate MaxSup’s applicability to downstream tasks by evaluating its performance on semantic segmentation using the widely adopted MMSegmentation framework [6]. Specifically, we adopt the UperNet [40] architecture with a DeiT-Small backbone, trained on ADE20K. Models pretrained on ImageNet-1K with either MaxSup or Label Smoothing are then fine-tuned under the same cross-entropy objective (Section 4.2.2).

Table 9: Semantic segmentation (multi-scale) on ADE20K using UperNet. All models are pretrained on ImageNet-1K; mIoU reported as percentage.

| Backbone   | Method          | mIoU            |
| ---------- | --------------- | --------------- |
| DeiT-Small | Baseline        | 42.1            |
|            | Label Smoothing | 42.4 (+0.3)     |
|            | MaxSup          | **42.8 (+0.7)** |

Table 9 shows that initializing with MaxSup-pretrained weights yields an mIoU of 42.8%, surpassing the 42.4% achieved by Label Smoothing. This improvement indicates that MaxSup fosters more

6 https://github.com/open-mmlab/mmsegmentation

discriminative feature representations conducive to dense prediction tasks. By more effectively capturing class boundaries and within-class variability, MaxSup promotes stronger segmentation results, underscoring its potential to deliver features that are both transferable and highly robust.

4.4 Visualization via Class Activation Maps

To better understand how MaxSup fundamentally differs from Label Smoothing (LS) in guiding model decisions, we employ Gradient-weighted Class Activation Mapping (Grad-CAM) [29], which highlights regions most influential for each prediction.

We evaluate DeiT-Small under three training setups: MaxSup (second row), LS (third row), and a baseline with standard cross-entropy (fourth row). As illustrated in Figure 2, MaxSup-trained models more effectively suppress background distractions than LS, which often fixates on unrelated objects—such as poles in “Bird,” tubes in “Goldfish,” and caps in “House Finch.” This behavior reflects LS’s error-enhancement mechanism, which can misdirect attention.

Moreover, MaxSup retains a wider spectrum of salient features, as exemplified in the “Shark” and “Monkey” images, where LS-trained models often omit crucial semantic details (e.g., fins, tails, or facial contours). These findings align with our analysis in Section 1, clearly demonstrating that MaxSup preserves richer intra-class information. Consequently, MaxSup-trained models produce more accurate and consistent predictions by effectively leveraging fine-grained object cues. Further quantitative Grad-CAM overlay metrics (e.g., precision and recall for target regions) confirm that MaxSup yields more focused and comprehensive activation maps, further underscoring its overall efficacy.

Figure 2: Grad-CAM [29] visualizations for DeiT-Small models under three training setups: MaxSup (2nd row), Label Smoothing (3rd row), and a baseline (4th row). The first row shows the original images. Compared to Label Smoothing, MaxSup more effectively filters out non-target regions and highlights essential features of the target class, reducing instances where the model partially or entirely focuses on irrelevant areas.


## Conclusion


5 Conclusion

We examined several shortcomings of Label Smoothing (LS) and proposed Max Suppression Regularization (MaxSup) to partially mitigate its error amplification effects. Our theoretical and empirical analyses indicate that MaxSup reduces overconfidence in misclassified samples more consistently than LS in many settings. While experimental results show improved accuracy and better intra-class variation preservation on standard benchmarks, some scenarios—such as severe class imbalance and out-of-distribution evaluation—highlight residual challenges and limitations. Therefore, MaxSup represents a promising step forward but does not universally resolve all issues inherent in label smoothing schemes. Future work should investigate adaptive or complementary strategies to further enhance generalization and robustness.

Limitations. Prior work [23] notes that LS-trained teachers may degrade knowledge distillation [13, 14], and Guo et al. [8] suggests LS accelerates convergence via improved conditioning. Examining MaxSup’s potential role in distillation and its overall impact on training dynamics would clarify these underlying effects. Recent studies [33, 7] also show that ℓ2 regularization biases final-layer features toward low-rank solutions, raising interesting questions about whether MaxSup behaves similarly.

Impact. In practical applications, MaxSup shows strong promise for systems demanding robust generalization and efficient transfer, and we have not observed any additional adverse effects or trade-offs. By offering researchers and practitioners both a clearer understanding of LS’s limitations and a straightforward, computationally light, and easily integrable method to overcome them, MaxSup may help guide the development of more reliable and interpretable deep learning models.


![Figure 46](image_46.jpg)
<!-- Caption: Conclusion, index 46 -->



## Appendix


## Acknowledgments

This work was supported by the University of Washington Faculty Startup Fund, the Carwein–Andrews Fellowship, the UW GSFEI Top Scholar Award, and the U.S. DOT PacTrans subcenter seed funding program. We also thank the anonymous reviewers for their constructive feedback and insightful suggestions, which helped improve the quality and clarity of this paper.


## References


## References

[1] Duarte M. Alves, Nuno M. Guerreiro, João Alves, José Pombal, Ricardo Rei, José G. C. de Souza, Pierre Colombo, and André F. T. Martins. Steering large language models for machine translation with fine-tuning and in-context learning. arXiv preprint arXiv:2310.13448, 2023.

[2] Keshigeyan Chandrasegaran, Ngoc-Trung Tran, Yunqing Zhao, and Ngai-Man Cheung. Revisiting label smoothing and knowledge distillation compatibility: What was missing? In Proceedings of the International Conference on Machine Learning (ICML), pages 2890–2916. PMLR, 2022.

[3] Shiming Chen, Guosen Xie, Yang Liu, Qinmu Peng, Baigui Sun, Hao Li, Xinge You, and Ling Shao. HSVA: Hierarchical semantic-visual adaptation for zero-shot learning. Advances in Neural Information Processing Systems (NeurIPS), 34:16622–16634, 2021.

[4] Yann Dauphin and Ekin Dogus Cubuk. Deconstructing the regularization of BatchNorm. In International Conference on Learning Representations (ICLR), 2021.

[5] Yutong Feng, Jianwen Jiang, Mingqian Tang, Rong Jin, and Yue Gao. Rethinking supervised pre-training for better downstream transfer. arXiv preprint arXiv:2110.06014, 2021.

[6] Yingbo Gao, Weiyue Wang, Christian Herold, Zijian Yang, and Hermann Ney. Towards a better understanding of label smoothing in neural machine translation. In Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing, pages 212–223, 2020.

[7] Connall Garrod and Jonathan P. Keating. The persistence of neural collapse despite low-rank bias: An analytic perspective through unconstrained features. arXiv preprint arXiv:2410.23169, 2024.

[8] Li Guo, Keith Ross, Zifan Zhao, George Andriopoulos, Shuyang Ling, Yufeng Xu, and Zixuan Dong. Cross-entropy versus label smoothing: A neural collapse perspective. arXiv preprint arXiv:2402.03979, 2024.

[9] Qiushan Guo, Xinjiang Wang, Yichao Wu, Zhipeng Yu, Ding Liang, Xiaolin Hu, and Ping Luo. Online knowledge distillation via collaborative learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11020–11029, 2020.

[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016.

[11] Markus Heinonen, Ba-Hien Tran, Michael Kampffmeyer, and Maurizio Filippone. Robust classification by coupling data mollification with label smoothing. In International Conference on Artificial Intelligence and Statistics, pages 4960–4968. PMLR, 2025.

[12] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In International Conference on Learning Representations (ICLR), 2019.

[13] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

[14] Xinting Hu, Kaihua Tang, Chunyan Miao, Xian-Sheng Hua, and Hanwang Zhang. Distilling Causal Effect of Data in Class-Incremental Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[15] Simon Kornblith, Ting Chen, Honglak Lee, and Mohammad Norouzi. Why Do Better Loss Functions Lead to Less Transferable Features? Advances in Neural Information Processing Systems (NeurIPS), 34:28648–28662, 2021.

[16] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3D Object Representations for Fine-Grained Categorization. In Proceedings of the IEEE International Conference on Computer Vision Workshops, pages 554–561, 2013.

[17] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. ImageNet Classification with Deep Convolutional Neural Networks. Advances in Neural Information Processing Systems, 25, 2012.

[18] Anders Krogh and John Hertz. A Simple Weight Decay Can Improve Generalization. Advances in Neural Information Processing Systems, 4, 1991.

[19] Yann LeCun. The MNIST database of handwritten digits. http://yann.lecun.com/exdb/mnist/, 1998.

[20] Dongkyu Lee, Ka Chun Cheung, and Nevin L. Zhang. Adaptive Label Smoothing with Self-Knowledge in Natural Language Generation. arXiv preprint arXiv:2210.13459, 2022.

[21] Jiajun Liang, Linze Li, Zhaodong Bing, Borui Zhao, Yao Tang, Bo Li, and Haoqiang Fan. Efficient One Pass Self-Distillation with Zipf’s Label Smoothing. In European Conference on Computer Vision (ECCV), pages 104–119. Springer, 2022.

[22] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 10012–10022, 2021.

[23] Rafael Müller, Simon Kornblith, and Geoffrey E. Hinton. When Does Label Smoothing Help? Advances in Neural Information Processing Systems (NeurIPS), 32, 2019.

[24] Zachary Novack, Julian McAuley, Zachary Chase Lipton, and Saurabh Garg. CHILS: Zero-Shot Image Classification with Hierarchical Label Sets. In Proceedings of the International Conference on Machine Learning (ICML), pages 26342–26362. PMLR, 2023.

[25] Gabriel Pereyra, George Tucker, Jan Chorowski, Łukasz Kaiser, and Geoffrey Hinton. Regularizing Neural Networks by Penalizing Confident Output Distributions. arXiv preprint arXiv:1701.06548, 2017.

[26] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision, 115(3):211–252, 2015.

[27] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. MobileNetV2: Inverted Residuals and Linear Bottlenecks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4510–4520, 2018.

[28] Mert Bulent Sariyildiz, Yannis Kalantidis, Karteek Alahari, and Diane Larlus. No Reason for No Supervision: Improved Generalization in Supervised Models. arXiv preprint arXiv:2206.15369, 2022.

[29] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. International Journal of Computer Vision, 128(2):336–359, October 2019. ISSN 1573-1405. doi: 10.1007/s11263-019-01228-7. URL http://dx.doi.org/10.1007/s11263-019-01228-7.

[30] Zhiqiang Shen, Zechun Liu, Dejia Xu, Zitian Chen, Kwang-Ting Cheng, and Marios Savvides. Is Label Smoothing Truly Incompatible with Knowledge Distillation: An Empirical Study. arXiv preprint arXiv:2104.00676, 2021.

[31] Carlos N. Silla and Alex A. Freitas. A survey of hierarchical classification across different application domains. Data Mining and Knowledge Discovery, 22:31–72, 2011.
[32] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15:1929–1958, 2014.
[33] Peter Súkeník, Marco Mondelli, and Christoph Lampert. Neural collapse versus low-rank bias: Is deep neural collapse really optimal? arXiv preprint arXiv:2405.14468, 2024.
[34] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the Inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2818–2826, 2016.
[35] Kaihua Tang, Jianqiang Huang, and Hanwang Zhang. Long-tailed classification by keeping the good and removing the bad momentum causal effect. Advances in Neural Information Processing Systems, 33:1513–1524, 2020.
[36] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In Proceedings of the International Conference on Machine Learning (ICML), pages 10347–10357. PMLR, 2021.
[37] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The Caltech-UCSD Birds-200-2011 dataset. 2011.
[38] Hongxin Wei, Renchunzi Xie, Hao Cheng, Lei Feng, Bo An, and Yixuan Li. Mitigating neural network overconfidence with logit normalization, 2022. URL https://arxiv.org/abs/2205.09310.
[39] Guoxuan Xia, Olivier Laurent, Gianni Franchi, and Christos-Savvas Bouganis. Understanding why label smoothing degrades selective classification and how to fix it. arXiv preprint arXiv:2403.14715, 2024.
[40] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In European Conference on Computer Vision (ECCV), pages 418–434, 2018.
[41] Jing Xu and Haoxiong Liu. Quantifying the variability collapse of neural networks. In Proceedings of the International Conference on Machine Learning (ICML), pages 38535–38550. PMLR, 2023.
[42] Kai Yi, Xiaoqian Shen, Yunhao Gou, and Mohamed Elhoseiny. Exploring hierarchical graph representation for large-scale zero-shot image classification. In European Conference on Computer Vision, pages 116–132. Springer, 2022.
[43] Li Yuan, Francis EH Tay, Guilin Li, Tao Wang, and Jiashi Feng. Revisiting knowledge distillation via label smoothing regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3903–3911, 2020.
[44] Matthew D. Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In European Conference on Computer Vision (ECCV), pages 818–833. Springer, 2014.
[45] Chang-Bin Zhang, Peng-Tao Jiang, Qibin Hou, Yunchao Wei, Qi Han, Zhen Li, and Ming-Ming Cheng. Delving deep into label smoothing. IEEE Transactions on Image Processing, 30:5984–5996, 2021.
[46] Jinxin Zhou, Chong You, Xiao Li, Kangning Liu, Sheng Liu, Qing Qu, and Zhihui Zhu. Are all losses created equal: A neural collapse perspective. Advances in Neural Information Processing Systems (NeurIPS), 35:31697–31710, 2022.
[47] Yuxuan Zhou, Wangmeng Xiang, Chao Li, Biao Wang, Xihan Wei, Lei Zhang, Margret Keuper, and Xiansheng Hua. SP-ViT: Learning 2D spatial priors for vision transformers. arXiv preprint arXiv:2206.07662, 2022.

[48] Fei Zhu, Zhen Cheng, Xu-Yao Zhang, and Cheng-Lin Liu. Rethinking confidence calibration for failure prediction. In European Conference on Computer Vision (ECCV), pages 518–536. Springer, 2022.

[49] Hui Zou and Trevor Hastie. Regularization and variable selection via the elastic net. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 67(2):301–320, 2005.


## Appendix


A Technical Appendices and Supplementary Material

Technical appendices with additional results, figures, graphs and proofs may be submitted with the paper submission before the full submission deadline (see above), or as a separate PDF in the ZIP file below before the supplementary material deadline. There is no page limit for the technical appendices.

A Proof of Lemma 3.2

Proof. We aim to demonstrate the validity of Lemma 3.2, which states:
$$H(\mathbf{s}, \mathbf{q}) = H(\mathbf{y}, \mathbf{q}) + L_{LS} \tag{10}$$
where
$$L_{LS} = \alpha \left( H\!\left(\frac{1}{K}, \mathbf{q}\right) - H(\mathbf{y}, \mathbf{q}) \right).$$

Let us proceed with the proof.

We begin by expressing the cross-entropy \(H(\mathbf{s}, \mathbf{q})\):
$$H(\mathbf{s}, \mathbf{q}) = -\sum_{k=1}^{K} s_k \log q_k \tag{11}$$

In the context of label smoothing, \(s_k\) is defined as:
$$s_k = (1 - \alpha) y_k + \frac{\alpha}{K} \tag{12}$$
where \(\alpha\) is the smoothing parameter, \(y_k\) is the original label, and \(K\) is the number of classes.

Substituting this expression for \(s_k\) into the cross-entropy formula:
$$H(\mathbf{s}, \mathbf{q}) = -\sum_{k=1}^{K} \left((1 - \alpha) y_k + \frac{\alpha}{K}\right) \log q_k \tag{13}$$

Expanding the sum:
$$H(\mathbf{s}, \mathbf{q}) = -(1 - \alpha) \sum_{k=1}^{K} y_k \log q_k - \frac{\alpha}{K} \sum_{k=1}^{K} \log q_k \tag{14}$$

We recognize that the first term is equivalent to \((1 - \alpha) H(\mathbf{y}, \mathbf{q})\), and the second term to \(\alpha H\!\left(\frac{1}{K}, \mathbf{q}\right)\). Thus:
$$H(\mathbf{s}, \mathbf{q}) = (1 - \alpha) H(\mathbf{y}, \mathbf{q}) + \alpha H\!\left(\frac{1}{K}, \mathbf{q}\right) \tag{15}$$

Rearranging the terms:
$$H(\mathbf{s}, \mathbf{q}) = H(\mathbf{y}, \mathbf{q}) + \alpha \left( H\!\left(\frac{1}{K}, \mathbf{q}\right) - H(\mathbf{y}, \mathbf{q}) \right) \tag{16}$$

We can now identify \(H(\mathbf{y}, \mathbf{q})\) as the original cross-entropy loss and \(L_{LS} = \alpha \left( H\!\left(\frac{1}{K}, \mathbf{q}\right) - H(\mathbf{y}, \mathbf{q}) \right)\) as the Label Smoothing loss.

Therefore, we have demonstrated that:
$$H(\mathbf{s}, \mathbf{q}) = H(\mathbf{y}, \mathbf{q}) + L_{LS} \tag{17}$$
with \(L_{LS}\) as defined in the lemma. It is noteworthy that the original cross-entropy loss \(H(\mathbf{y}, \mathbf{q})\) remains unweighted by \(\alpha\) in this decomposition, which is consistent with the statement in Lemma 3.2.

B Proof of Theorem 3.3

Proof. We aim to prove the equation:
L_{LS} = \alpha ( z_{gt} - \frac{1}{K}\sum_{k=1}^{K} z_k ) (18)

Let s be the smoothed label vector and q be the predicted probability vector. We start with the cross-entropy between s and q:
H(\mathbf{s}, \mathbf{q}) = -\sum_{k=1}^{K} s_k \log q_k (19)

With label smoothing, s_k = (1 - \alpha) y_k + \frac{\alpha}{K}, where y is the one-hot ground-truth vector and \alpha is the smoothing parameter. Substituting this:
H(\mathbf{s}, \mathbf{q}) = -\sum_{k=1}^{K}\big[(1-\alpha)y_k + \frac{\alpha}{K}\big]\log q_k (20)

Expanding:
H(\mathbf{s}, \mathbf{q}) = -(1-\alpha)\sum_{k=1}^{K} y_k \log q_k - \frac{\alpha}{K}\sum_{k=1}^{K} \log q_k (21)

Since y is a one-hot vector, \sum_{k=1}^{K} y_k \log q_k = \log q_{gt}, where gt is the index of the ground-truth class:
H(\mathbf{s}, \mathbf{q}) = -(1-\alpha)\log q_{gt} - \frac{\alpha}{K}\sum_{k=1}^{K} \log q_k (22)

Using the softmax function, q_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, we can express \log q_k in terms of logits:
\log q_k = z_k - \log\Big(\sum_{j=1}^{K} e^{z_j}\Big) (23)

Substituting this into our expression:
H(\mathbf{s}, \mathbf{q}) = -(1-\alpha)\big[z_{gt} - \log\Big(\sum_{j=1}^{K} e^{z_j}\Big)\big]
-\frac{\alpha}{K}\sum_{k=1}^{K}\big[z_k - \log\Big(\sum_{j=1}^{K} e^{z_j}\Big)\big]
= -(1-\alpha)z_{gt} + (1-\alpha)\log\Big(\sum_{j=1}^{K} e^{z_j}\Big)
-\frac{\alpha}{K}\sum_{k=1}^{K} z_k + \alpha \log\Big(\sum_{j=1}^{K} e^{z_j}\Big)
= -(1-\alpha)z_{gt} - \frac{\alpha}{K}\sum_{k=1}^{K} z_k + \log\Big(\sum_{j=1}^{K} e^{z_j}\Big) (24)

Note that the standard cross-entropy with one-hot labels is
H(\mathbf{y}, \mathbf{q}) = -z_{gt} + \log\Big(\sum_{j=1}^{K} e^{z_j}\Big).
Therefore,
H(\mathbf{s}, \mathbf{q}) - H(\mathbf{y}, \mathbf{q}) = \alpha z_{gt} - \frac{\alpha}{K}\sum_{k=1}^{K} z_k,
which proves (18).

Rearranging:

$$H(\mathbf{s}, \mathbf{q}) = -z_{gt} + \log\Big(\sum_{j=1}^{K} e^{z_j}\Big) + \alpha\Big[z_{gt} - \frac{1}{K}\sum_{k=1}^{K} z_k\Big]$$
(25)

We can identify:

• $$H(\mathbf{y}, \mathbf{q}) = -z_{gt} + \log\Big(\sum_{j=1}^{K} e^{z_j}\Big)$$ (cross-entropy for one-hot vector $\mathbf{y}$)

• $$L_{LS} = \alpha\Big[z_{gt} - \frac{1}{K}\sum_{k=1}^{K} z_k\Big]$$

Thus, we have proven:

$$H(\mathbf{s}, \mathbf{q}) = H(\mathbf{y}, \mathbf{q}) + L_{LS}$$
(26)

Due to the broad usage of CutMix and Mixup in the training recipe of modern Neural Networks, we additionally take their impact into account together with Label Smoothing. Now we additionally prove the case with CutMix and Mixup:

$$L'_{LS} = \alpha\Big((\lambda z_{gt1} + (1-\lambda)z_{gt2}) - \frac{1}{K}\sum_{k=1}^{K} z_k\Big)$$
(27)

With CutMix and Mixup, the smoothed label becomes:

$$s_k = (1-\alpha)(\lambda y_{k1} + (1-\lambda)y_{k2}) + \frac{\alpha}{K}$$
(28)

where $y_{k1}$ and $y_{k2}$ are one-hot vectors for the two ground truth classes from mixing, and $\lambda$ is the mixing ratio.

Starting with the cross-entropy:

$$H(\mathbf{s}, \mathbf{q}) = -\sum_{k=1}^{K} s_k \log q_k$$
(29)

$$= -\sum_{k=1}^{K}\Big[(1-\alpha)(\lambda y_{k1} + (1-\lambda)y_{k2}) + \frac{\alpha}{K}\Big] \log q_k$$
(30)

$$= -(1-\alpha)\sum_{k=1}^{K}(\lambda y_{k1} + (1-\lambda)y_{k2}) \log q_k - \frac{\alpha}{K}\sum_{k=1}^{K} \log q_k$$
(31)

Since $y_{k1}$ and $y_{k2}$ are one-hot vectors:

$$H(\mathbf{s}, \mathbf{q}) = -(1-\alpha)(\lambda \log q_{gt1} + (1-\lambda) \log q_{gt2}) - \frac{\alpha}{K}\sum_{k=1}^{K} \log q_k$$
(32)

where $gt1$ and $gt2$ are the indices of the two ground truth classes.

Using $q_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$, we express in terms of logits:

$$H(\mathbf{s}, \mathbf{q}) = -(1-\alpha)\Big[\lambda\big(z_{gt1} - \log\big(\sum_{j=1}^{K} e^{z_j}\big)\big) + (1-\lambda)\big(z_{gt2} - \log\big(\sum_{j=1}^{K} e^{z_j}\big)\big)\Big]$$
(33)

$$-\frac{\alpha}{K}\sum_{k=1}^{K}\Big[z_k - \log\Big(\sum_{j=1}^{K} e^{z_j}\Big)\Big]$$
(34)

Simplifying:

$$H(\mathbf{s}, \mathbf{q}) = -(1-\alpha)[\lambda z_{gt1} + (1-\lambda)z_{gt2}] + (1-\alpha)\log\left(\sum_{j=1}^{K} e^{z_j}\right) \tag{35}$$

$$-\frac{\alpha}{K}\sum_{k=1}^{K} z_k + \alpha\log\left(\sum_{j=1}^{K} e^{z_j}\right) \tag{36}$$

$$= -(1-\alpha)[\lambda z_{gt1} + (1-\lambda)z_{gt2}] - \frac{\alpha}{K}\sum_{k=1}^{K} z_k + \log\left(\sum_{j=1}^{K} e^{z_j}\right) \tag{37}$$

Rearranging:

$$H(\mathbf{s}, \mathbf{q}) = -[\lambda z_{gt1} + (1-\lambda)z_{gt2}] + \log\left(\sum_{j=1}^{K} e^{z_j}\right) \tag{38}$$

$$+ \alpha\left[\lambda z_{gt1} + (1-\lambda)z_{gt2} - \frac{1}{K}\sum_{k=1}^{K} z_k\right] \tag{39}$$

We can identify:

• $$H(\mathbf{y}', \mathbf{q}) = -[\lambda z_{gt1} + (1-\lambda)z_{gt2}] + \log\left(\sum_{j=1}^{K} e^{z_j}\right)$$ (cross-entropy for mixed label $$\mathbf{y}'$$)

• $$L'_{LS} = \alpha\left[\lambda z_{gt1} + (1-\lambda)z_{gt2} - \frac{1}{K}\sum_{k=1}^{K} z_k\right]$$

Thus, we have proven:

$$H(\mathbf{s}, \mathbf{q}) = H(\mathbf{y}', \mathbf{q}) + L'_{LS} \tag{40}$$

This completes the proof for both cases of Theorem 3.3.

# C Gradient Analysis

## C.1 New Objective Function

The Cross Entropy with Max Suppression is defined as:

$$L_{\text{MaxSup},t}(x, y) = H\left(y_k + \frac{\alpha}{K} - \alpha \cdot \mathbf{1}_{k=\text{argmax}(\mathbf{q})},\, q_t^S(x)\right)$$

where $$H(\cdot, \cdot)$$ denotes the cross-entropy function.

## C.2 Gradient Analysis

The gradient of the loss with respect to the logit $$z_i$$ for each class $$i$$ is derived as:

$$\partial_i^{\text{MaxSup},t} = q_{t,i} - y_i - \frac{\alpha}{K} + \alpha \cdot \mathbf{1}_{i=\text{argmax}(\mathbf{q})}$$

We analyze this gradient under two scenarios:

**Scenario 1: Model makes correct prediction**

In this case, Max Suppression is equivalent to Label Smoothing. When the model correctly predicts the target class ($$\text{argmax}(\mathbf{q}) = \text{GT}$$), the gradients are:

• For the target class (GT): $$\partial_{\mathrm{GT}}^{\mathrm{MaxSup},t} = q_{t,\mathrm{GT}} - \left(1 - \alpha\!\left(1 - \frac{1}{K}\right)\right)$$

• For non-target classes: $$\partial_{i}^{\mathrm{MaxSup},t} = q_{t,i} - \frac{\alpha}{K}$$

## Scenario 2: Model makes wrong prediction

When the model incorrectly predicts the most confident class (argmax($\mathbf{q}$) ≠ GT), the gradients are:

• For the target class (GT): $$\partial_{\mathrm{GT}}^{\mathrm{MaxSup},t} = q_{t,\mathrm{GT}} - \left(1 + \frac{\alpha}{K}\right)$$

• For non-target classes (not most confident): $$\partial_{i}^{\mathrm{MaxSup},t} = q_{t,i} - \frac{\alpha}{K}$$

• For the most confident non-target class: $$\partial_{i}^{\mathrm{MaxSup},t} = q_{t,i} + \alpha\!\left(1 - \frac{1}{K}\right)$$

The Max Suppression regularization technique implements a sophisticated gradient redistribution strategy, particularly effective when the model misclassifies samples. When the model's prediction (argmax($\mathbf{q}$)) differs from the ground truth (GT), the gradient for the incorrectly predicted class is increased by $\alpha\!\left(1 - \frac{1}{K}\right)$, resulting in $$\partial_{\mathrm{argmax}(\mathbf{q})}^{\mathrm{MaxSup},t} = q_{t,\mathrm{argmax}(\mathbf{q})} + \alpha\!\left(1 - \frac{1}{K}\right).$$ Simultaneously, the gradient for the true class is decreased by $\frac{\alpha}{K}$, giving $$\partial_{\mathrm{GT}}^{\mathrm{MaxSup},t} = q_{t,\mathrm{GT}} - \left(1 + \frac{\alpha}{K}\right),$$ while for all other classes, the gradient is slightly reduced by $\frac{\alpha}{K}$: $$\partial_{i}^{\mathrm{MaxSup},t} = q_{t,i} - \frac{\alpha}{K}.$$ This redistribution adds a substantial positive gradient to the misclassified class while slightly reducing the gradients for other classes. The magnitude of this adjustment, controlled by the hyperparameter $\alpha$, effectively penalizes overconfident errors and encourages the model to focus on challenging examples. By amplifying the learning signal for misclassifications, Max Suppression regularization promotes more robust learning from difficult or ambiguous samples.

## Algorithm 1 Gradient Descent with Max Suppression (MaxSup)

Require: Training set $D = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^N$; learning rate $\eta$; number of iterations $T$; smoothing parameter $\alpha$; a neural network $f_\theta(\cdot)$; batch size $B$; total classes $K$.

1: Initialize network weights $\theta$ (e.g., randomly).

2: for $t = 1$ to $T$ do

// Each iteration processes mini-batches of size $B$.

3: for each mini-batch $\{(\mathbf{x}^{(j)}, y^{(j)})\}_{j=1}^B$ in $D$ do

4: Compute logits: $\mathbf{z}^{(j)} \leftarrow f_\theta(\mathbf{x}^{(j)})$ for each sample in the batch.

5: Compute predicted probabilities: $\mathbf{q}^{(j)} \leftarrow \mathrm{softmax}(\mathbf{z}^{(j)})$.

6: Compute cross-entropy loss:
$$L_{\mathrm{CE}} \leftarrow \frac{1}{B} \sum_{j=1}^B H\!\left(y^{(j)}, \mathbf{q}^{(j)}\right).$$

7: // MaxSup component: penalize the top-1 logit

8: For each sample $j$:
$$z_{\max}^{(j)} = \max_{k \in \{1,\ldots,K\}} z_k^{(j)}, \quad \bar{z}^{(j)} = \frac{1}{K} \sum_{k=1}^K z_k^{(j)}.$$
$$L_{\mathrm{MaxSup}} \leftarrow \frac{1}{B} \sum_{j=1}^B \left[z_{\max}^{(j)} - \bar{z}^{(j)}\right].$$

9: Total loss:
$$L \leftarrow L_{\mathrm{CE}} + \alpha\, L_{\mathrm{MaxSup}}.$$

10: Update parameters:
$$\theta \leftarrow \theta - \eta\, \nabla_\theta L.$$

11: end for

12: end for

## D Pseudo Code

Algorithm 1 presents pseudo code illustrating gradient descent with Max Suppression (MaxSup). The main difference from standard Label Smoothing lies in penalizing the highest logit rather than the ground-truth logit.

## E Robustness Under Different Training Recipes

We assess MaxSup’s robustness by testing it under a modified training recipe that reduces total training time and alters the learning rate schedule. This setup models scenarios where extensive training is impractical due to limited resources.

Concretely, we adopt the TorchVision V1 Weights strategy, reducing the total number of epochs to 90 and replacing the cosine annealing schedule with a step learning-rate scheduler (step size = 30). We also set the initial learning rate to 0.1 and use a batch size of 512. This streamlined recipe aims to reach reasonable accuracy within a shorter duration.

As reported in Table 10, MaxSup continues to deliver strong performance across multiple convolutional architectures, generally surpassing Label Smoothing and its variants. Although all methods see a performance decline in this constrained regime, MaxSup remains among the top performers, reinforcing its effectiveness across diverse training conditions.

Table 10: Performance comparison on ImageNet for various convolutional neural network architectures. Results are presented as “mean ± std” (percentage). Bold and underlined entries indicate best and second-best, respectively. (*: implementation details adapted from the official repositories.)

| Method          | ResNet-18        | ResNet-50        | ResNet-101       | MobileNetV2      |
| --------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| Baseline        | 69.11±0.12       | 76.44±0.10       | 76.00±0.18       | 71.42±0.12       |
| Label Smoothing | 69.38±0.19       | 76.65±0.11       | 77.01±0.15       | 71.40±0.09       |
| Zipf-LS*        | 69.43±0.13       | 76.89±0.17       | 76.91±0.14       | 71.24±0.16       |
| OLS*            | 69.45±0.15       | 76.81±0.21       | 77.12±0.17       | 71.29±0.11       |
| **MaxSup**      | **69.59±0.13**   | **77.08±0.07**   | **77.33±0.12**   | **71.59±0.17**   |
| Logit Penalty   | 66.97±0.11       | 74.21±0.16       | 75.17±0.12       | 70.39±0.14       |

## F Increasing Smoothing Weight Schedule

Building on the intuition that a model’s confidence naturally grows as training progresses, we propose a linearly increasing schedule for the smoothing parameter α. Concretely, α is gradually raised from an initial value (e.g., 0.1) to a higher value (e.g., 0.2) by the end of training. This schedule aims to counteract the model’s increasing overconfidence, ensuring that regularization remains appropriately scaled throughout.

Experimental Evidence As shown in Table 11, both Label Smoothing and MaxSup benefit from this α scheduler. For Label Smoothing, accuracy improves from 75.91% to 76.16%, while MaxSup sees a more pronounced gain, from 76.12% to 76.58%. This greater improvement for MaxSup (+0.46%) compared to Label Smoothing (+0.25%) corroborates our claim that MaxSup successfully addresses the inconsistent regularization and error-enhancement issues of Label Smoothing during misclassifications.

Table 11: Effect of an α scheduler on model performance. Here, t and T denote the current and total epochs, respectively. The baseline model does not involve any label smoothing parameter (α).

| Configuration | Formulation                                               | α = 0.1 | α = 0.1 + 0.1 · (t/T) | Remarks          |
| ------------- | --------------------------------------------------------- | ------- | --------------------- | ---------------- |
| Baseline      | –                                                         | 74.21   | 74.21                 | α not used       |
| LS            | α (z_gt − (1/K) Σ_k z_k)                                  | 75.91   | 76.16                 |                  |
| MaxSup        | α (z_max − (1/K) Σ_k z_k)                                 | 76.12   | 76.58                 |                  |

## G Extended Evaluation of Linear Transferability on Different Datasets

To further demonstrate the substantial improvement in feature representation compared to other methods, we compare the linear transfer accuracies of different methods on a broader range of datasets in Table 12.

Table 12: Validation performance of different methods, evaluated using multinomial logistic regression with L2 regularization. Although Label Smoothing and OLS improve ImageNet accuracy, they substantially degrade transfer accuracy compared to MaxSup. Following [15], we selected from 45 logarithmically spaced values between 10⁻⁶ and 10⁵. Notably, the search range is larger than the search range used in Table 3, thus leading to higher overall accuracies on CIFAR10.

| Datasets | CIFAR10   | CIFAR100  | CUB     | Flowers   | Foods     | Pets      |
| -------- | --------- | --------- | ------- | --------- | --------- | --------- |
| CE       | 91.74     | 75.35     | 70.21   | 90.96     | 72.44     | 92.30     |
| LS       | 90.14     | 71.28     | 64.50   | 84.84     | 67.76     | 91.96     |
| OLS      | 90.29     | 73.13     | *67.86* | 87.47     | 69.34     | 92.21     |
| MaxSup   | **91.00** | **73.93** | 67.29   | **88.84** | **70.94** | **92.93** |

## H Extended Comparison to More Label Smoothing Alternatives

We have included Confidence Penalty [25] and Adaptive Label Smoothing with Self-Knowledge [20] for an extended comparison in Table 13. We follow their recommended hyperparameter settings: weight coefficient is set to 0.1 for Confidence Penalty, and the checkpoint with the highest validation accuracy is treated as the teacher for Adaptive Label Smoothing.

To further address the novelty of our work, we additionally compared our method to recent approaches that also identify and aim to fix the issues of Label Smoothing on misclassified samples [6, 38]. Specifically, the method proposed in [38] aims to mitigate overconfidence via logit normalization during training. With its default hyperparameter settings, it achieves an accuracy of 74.32% on ImageNet with a ResNet-50, which is significantly lower than the 76.91% achieved by standard Label Smoothing. This performance aligns with that of Logit Penalty, which similarly minimizes the global l2-norm of logits and can struggle to match baseline LS performance. We also note that these norm-based methods [6, 38, 4] are often highly sensitive to hyperparameter choices, which can limit their practical applicability.

As shown in Table 13, MaxSup outperforms all these alternatives on ImageNet with ResNet-50. This aligns with our theoretical analysis that selectively penalizing z_max yields a more consistent and effective regularization than penalizing all logits (Confidence Penalty), dynamically smoothing the label distribution with self-knowledge (Adaptive LS), or applying global norm-based penalties (Logit Penalty, Logit Normalization).

Table 13: Comparison of classic convolutional neural networks on ImageNet. Results are reported as "mean ± std" (percentage). **Bold** entries highlight the best performance; <u>underlined</u> entries mark the second best. (Methods with * denote code adaptations from official repositories; see text for details.)

| Method                     | ImageNet         |
| -------------------------- | ---------------- |
|                            | ResNet-50        |
| Baseline                   | 76.41 ± 0.10     |
| Label Smoothing            | 76.91 ± 0.11     |
| Zipf-LS*                   | 76.73 ± 0.17     |
| OLS*                       | *77.23 ± 0.21*   |
| MaxSup                     | **77.69 ± 0.07** |
| Logit Penalty              | 76.73 ± 0.10     |
| Logit Normalization [3*]   | 74.32            |
| Confidence Penalty         | 76.58 ± 0.12     |
| Adaptive Label Smoothing*  | 77.16 ± 0.15     |

## I Visualization of the Learned Feature Space

To illustrate the differences between Max Suppression Regularization and Label Smoothing, we follow the projection technique of Müller et al. [23]. Specifically, we select three semantically related classes and construct an orthonormal basis for the plane intersecting their class templates in feature space. We then project each sample's penultimate-layer activation vector onto this plane. To ensure the visual clarity of the resulting plots, we randomly sample 80 images from the training or validation set for each of the three classes.

Selection Criteria
We choose these classes according to two main considerations:

1. Semantic Similarity. We pick three classes that are visually and semantically close.

2. Confusion. We identify a class that the Label Smoothing (LS)–trained model frequently misclassifies and select two additional classes involved in those misclassifications (Figure 3c, Figure 4c). Conversely, we also examine a scenario where a class under Max Suppression is confused with others, highlighting key differences (Figure 3d, Figure 4d).

| ![Projections MaxSup showing Class 66, 67, 68]()              | ![Projections MaxSup showing Class 281, 282, 292]()              | ![Projections MaxSup showing Class 709, 748, 893]()              | ![Projections MaxSup showing Class 664, 782, 851]()              |
| ------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| ![Projections w/o Label Smoothing showing Class 66, 67, 68]() | ![Projections w/o Label Smoothing showing Class 281, 282, 292]() | ![Projections w/o Label Smoothing showing Class 709, 748, 893]() | ![Projections w/o Label Smoothing showing Class 664, 782, 851]() |
| ![Projections Label Smoothing showing Class 66, 67, 68]()     | ![Projections Label Smoothing showing Class 281, 282, 292]()     | ![Projections Label Smoothing showing Class 709, 748, 893]()     | ![Projections Label Smoothing showing Class 664, 782, 851]()     |
| (a) Semantically Similar Classes                              | (b) Semantically Similar Classes                                 | (c) Confusing Classes (LS)                                       | (d) Confusing Classes (MaxSup)                                   |

Figure 3: Visualization of penultimate-layer activations from DeiT-Small (trained with CutMix and Mixup) on the ImageNet validation set. The top row shows embeddings for a MaxSup-trained model, and the bottom row shows embeddings for a Label Smoothing (LS)–trained model. In each subfigure, classes are either semantically similar or confusingly labeled. Compared to LS, MaxSup yields more pronounced inter-class separability and richer intra-class diversity, suggesting stronger representation and classification performance.

Observations
As shown in Figures 3 and 4, models trained with Max Suppression exhibit:

• Enhanced inter-class separability. Distinct classes occupy more clearly separated regions, aligning with improved classification performance.

• Greater intra-class variation. Instances within a single class are not overly compressed, indicating a richer representation of subtle differences.

For instance, images of Schipperke dogs can differ markedly in viewpoint, lighting, background, or partial occlusions. Max Suppression preserves such intra-class nuances in the feature space, enabling the semantic distances to visually related classes (e.g., Saluki, Grey Fox, or Belgian Sheepdog) to dynamically adjust for each image. Consequently, Max Suppression provides a more flexible, fine-grained representation that facilitates better class discrimination.

J. Ablation on the Weight Schedule
We conducted an ablation study on the α schedule, as shown in Table 15. The consistently high accuracy across settings demonstrates the robustness of MaxSup. The adaptive α schedule, adopted from [20], further highlights the method’s integrity and compatibility with principled design choices.


![Figure 71](image_71.jpg)
<!-- Caption: Appendix, index 71 -->


(a) Semantically Similar Classes (b) Semantically Similar Classes (c) Confusing Classes (LS) (d) Confusing Classes (MaxSup)

Figure 4: Visualization of the penultimate-layer activations for DeiT-Small (trained with CutMix and Mixup) on selected ImageNet classes. The top row shows results for a MaxSup-trained model; the bottom row shows Label Smoothing (LS). In (a,b), the model must distinguish semantically similar classes (e.g., Saluki vs. Grey Fox; Tow Truck vs. Pickup), while (c,d) involve confusing categories (e.g., Jean vs. Shoe Shop; Stinkhorn vs. related objects). Compared to LS, MaxSup yields both improved inter-class separability and richer intra-class variation, indicating more robust representation learning.

Table 14: Feature representation metrics for a ResNet-50 model trained on ImageNet-1K, reported on both training and validation sets. We measure intra-class variation (d̄_within) and overall average distance (d̄_total). Inter-class separability (R^2) is calculated as R^2 = 1 − (d̄_within / d̄_total). Higher values (↑) of d̄_within and R^2 are preferred.

Method: Baseline
d̄_within — Train: 0.3114; Val: 0.3313
d̄_total — Train: 0.5212; Val: 0.5949
R^2 — Train: 0.4025; Val: 0.4451

Method: LS
d̄_within — Train: 0.2632; Val: 0.2543
d̄_total — Train: 0.4862; Val: 0.4718
R^2 — Train: 0.4690; Val: 0.4611

Method: OLS
d̄_within — Train: 0.2707; Val: 0.2820
d̄_total — Train: 0.6672; Val: 0.6570
R^2 — Train: 0.5943; Val: 0.5708

Method: Zipf’s
d̄_within — Train: 0.2611; Val: 0.2932
d̄_total — Train: 0.5813; Val: 0.5628
R^2 — Train: 0.5522; Val: 0.4790

Method: MaxSup
d̄_within — Train: 0.2926 (+0.03); Val: 0.2998 (+0.05)
d̄_total — Train: 0.6081 (+0.12); Val: 0.5962 (+0.12)
R^2 — Train: 0.5188 (+0.05); Val: 0.4972 (+0.04)

Method: Logit Penalty
d̄_within — Train: 0.2840; Val: 0.3144
d̄_total — Train: 0.7996; Val: 0.7909
R^2 — Train: 0.6448; Val: 0.6024

K Analysis of Computation Efficiency

Beyond the standard cross-entropy operations, MaxSup only requires:
1. A max operation to determine the largest logit (O(K) complexity),
2. A mean operation over the K-dimensional logit vector, and
3. One subtraction between these two scalars.

Table 15: Ablation study on alpha schedules using ResNet-50 on ImageNet-1K.
Schedule — α = 0.1 + 0.1 (t/T): 77.65
Schedule — α = 0 + 0.1 (t/T): 77.62
Schedule — α = 0.2 + 0.1 (t/T): 77.43
Schedule — Adaptive Alpha: 77.70


![Figure 74](image_74.jpg)
<!-- Caption: Appendix, index 74 -->


Figure 5: Comparison of logit distributions under different regularizers.

Since K (the number of classes) is usually small (e.g., 1000 for ImageNet-1K), this overhead is minimal compared to the overall forward/backward pass of a deep network. In Table 16, we report the average training time per epoch using a ResNet-50 model on the ImageNet-1K dataset.

Table 16: Average training time per epoch on ImageNet-1K with ResNet-50.

| Method     | CE (One-Hot) | CE + LS    | CE + MaxSup |
| ---------- | ------------ | ---------- | ----------- |
| Time/Epoch | 3 min 51 s   | 3 min 52 s | 3 min 51 s  |

As seen above, the measured run times are nearly identical across all three configurations. Thus, the additional cost of MaxSup is negligible compared to the total computation for large-scale training.

L. Logits Visualization

As mentioned in Section 4.1.2, all Label Smoothing variants apply a different penalty on the logits. To illustrate the impact of different methods on the logits, we plot the histogram of logits of ResNet-50 networks trained with each method over the ImageNet validation set, as shown in Figure 5.


![Figure 77](image_77.jpg)
<!-- Caption: Appendix, index 77 -->
