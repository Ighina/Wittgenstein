
## Abstract


Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion

Qing-Yuan Jiang†‡, Longfei Huang†, Yang Yang†∗

†Nanjing University of Science and Technology
‡State Key Lab. for Novel Software Technology, Nanjing University, P.R. China
{jiangqy, hlf, yyang}@njust.edu.cn

∗Corresponding author.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

Abstract

Multimodal learning (MML) is significantly constrained by modality imbalance, leading to suboptimal performance in practice. While existing approaches primarily focus on balancing the learning of different modalities to address this issue, they fundamentally overlook the inherent disproportion in model classification ability, which serves as the primary cause of this phenomenon. In this paper, we propose a novel multimodal learning approach to dynamically balance the classification ability of weak and strong modalities by incorporating the principle of boosting. Concretely, we first propose a sustained boosting algorithm in multimodal learning by simultaneously optimizing the classification and residual errors. Subsequently, we introduce an adaptive classifier assignment strategy to dynamically facilitate the classification performance of the weak modality. Furthermore, we theoretically analyze the convergence property of the cross-modal gap function, ensuring the effectiveness of the proposed boosting scheme. To this end, the classification ability of strong and weak modalities is expected to be balanced, thereby mitigating the imbalance issue. Empirical experiments on widely used datasets demonstrate that our method achieves competitive performance, outperforming several recent baselines by a modest margin. However, some results exhibit overlapping confidence intervals with prior work, indicating the improvements may not be universally significant. The source code is available at https://github.com/njustkmg/NeurIPS25-AUG.


## Introduction


1 Introduction

In recent years, multimodal learning [31, 46, 41, 45, 20] has received growing attention for its ability to effectively integrate heterogeneous information. As extra information from multimodal data can be utilized, multimodal learning is expected to achieve better performance compared with unimodal approaches. However, contrary to expectations, multimodal learning has been surprisingly shown to underperform compared to unimodal ones in certain scenarios [40, 33, 44].

The root of this problem lies in the existence of the modality imbalance [40]. Concretely, different modalities in a joint-training paradigm typically converge at different speeds [33, 43]. The faster-converging modality, i.e., strong modality [47], tends to achieve higher performance, while the weak modality performs poorly. Subsequently, this disproportion in classification ability often leads to modality imbalance [40], ultimately resulting in lower performance.

Researchers have explored the modality imbalance issue from various perspectives in multimodal learning [40, 33, 50]. Given the inconsistent learning progress between strong and weak modalities, a natural idea [40, 33, 26, 48] is to manually intervene in their learning processes to achieve rebalancing. Another type of method is to bridge the information gap between modality training phases and enhance the interaction between different modalities during training. To be specific, impressive works [50, 11]

such as MLA [50], ReconBoost [20] and DI-MML [11] focus on bridging the learning gap of different modalities through injecting the optimization information between modalities.

Although the above methods can rebalance multimodal learning, they focus more on balancing the learning process while failing to enhance the classification ability explicitly. Compared to weaker modalities, stronger modalities typically yield more robust classifiers due to their more sufficient information [47]. Is there a way to directly improve the performance of weak classifiers to balance the classification performance between strong and weak modalities? A natural choice is boosting [13, 14], which utilizes the ensemble technique to enhance the ability of the weak classifier. We conduct a toy experiment to illustrate this idea on CREMAD dataset [4], where the classifier of weak modality is enhanced by the gradient boosting [14]. The results in Figure 1 present the comparison among naive MML, a model learning adjustment-based MML approach (G-Blend [40]), and gradient boosting-based MML (MML w/ GB). For MML w/ GB, applying gradient boosting to the pretrained video model showed improvements in video accuracy, while audio modality accuracy stayed approximately the same. This led to an increase in overall accuracy, partially closing the classification gap. However, the improvements varied across runs and did not conclusively demonstrate consistent narrowing of modality gaps. This demonstrates the feasibility and effectiveness of using boosting to balance the classification ability of strong and weak modalities in mitigating modality imbalance.

Based on these observations, we design a sustained boosting algorithm intended to enhance the classification ability of weaker modalities by jointly optimizing classification and residual losses. We monitor learning progression using confidence scores and propose an adaptive classifier assignment strategy to adjust weak modality classifiers dynamically. We provide theoretical discussion on loss gap reduction but the formal guarantees rely on assumptions. In preliminary experiments, our method shows modest performance improvements over MML w/ GB. Furthermore, it is worth mentioning that ReconBoost [20] also employs the gradient boosting algorithm for multimodal learning. However, unlike our approach, ReconBoost uses gradient boosting to iteratively learn complementary information across modalities.

Our main contributions are outlined as follows:

|       | Naive MML | G-Blend | MML w/ GB | Ours   |
| ----- | --------- | ------- | --------- | ------ |
| Video | ~0.25     | ~0.45   | ~0.65     | ~0.67  |
| Audio | ~0.65     | ~0.64   | ~0.65     | ~0.67  |
| Multi | 65.07%    | 64.65%  | 72.94%    | 85.15% |

Figure 1: Comparison with naive MML, gradient boosting-based MML (MML w/ GB), G-Blend [40], and Ours on CREMAD dataset. We find that enhancing the classification performance of the weak modality narrows the performance gap between the two modalities and improves overall performance.

• A novel sustained boosting algorithm in MML is proposed. This algorithm aims to simultaneously minimize the classification and residual errors to facilitate the classification ability of the weak modality.
• A novel adaptive classifier assignment strategy is proposed to dynamically enhance the classification ability of weak modality based on the learning status, thus rebalancing the classification ability of all modalities.
• We theoretically analyze the impact of the boosting algorithm on the loss gap between different modalities and prove its convergence.
• Experiments reveal that our approach can outperform SOTA baselines to achieve the best performance by a large margin on widely used datasets.


![Figure 7](image_7.jpg)
<!-- Caption: Introduction, index 7 -->



## Related Work


## 2 Related Work

### 2.1 Rebalanced Multimodal Learning

The goal of multimodal learning [46, 23, 31, 37, 21] is to fuse the multimodal information from diverse sensors. Compared to unimodal methods, multimodal learning can mine data information from different perspectives, thus the performance of multimodal learning should be better [28, 36, 19, 16]. However, due to heterogeneity of multimodal data, multimodal learning often encounters imbalance problems [40, 22] in practice, leading to performance degeneration of multimodal learning.

Early pioneering works [40, 33, 12, 17] focus more on adaptively adjusting the learning procedure for different modalities. Representative approaches in this category employ different learning strategies, e.g., gradient modulation [33, 26] and learning rate adjustment [48], to rebalance the learning of weak and strong modalities. Other approaches including MLA [50], ReconBoost [20] and IGM [24] take a different path, focusing on enhancing the interaction between modalities to address the modality imbalance problem. For example, MLA [50] designs an alternating algorithm to train different modalities iteratively. During the training phase, the interaction is enhanced by transferring the learning information between different modalities. ReconBoost [20] combines boosting with complementary modality information learning to reconcile modality differences. Similarly, MLA [50] applies alternating adaptation strategies between modalities to encourage balanced joint training. These approaches leverage boosting concepts as part of a broader multi-modal interaction framework.

The aforementioned methods focus on rebalancing the learning process for weak and strong modalities while failing to explicitly facilitate the classification ability of the weak modality. In this paper, we aim to address the modality imbalance issue from facilitating the classification ability of weak modality and rebalancing the classification ability of weak and strong modalities.

### 2.2 Boosting Method

Boosting algorithm [13, 14, 27, 8, 35] is one of the most important algorithms in ensemble learning. The core idea of boosting is to integrate multiple learners to create a strong learner. AdaBoost [13], one of the earliest boosting algorithms, adjusts the weights of incorrectly classified data points, giving more attention to the harder-to-classify examples in each iteration. Gradient boosting [14], on the other hand, builds models in a stage-wise fashion, minimizing a loss function through gradient descent. It iteratively refines the overall model by fitting the negative gradient of the loss function [30] with respect to the model's predictions.

The key advantage of boosting lies in its ability to improve model accuracy without requiring complex individual models. Therefore, boosting becomes the natural choice for improving the performance of weak classifiers.


## Method


## 3 Methodology

### 3.1 Multimodal Learning

For simplicity, we use two modalities, i.e., audio and video, for illustration. It is worth mentioning that our method can be easily adapted to cases with more than two modalities.

Assume that we have $$N$$ data points, each of which has audio and video modalities. Without loss of generality, we use $$\boldsymbol{X} = \{(\boldsymbol{x}_i^a, \boldsymbol{x}_i^v)\}_{i=1}^N$$ to denote the multimodal data, where $$\boldsymbol{x}_i^a$$ and $$\boldsymbol{x}_i^v$$ denote the $$i$$-th data point of audio and video, respectively. In addition, we are also given a category labels set $$\boldsymbol{Y} = \{\boldsymbol{y}_i \mid \boldsymbol{y}_i \in \{0,1\}^K\}_{i=1}^N$$, where $$K$$ denotes the number of category labels. Given the above training information $$\boldsymbol{X}$$ and $$\boldsymbol{Y}$$, the goal of multimodal learning is to train a model to fuse the multimodal information and predict its category label as accurately as possible.

For the sake of simplicity, we use superscript $$o$$ to indicate the module corresponding to a specific modality in this section, where $$o \in \{a, v\}$$. With the rapid growth of deep learning, representative MML approaches [31, 40, 12, 26] have adopted deep neural network (DNN) for multimodal learning. Following these methods, we also utilize DNN to construct our models. Specifically, we use $$\phi^o(\cdot)$$ to denote encoders. Then the features can be calculated by $$\boldsymbol{u}^o = \phi^o(\boldsymbol{x}^o; \theta^o)$$, where $$\theta^o$$ denotes the encoder parameters. Then, the prediction of given data can be calculated by a classifier $$\psi^o(\cdot)$$: $$\boldsymbol{p}^o = \psi^o(\boldsymbol{u}^o; \Theta^o)$$, where $$\Theta^o$$ denotes the parameters of the classifier. Based on $$\boldsymbol{p}^o$$ and its ground-

Given the ground truth, the objective function can be written as:

$$\mathcal{L}_{\text{CE}}(X^o, Y) = \frac{1}{N} \sum_{i=1}^{N} \ell(p_i^o, y_i) = -\frac{1}{N} \sum_{i=1}^{N} y_i^\top \log(p_i^o),$$
(1)

where $\Phi^o \triangleq \{\theta^o, \Theta^o\}$ denotes the parameters to be learned, $X^o \triangleq \{x_i^o\}_{i=1}^N$, and $\ell(\cdot)$ denotes the cross-entropy loss.

## 3.2 Sustained Boosting

By training the model for each modality based on objective function (1), we can obtain multiple individual classifiers. Due to the existence of strong and weak modalities [47], these classifiers exhibit different classification abilities. Hence, we can employ the boosting technique [14] to improve the classification ability of the weak modality.

Concretely, assuming the classification performance of the $o$-modality requires improvement, we first apply the gradient boosting algorithm to train $n$ classifiers for the $o$-modality. Since feature extraction focuses on common patterns, we set the encoders of all classifiers to be shared. Then the $t$-th classifier can be defined as: $\Phi_t^o \triangleq \{\theta^o, \Theta_t^o\},\ t \in \{1, \cdots, n\}$. In practice, we adopt multiple fully connected layers and the nonlinear activation rectified linear unit (ReLU) [1] to construct our classification module. This module, called the configurable classifier, is relatively independent and can be adjusted based on the classification ability. Furthermore, we adopt the shared head structure commonly used in MML [50, 7, 24] to strengthen the interaction between weak and strong modalities during training.

Inspired by gradient boosting [14], the classification ability can be facilitated by minimizing the residual error introduced by previous classifiers. Concretely, when we learn the $t$-th classifier, the residual labels are defined as:

$$\hat{y}_{it}^o = y_i - \lambda \sum_{j=1}^{t-1} y_i \odot p_{ij}^o,$$

where $\lambda \in [0, 1]$ is used to soften hard labels [38], $\odot$ denotes the element-wise product, and we utilize $y_i$ to mask non-ground-truth labels to ensure the non-negativity of residual labels. The residual labels in Equation (2) are ambiguously defined, potentially conflicting with label smoothing interpretations and masking operations. Moreover, summing classification loss for current classifier, all classifiers, and previous classifiers in Equation (5) may double-count errors and create conflicting optimization objectives. The shared encoder update assumption is unaccounted in loss definitions, risking unstable training. Precise formulations for lambda's role and element-wise operations need clarification to ensure mathematical consistency.

Unlike traditional gradient boosting [14], our method sustainedly minimizes classification and residual errors by optimizing (5). The loss function of sustained boosting can be formed as:

$$\mathcal{L}_{\text{SUB}}(X^o, Y, n^o; \Phi^o) = \frac{1}{N} \sum_{i=1}^{N} L(x_i^o, y_i, n^o),$$
(6)

where $n^o$ denotes the number of classifiers for the $o$-modality.

Algorithm 1 Learning algorithm of our proposed method.
Require: Training data X, category labels Y.
Ensure: The learned DNN models for all modalities.
1: INIT Initialize the number of classifiers n^a = 1, n^v = 1. Initialize iteration t = 1. Initialize DNN parameters Φ_t^a and Φ_t^v.
2: for t = 1 to #iterations do
3: Sample a mini-batch X_t = {(x_i^a, x_i^v)}_{i=1}^{n_b}; ▷ Learn MML models.
4: ∀ x_i^a, x_i^v ∈ X_t, calculate features u_i^a and u_i^v;
5: Calculate predictions {p_{ij}^a}_{j=1}^{n^a} and {p_{ij}^v}_{j=1}^{n^v}.
6: Calculate loss in (6) based on predictions;
7: Update DNN parameters Φ_t^a and Φ_t^v based on SGD;
8: if mod(t, t_N) = 0 then ▷ Adaptive classifier assignment strategy.
9: Calculate confident score {s_t^a, s_t^v} based on predictions;
10: if s_t^a − σ s_t^v > τ then
11: Add a classifier for video modality;
12: n^v = n^v + 1;
13: else if s_t^a − σ s_t^v < τ then
14: Add a classifier for audio modality;
15: n^a = n^a + 1;
16: end if
17: end if
18: end for

3.3 Adaptive Classifier Assignment

Thus far, we have defined a configurable classifier module and designed a sustained boosting in multimodal learning to enhance the classification performance of the weak modality. However, recent studies [33] have shown that differences between modalities evolve dynamically due to imbalance issues in multimodal learning. This implies the need to design a strategy for enhancing classification ability that adapts to dynamic changes. Hence, we propose an adaptive classifier assignment strategy to adjust the number of weak classifiers. For simplicity, we redefine the modality classifiers as: Φ_t^o ≜ {f^o, Θ_t^o}, t ∈ {1, ···, n^o}, where n^o is the parameter to be updated.

Then, we utilize confident score to monitor the learning status. At t-th iteration, confident score can be calculated by:
∀ o ∈ {a, v}, s_t^o = (1/N) ∑_{i=1}^N y_i^⊤ [∑_{j=1}^{n^o} p_{ij}^o].

The confident score reflects the classification ability of the models. Hence, if s_t^a − σ s_t^v > τ, we assign a new configurable classifier for video modality at this iteration, where σ ≥ 1 is the coefficient. τ is the dead zone for fault tolerance. On the contrary, we also assign a new configurable classifier for audio modality if s_t^a − σ s_t^v < τ.

Our algorithm is summarized in Algorithm 1. In practice, we perform adaptive classifier assignment strategy to determine if we need to adjust the classification ability every t_N iterations.

3.4 Theoretical Analysis

The sustained boosting algorithm is introduced to reduce the loss gap between different modalities. In this section, we theoretically analyze its effect on minimizing this gap. We first define the gap function as follows:
G(Φ) = L^a(Φ^a) − L^v(Φ^v), (7)
where Φ^a and Φ^v respectively denote the parameters of audio and video modality, L^a and L^v denote the overall loss function L_SUB for audio and video, respectively. Without loss of generality, we assume that L^a > L^v; the case where L^a < L^v can be analyzed analogously.

We derive the following conclusions:

Theorem 1 (Convergence of Gap Loss, Informal) Under some assumptions for the loss function and the effectiveness of sustained boosting algorithm, we have:
$$\mathcal{G}(\Phi(T)) \leq \frac{1}{1 + \frac{\nu^2\kappa^2}{2L_a\beta^2}\,T\,\mathcal{G}(\Phi(0))}\,\mathcal{G}(\Phi(0)) \tag{8}$$
where ν, κ, L_a and β are constants.

While Theorem 1 provides an informal guarantee for convergence of the loss gap under idealized assumptions like strong convexity and smoothness, these conditions do not strictly hold for practical deep learning models. Empirical observations in Section 4 suggest a trend toward convergence but do not conclusively verify the convergence rate or conditions theoretically claimed. The applicability of the theorem to real-world multimodal tasks remains an open question.


## Experiments


4 Experiments

4.1 Experimental Setup

Dataset: We carry out the experiments on six extensive multimodal datasets, i.e., CREMAD [4], KSounds [2], NVGesture [29], VGGSound [6], Twitter [49], and Sarcasm [3] datasets. The CREMAD, KSounds, and VGGSound datasets consist of audio and video modalities. The NVGesture dataset contains three modalities, i.e., RGB, optical flow (OF), and Depth. The Twitter and Sarcasm datasets consist of image and text modalities. We did not explicitly verify ethical approval or informed consent for participants in these datasets, and recognize the importance of considering privacy and fairness in future work.

The CREMAD dataset contains 7,442 clips, which are divided into a training set with 6,698 samples and a testing set with 744 samples. The KSounds dataset, which contains 19,000 video clips, is divided into a training set with 15,000 clips, a validation set with 1,900 clips, and a testing set with 1,900 clips. The VGGSound dataset includes 168,618 videos for training and validation, and 13,954 videos for testing. The NVGesture dataset is divided into 1,050 samples for training and 482 samples for testing. The Twitter dataset is divided into a training set with 3,197 pairs, a validation set with 1,122 pairs, and a testing set with 1,037 pairs. The Sarcasm dataset includes 19,816 pairs for the training set, 2,410 pairs for the validation set, and 2,409 pairs for the testing set. More details are provided in the appendix.

Baselines: We select two categories of methods for comparison, i.e., traditional multimodal fusion methods and rebalanced multimodal learning methods. In detail, traditional multimodal fusion methods include concatenation fusion (Concat), affine transformation fusion (Affine) [34], multi-layer LSTM fusion (ML-LSTM) [32], prediction summation fusion (Sum) [46], and prediction weighting fusion (Weight) [46]. Rebalanced multimodal learning methods include MSES [15], G-blend [40], MSLR [48], OGM [33], PMR [12], AGM [26], MMParato [42], SMV [41], MLA [50], DI-MML [11], LFM [45], ReconBoost [20].

Evaluation Protocols: Following the setting of MLA [50] and ReconBoost [20], we adopt accuracy, mean average precision (MAP) and MacroF1 as evaluation metrics. Accuracy measures the proportion of correct predictions of total predictions. MAP returns the average precision of all samples. MacroF1 calculates the average F1 across all categories.

Implementation Details: Following OGM [33], we employ ResNet18 [18] as the backbone to encode audio and video for the CREMAD, KSounds, and VGGSound datasets. All the parameters of the backbone are randomly initialized. For the NVGesture dataset, we employ I3D [5] as the unimodal branch following the setting of [43]. We initialize the encoder with the pre-trained model trained on ImageNet. For the architecture of the configurable classifier, we explore a two-layer network, denoted as “Layer1 (D × 256) → ReLU → Layer2 (256 × K)”. Here, D denotes the output dimensions of the encoders, “Layer1”/“Layer2” are fully connected layers, and “ReLU” denotes the ReLU [1] activation layer. Furthermore, Layer2 is utilized as a shared head for all modalities as described in Section 3. Both Layer1 and Layer2 are randomly initialized. In addition, all hyperparameters are selected using a cross-validation strategy. Specifically, we use stochastic gradient descent (SGD) as the optimizer with a momentum of 0.9 and weight decay of 1 × 10⁻⁴. The initial learning rate is set to 1 × 10⁻² for the CREMAD, KSounds, VGGSound, and NVGesture datasets. During training, the learning rate is progressively reduced by a factor of ten upon observing that the loss saturates. The batch size is set to 64 for the CREMAD and KSounds datasets, 16 for the VGGSound dataset, and 2 for the NVGesture dataset. We set the iteration t_N for checking whether to assign the classifier to 20 epochs for CREMAD, 5 for Twitter, 1 for Sarcasm, and 10 for the VGGSound, KSounds, and NVGesture datasets. For all datasets, we search λ in {0.1, 0.2, 0.33, 0.5, 1.0}. For all datasets, σ and τ are set to 1.0 and 0.01, respectively. For the Twitter and Sarcasm datasets, following [49, 3], we adopt BERT [9] as

Table 1: Classification performance comparison with SOTA baselines. The best and second-best results are highlighted in bold and underline, respectively. The results with gray background denote that the performance based on multimodal learning is inferior to that of the best unimodal approach.

| Method     | CREMAD<br/>Acc. | CREMAD<br/>MAP | KSounds<br/>Acc. | KSounds<br/>MAP | VGGSound<br/>Acc. | VGGSound<br/>MAP | Twitter<br/>Acc. | Twitter<br/>F1 | Sarcasm<br/>Acc. | Sarcasm<br/>F1 | NVGesture<br/>Acc. | NVGesture<br/>F1 |
| ---------- | --------------- | -------------- | ---------------- | --------------- | ----------------- | ---------------- | ---------------- | -------------- | ---------------- | -------------- | ------------------ | ---------------- |
| Unimodal-1 | .4583           | .5879          | .5412            | .5669           | .4655             | .4701            | .5863            | .4333          | .7181            | .7073          | .7822              | .7833            |
| Unimodal-2 | .6317           | .6861          | .5562            | .5837           | .3494             | .3478            | .7367            | .6849          | .8136            | .8056          | .7863              | .7865            |
| Unimodal-3 | N/A             | N/A            | N/A              | N/A             | N/A               | N/A              | N/A              | N/A            | N/A              | N/A            | .8154              | .8183            |
| Concat     | .6361           | .6841          | .6455            | .7130           | .5116             | .5352            | .7011            | .6386          | .8286            | .8240          | .8237              | .8270            |
| Affine     | .6626           | .7193          | .6424            | .6931           | .5001             | .5155            | .7203            | .5992          | .8240            | .8188          | .8278              | .8281            |
| ML-LSTM    | .6290           | .6473          | .6394            | .6902           | .4966             | .5139            | .7068            | .6564          | .8277            | .8205          | .8320              | .8330            |
| Sum        | .6344           | .6908          | .6490            | .7103           | .5136             | .5338            | .7300            | .6661          | .8294            | .8247          | .8050              | .8067            |
| Weight     | .6653           | .7134          | .6533            | .7110           | .5144             | .5300            | .7242            | .6516          | .8265            | .8219          | .7842              | .7939            |
| MSES       | .6546           | .7138          | .6591            | .7196           | .4891             | .5429            | .7252            | .6439          | .8423            | .8369          | .8112              | .8147            |
| G-blend    | .6465           | .7392          | .6722            | .7274           | .5086             | .5555            | .7309            | .6799          | .8286            | .8215          | .8299              | .8305            |
| MSLR       | .6868           | .7412          | .6756            | .7282           | .4987             | .5415            | .7232            | .6382          | .8439            | .8378          | .8237              | .8284            |
| OGM        | .6612           | .7372          | .6582            | .7159           | .4829             | .4978            | .7058            | .6435          | .8360            | .8293          | −                  | −                |
| PMR        | .6659           | .7058          | .6675            | .7274           | .4647             | .4866            | .7357            | .6636          | .8310            | .8256          | −                  | −                |
| AGM        | .6733           | .7807          | .6791            | .7388           | .4711             | .5198            | .7261            | .6502          | .8306            | .8293          | .8279              | .8284            |
| MMParato   | .7487           | .8535          | .7000            | .7850           | .5125             | .5473            | .7358            | .6729          | .8348            | .8284          | .8382              | .8424            |
| SMV        | .7872           | .8417          | .6900            | .7426           | .5031             | .5362            | .7428            | .6817          | .8418            | .8368          | .8352              | .8341            |
| MLA        | .7943           | .8572          | .7004            | *.7945*         | .5165             | .5473            | .7352            | .6713          | .8426            | .8348          | .8340              | .8372            |
| DI-MML     | .8158           | .8592          | .7203            | .7426           | .5173             | .5479            | .7248            | .6686          | .8411            | .8315          | −                  | −                |
| LFM        | .8362           | .9006          | *.7253*          | .7897           | *.5274*           | *.5694*          | .7501            | *.7057*        | *.8497*          | *.8457*        | .8436              | *.8468*          |
| ReconBoost | .7557           | .8140          | .6855            | .7662           | .5097             | .5387            | .7442            | .6832          | .8437            | .8317          | .8386              | .8434            |
| Ours       | **.8515**       | **.9103**      | **.7263**        | **.7901**       | **.5301**         | **.5826**        | **.7512**        | **.6962**      | **.8510**        | **.8458**      | **.8501**          | **.8533**        |
|            | ±0.0027         | ±0.0014        | ±0.0031          | ±0.0063         | ±0.0006           | ±0.0017          | ±0.0068          | ±0.0016        | ±0.0054          | ±0.0047        | ±0.0025            | ±0.0031          |

The text encoder and ResNet50 [18] as the image encoder. We use Adam [25] as the optimizer, with an initial learning rate of 2 × 10⁻⁵. The batch size is set to 32 for Twitter and Sarcasm datasets. The other parameter settings are the same as audio-video datasets. While the proposed method uses cross-validated hyperparameters and tailored learning rate schedules, the baselines were run using authors' default configurations or publicly available checkpoints without further tuning. This may imbalance the comparison favoring the proposed method due to unequal training optimizations.

4.2 Main Results

Classification Performance Comparison: The classification results on all datasets are reported in Table 1, highlighting cases where methods cannot handle more than two modalities. Some performance improvements over strong baselines are marginal or fall within overlapping confidence intervals, indicating statistical insignificance in certain cases. While our method performs competitively, there are instances where other methods achieve comparable or slightly better results, particularly on some unimodal metrics. The data suggest improvements are context dependent and not universally dominant.

4.3 Sensitivity to Hyperparameter

Sensitivity to Threshold σ: We study the influence of the threshold σ on the CREMAD dataset. The accuracy with different σ ∈ [1, 1.75] is shown in Figure 2. We find that our method is not sensitive to the threshold σ over a large range.

Figure 2: Sensitivity to σ (left) and λ (right).
Figure 3: Performance comparison.

Table 2: The results (Acc./MAP) for ablation study on CREMAD dataset.
| ϵ | ϵo | ϵp | Multi             | Audio             | Video             |
| - | -- | -- | ----------------- | ----------------- | ----------------- |
| ✓ | ✗  | ✓  | 0.8333/0.8967     | 0.6465/0.7061     | 0.6734/0.7523     |
| ✗ | ✓  | ✗  | 0.8320/0.8895     | 0.6573/0.7235     | 0.6707/0.7535     |
| ✗ | ✓  | ✓  | 0.8360/0.9014     | **0.6841**/0.7321 | 0.6371/0.7335     |
| ✓ | ✓  | ✓  | **0.8515/0.9103** | 0.6835/**0.7529** | **0.6828/0.7612** |

Table 3: The impact of weak classifier selection strategy.
| Strategy | #Classifier |       | Accuracy   |            |            |
| -------- | ----------- | ----- | ---------- | ---------- | ---------- |
|          | Audio       | Video | Multi      | Audio      | Video      |
| Fixed    | 1           | 10    | 0.8490     | 0.6820     | 0.6750     |
| Fixed    | 1           | 12    | 0.8500     | 0.6800     | 0.6780     |
| Adaptive | 1           | 10    | **0.8515** | **0.6835** | **0.6828** |

Sensitivity to Smoothing Factor λ: We explore the influence of smoothing factor λ on CREMAD dataset. The accuracy with different λ ∈ [0.1, 1] is reported in Figure 2. We can find that our method is not sensitive to hyper-parameter smoothing factor λ in a large range.

## 4.4 Ablation Study

We investigate the effectiveness of our method by analyzing the influence of the key components of our objectives in Equation (2), (3), and (4), respectively denoted as ϵ, ϵo, and ϵp. The results on CREMAD dataset are reported in Table 2. From Table 2, we can find that: (1) Both objectives in Equation (2), (3), and (4) can boost multimodal performance in terms of accuracy and MAP; (2) While the unimodal performance of the method using all objectives may not always reach the highest level, it achieves a more balanced classification performance across modalities.

We further investigate the impact of residual learning on classification performance by comparing the performance of all t classifiers with that of the first t − 1 classifiers during the training. The results are presented in Figure 3, where the former accuracy is denoted as "Full Prediction" and the latter is denoted as "Prediction of t − 1 CLS". In Figure 3, we also present the number of video classifiers. We observe that the number of classifiers for the video modality has increased, and the performance of all t classifiers is generally superior to that of the first t − 1 classifiers. This performance gain arises from our learning of the residual objective.

## 4.5 Further Analysis

Impact of Weak Classifier Assignment Strategy: We conduct an experiment to study the influence of adaptive classifier assignment strategy. Specifically, we design a fixed classifier assignment strategy for comparison. This approach allocates n^{fix} classifiers for weak modality during the init stage. And we no longer dynamically adjust the number of classifiers during training for weak modality.

The results on CREMAD dataset are reported in Table 3, where n^{fix} is set to be 10 and 12. The results in Table 3 demonstrate that our proposed adaptive classifier assignment strategy can boost performance compared with fixed classifier strategy. This is because our method dynamically adjusts modality classification performance in response to modality imbalance during training.

Visualization Results: We further study the property of embeddings through visualization. Specifically, we illustrate the t-SNE [39] results on CREMAD dataset for naive multimodal learning (naive MML), ReconBoost [20], and our method in Figure 4. From Figure 4, we can find that: (1) Compared to naive MML, our method and ReconBoost can learn more discriminative multimodal features, as both approaches enhance the weak modality using information from the strong modality; (2) Compared to ReconBoost, our method demonstrates significantly superior classification performance on the video modality, with several distinct categories highlighted by circle markers in Figure 4 (f).


![Figure 27](image_27.jpg)
<!-- Caption: Experiments, index 27 -->



![Figure 28](image_28.jpg)
<!-- Caption: Experiments, index 28 -->


(a) Naive MML@Audio | (b) ReconBoost@Audio | (c) Ours@Audio

(d) Naive MML@Video | (e) ReconBoost@Video | (f) Ours@Video

Figure 4: Visualization on CREMAD dataset. The video visualization highlights the need to improve weak-modality classification.

Figure 5: Training time (hrs) | Figure 6: Loss visualization | Figure 7: Accuracy change

This improvement is primarily attributed to our explicit enhancement of the classification capabilities of the weaker modality.

Training Overhead: Considering that our method introduces additional computational overhead due to the sustained boosting strategy, we compare its training cost with that of competitive SOTA baselines, including Naive, PMR, AGM, MLA, and ReconBoost, through empirical experiments under the same setting. The results are shown in Figure 5. It can be observed that our method achieves the best accuracy while maintaining competitive training time.

Convergence of Gap Loss: To further validate the convergence of the gap function, we conducted experiments on the CREMAD dataset. The change of the gap function 𝒢(Φ) during the training process is reported in Figure 6, where we also report the unimodal loss and multimodal loss. It can be observed that as training progresses, the gap function gradually converges. Moreover, we report the accuracy changes during training in Figure 7. From Figure 7, it can be seen that the accuracy exhibits a similar convergence trend.

Impact of the Model Capacity: Since our method utilizes multiple designed classifiers for the weak modality, it essentially uses more model parameters than the baselines. Does our method benefit solely from having more model parameters? We conduct an experiment to verify this. Specifically, we replace the backbone ResNet18 (R18) of the weak-modality video with a larger backbone, i.e., ResNet34 (R34). Then we run the baselines and our method on the CREMAD dataset. The results are shown in Table 4. From Table 4, we find that our approach improves accuracy by nearly 20% with only an additional 1M parameters compared to naive MML with the same network architecture. Our method also outperforms the naive MML baseline with a larger backbone. Furthermore, we observe an


![Figure 34](image_34.jpg)
<!-- Caption: Experiments, index 34 -->



![Figure 35](image_35.jpg)
<!-- Caption: Experiments, index 35 -->



![Figure 36](image_36.jpg)
<!-- Caption: Experiments, index 36 -->



![Figure 37](image_37.jpg)
<!-- Caption: Experiments, index 37 -->



![Figure 38](image_38.jpg)
<!-- Caption: Experiments, index 38 -->



![Figure 39](image_39.jpg)
<!-- Caption: Experiments, index 39 -->



![Figure 40](image_40.jpg)
<!-- Caption: Experiments, index 40 -->



![Figure 41](image_41.jpg)
<!-- Caption: Experiments, index 41 -->



![Figure 42](image_42.jpg)
<!-- Caption: Experiments, index 42 -->


| Method   | Arch.<br/>Audio | Arch.<br/>Video | #params.<br/>Audio | #params.<br/>Video | Acc.   |
| -------- | --------------- | --------------- | ------------------ | ------------------ | ------ |
| Unimodal | -               | R18             | -                  | 11.8M              | 0.4718 |
| Unimodal | -               | R34             | -                  | 23.3M              | 0.4731 |
| Naive    | R18             | R18             | 11.8M              | 11.8M              | 0.6507 |
| Naive    | R18             | R34             | 11.8M              | 23.3M              | 0.6277 |
| Ours     | R18             | R18             | 11.8M              | 12.8M              | 0.8515 |

interesting and counterintuitive phenomenon. That is, the method with ResNet34 is worse than that with ResNet18. The reason behind this may be that the ResNet34-based method is more difficult to converge.

| Method     | rate = 0% | rate = 20% | rate = 50% |
| ---------- | --------- | ---------- | ---------- |
| Naive MML  | 0.6507    | 0.5849     | 0.5242     |
| ReconBoost | 0.7557    | 0.6321     | 0.5568     |
| MLA        | 0.7943    | 0.6935     | 0.5753     |
| Ours       | 0.8515    | 0.7540     | 0.6008     |

Stability under Modality Missing: Our method is adaptable to scenarios involving missing modalities. We comprehensively evaluate its performance under test-time missing modality conditions. Specifically, test-time missing refers to cases where the modalities are complete during training but missing during the testing phase. The experiments are conducted on CREMAD dataset with different missing rate [50]. Naive MML, ReconBoost [20], and MLA [50] are selected as baselines for comparison, where MLA introduces specifically designed algorithms to address the corresponding challenges in modality missing. We report the results with missing rate 20% and 50% in Table 5. We can find that as the modality missing rate increases, the performance of all methods declines. Nevertheless, our method consistently achieves the best performance under all missing rates, demonstrating the effectiveness of our method in scenario with missing modality.


## Conclusion


## 5 Conclusion

To address the modality imbalance issue, we propose a novel multimodal learning approach by designing a sustained boosting algorithm to dynamically enhance the classification ability of weak modality. Concretely, we first propose a sustained boosting algorithm for multimodal learning by minimizing the classification and residual errors simultaneously. Then, we propose an adaptive classifier assignment strategy to dynamically facilitate the classification ability of weak modality. The effectiveness of the proposed boosting algorithm is theoretically guaranteed by analyzing the convergence properties of the cross-modal gap function. To this end, the classification ability can be rebalanced adaptively during the training procedure. Experiments on widely used datasets reveal that our proposed method can achieve state-of-the-art performance compared with various baselines by a large margin.

Limitations: Our proposed method mainly focuses on the classifier of each modality and extends to early fusion MML scenarios. However, we do not analyze potential ethical concerns such as biases introduced by imbalanced modality representations, risks arising from missing modality data in deployment, or fairness considerations. Future work should include explicit examination of these aspects to ensure responsible multimodal learning applications.


## Appendix


## Acknowledgments and Disclosure of Funding

This work was supported in part by the National Key R&D Program of China (2022YFF0712100), in part by the NSFC (62276131, 62506168), in part by the Natural Science Foundation of Jiangsu Province of China under Grant (BK20240081, BG2024042), and in part by the Fundamental Research Funds for the Central Universities (No.30925010205).


## References


## References

[1] A. F. Agarap. Deep learning using rectified linear units (relu). CoRR, abs/1803.08375, 2018.

[2] R. Arandjelovic and A. Zisserman. Look, listen and learn. In ICCV, pages 609–617. IEEE, 2017.

[3] Y. Cai, H. Cai, and X. Wan. Multi-Modal Sarcasm Detection in Twitter with Hierarchical Fusion Model. In *ACL*, pages 2506–2515, 2019.

[4] H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, and R. Verma. CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset. *TAC*, 5(4):377–390, 2014.

[5] J. Carreira and A. Zisserman. Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. In *CVPR*, pages 4724–4733. Computer Vision Foundation / IEEE, 2017.

[6] H. Chen, W. Xie, A. Vedaldi, and A. Zisserman. VGGSound: A Large-Scale Audio-Visual Dataset. In *ICASSP*, pages 721–725. IEEE, 2020.

[7] J. Chen, H. Mao, W. L. Woo, and X. Peng. Deep Multiview Clustering by Contrasting Cluster Assignments. In *ICCV*, pages 16706–16715. IEEE, 2023.

[8] C. Cortes, M. Mohri, and U. Syed. Deep Boosting. In *ICML*, pages 1179–1187. PMLR, 2014.

[9] J. Devlin, M. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In *NAACL*, pages 4171–4186. Association for Computational Linguistics, 2019.

[10] W. et al. What Makes Training Multi-Modal Classification Networks Hard? In *CVPR*, 2020.

[11] Y. Fan, W. Xu, H. Wang, J. Liu, and S. Guo. Detached and Interactive Multimodal Learning. In *ACM MM*. ACM, 2024.

[12] Y. Fan, W. Xu, H. Wang, J. Wang, and S. Guo. PMR: Prototypical Modal Rebalance for Multimodal Learning. In *CVPR*, pages 20029–20038. Computer Vision Foundation / IEEE, 2023.

[13] Y. Freund and R. E. Schapire. A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting. In *EuroCOLT*, volume 904 of *Lecture Notes in Computer Science*, pages 23–37. Springer, 1995.

[14] J. H. Friedman. Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, pages 1189–1232, 2001.

[15] N. Fujimori, R. Endo, Y. Kawai, and T. Mochizuki. Modality-Specific Learning Rate Control for Multimodal Classification. In *ACPR*, volume 12047, pages 412–422, 2019.

[16] R. Gao, T. Oh, K. Grauman, and L. Torresani. Listen to Look: Action Recognition by Previewing Audio. In *CVPR*, pages 10454–10464. Computer Vision Foundation / IEEE, 2020.

[17] Y. Ge, J. Ren, A. Gallagher, Y. Wang, M. Yang, H. Adam, L. Itti, B. Lakshminarayanan, and J. Zhao. Improving Zero-Shot Generalization and Robustness of Multi-Modal Models. In *CVPR*, pages 11093–11101. Computer Vision Foundation / IEEE, 2023.

[18] K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image Recognition. In *CVPR*, pages 770–778. Computer Vision Foundation / IEEE, 2016.

[19] D. Hu, X. Li, and X. Lu. Temporal Multimodal Learning in Audiovisual Speech Recognition. In *CVPR*, pages 3574–3582. Computer Vision Foundation / IEEE, 2016.

[20] C. Hua, Q. Xu, S. Bao, Z. Yang, and Q. Huang. ReconBoost: Boosting Can Achieve Modality Reconcilement. In *ICML*. PMLR, 2024.

[21] Y. Huang, C. Du, Z. Xue, X. Chen, H. Zhao, and L. Huang. What Makes Multi-Modal Learning Better than Single-Modal Learning? (Provably). In *NeurIPS*, pages 10944–10956, 2021.

[22] Y. Huang, J. Lin, C. Zhou, H. Yang, and L. Huang. Modality Competition: What Makes Joint Training of Multi-Modal Networks Fail in Deep Learning? (Provably). In *ICML*, volume 162 of *Proceedings of Machine Learning Research*, pages 9226–9259. PMLR, 2022.

[23] C. Jia, Y. Yang, Y. Xia, Y. Chen, Z. Parekh, H. Pham, Q. V. Le, Y. Sung, Z. Li, and T. Duerig. Scaling Up Visual and Vision-Language Representation Learning with Noisy Text Supervision. In *ICML*, pages 4904–4916. PMLR, 2021.

[24] Q. Jiang, Z. Chi, and Y. Yang. Interactive Multimodal Learning via Flat Gradient Modification. In *IJCAI*, pages 5489–5497. ijcai.org, 2025.

[25] D. P. Kingma and J. Ba. Adam: A Method for Stochastic Optimization. In *ICLR*. OpenReview.net, 2015.

[26] H. Li, X. Li, P. Hu, Y. Lei, C. Li, and Y. Zhou. Boosting Multi-Modal Model Performance with Adaptive Gradient Modulation. In *ICCV*, pages 22157–22167. IEEE, 2023.

[27] P. Liu, S. Han, Z. Meng, and Y. Tong. Facial expression recognition via a boosted deep belief network. In CVPR, pages 1805–1812. Computer Vision Foundation / IEEE, 2014.
[28] F. Lv, X. Chen, Y. Huang, L. Duan, and G. Lin. Progressive modality reinforcement for human multimodal emotion recognition from unaligned multimodal sequences. In CVPR, pages 2554–2562. Computer Vision Foundation / IEEE, 2021.
[29] P. Molchanov, X. Yang, S. Gupta, K. Kim, S. Tyree, and J. Kautz. Online detection and classification of dynamic hand gestures with recurrent 3D convolutional neural networks. In CVPR, pages 4207–4215. Computer Vision Foundation / IEEE, 2016.
[30] A. Natekin and A. C. Knoll. Gradient boosting machines: a tutorial. Frontiers in Neurorobotics, 7:21, 2013.
[31] J. Ngiam, A. Khosla, M. Kim, J. Nam, H. Lee, and A. Y. Ng. Multimodal deep learning. In ICML, pages 689–696. Omnipress, 2011.
[32] W. Nie, Y. Yan, D. Song, and K. Wang. Multi-modal feature fusion based on multi-layer LSTM for video emotion recognition. MTA, 80(11):16205–16214, 2021.
[33] X. Peng, Y. Wei, A. Deng, D. Wang, and D. Hu. Balanced multimodal learning via on-the-fly gradient modulation. In CVPR, pages 8228–8237. Computer Vision Foundation / IEEE, 2022.
[34] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. C. Courville. FiLM: Visual reasoning with a general conditioning layer. In AAAI, pages 3942–3951. AAAI Press, 2018.
[35] L. O. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin. CatBoost: Unbiased boosting with categorical features. In NeurIPS, pages 6639–6649, 2018.
[36] K. Simonyan and A. Zisserman. Two-stream convolutional networks for action recognition in videos. In NeurIPS, pages 568–576, 2014.
[37] A. Singh, R. Hu, V. Goswami, G. Couairon, W. Galuba, M. Rohrbach, and D. Kiela. FLAVA: A foundational language and vision alignment model. In CVPR, pages 15617–15629. Computer Vision Foundation / IEEE, 2022.
[38] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In CVPR, pages 2818–2826. IEEE Computer Society, 2016.
[39] L. van der Maaten and G. Hinton. Visualizing data using t-SNE. JMLR, 9:2579–2605, 2008.
[40] W. Wang, D. Tran, and M. Feiszli. What makes training multi-modal classification networks hard? In CVPR, pages 12692–12702. Computer Vision Foundation / IEEE, 2020.
[41] Y. Wei, R. Feng, Z. Wang, and D. Hu. Enhancing multimodal cooperation via sample-level modality valuation. In CVPR, pages 27338–27347. Computer Vision Foundation / IEEE, 2024.
[42] Y. Wei and D. Hu. MMPareto: Boosting multimodal learning with innocent unimodal assistance. In ICML, Proceedings of Machine Learning Research. PMLR, 2024.
[43] N. Wu, S. Jastrzebski, K. Cho, and K. J. Geras. Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks. In ICML, volume 162, pages 24043–24055. PMLR, 2022.
[44] Y. Yang, H. Pan, Q. Jiang, Y. Xu, and J. Tang. Learning to rebalance multi-modal optimization by adaptively masking subnetworks. TPAMI, 47(6):4553–4566, 2025.
[45] Y. Yang, F. Wan, Q. Jiang, and Y. Xu. Facilitating multimodal classification via dynamically learning modality gap. In NeurIPS, 2024.
[46] Y. Yang, K. Wang, D. Zhan, H. Xiong, and Y. Jiang. Comprehensive semi-supervised multi-modal learning. In IJCAI, pages 4092–4098. ijcai.org, 2019.
[47] Y. Yang, H. Ye, D. Zhan, and Y. Jiang. Auxiliary information regularized machine for multiple modality feature learning. In IJCAI, pages 1033–1039. AAAI Press, 2015.
[48] Y. Yao and R. Mihalcea. Modality-specific learning rates for effective multimodal additive late-fusion. In ACL, pages 1824–1834, 2022.
[49] J. Yu and J. Jiang. Adapting BERT for target-oriented multimodal sentiment classification. In IJCAI, pages 5408–5414. ijcai.org, 2019.
[50] X. Zhang, J. Yoon, M. Bansal, and H. Yao. Multimodal representation learning by alternating unimodal adaptation. In CVPR, pages 27456–27466. IEEE, 2024.


## Appendix


A Proof

During training process, the gradient boosting strategy will be applied when the performance gap is larger than a threshold at each c iterations. Now we analyze the convergence rate of 𝓖 with gradient boosting strategy.

Furthermore, we make the following assumption:

Assumption 1 (Strongly Convexity) Although deep neural network loss surfaces are known to be nonconvex, for theoretical tractability, we simplify the analysis by assuming that the functions {𝓛^a(·), 𝓛^v(·)} exhibit locally strongly convex behavior in neighborhoods of interest. This is a strong abstraction justified heuristically but not rigorously guaranteed for all training stages.

Similarly, smoothness assumptions hold approximately in local regions but do not extend globally for deep models.

Assumption 2 (Smoothness) We assume that the functions {𝓖(·), 𝓛^a(·), 𝓛^v(·)} are L-smooth. That is, for any Φ, Φ′, we have
∀ o ∈ {a, v}, 𝓛^o(Φ) − 𝓛^o(Φ′) ≤ ⟨∇𝓛^o(Φ′), Φ − Φ′⟩ + (L_o/2)‖Φ − Φ′‖^2. (11)
and
𝓖(Φ) − 𝓖(Φ′) ≤ ⟨∇𝓖(Φ′), Φ − Φ′⟩ + (L_g/2)‖Φ − Φ′‖^2. (12)

Assumption 3 (Effectiveness of Weak Classifier) We assume that the gradient boosting strategy for weak modality is effective. That is, there exists a ν ∈ (0, 1):
⟨∇𝓛^a(Φ(t)), h(Φ(t))⟩ ≤ −ν‖∇𝓛^a(Φ(t))‖^2. (13)
And there exists a constant β > 0:
‖h(Φ(t))‖ ≤ β‖∇𝓛^a(Φ(t))‖. (14)
where h(·) denotes the function obtained by the boosting algorithm. h(·) aims to fit the negative gradient of the weak modality model.

Based on the definition of h(·), the updating rule can be formed as:
Φ(t + 1) = Φ(t) − η∇𝓛(Φ(t)) (15)
= Φ(t) + η h(Φ(t)) (16)
where η denotes the learning rate.

Assumption 4 (Optimal of Multimodels) We assume that there exists an optimal solution Φ* so that we have: ∇𝓛^a(Φ*) = ∇𝓛^v(Φ*) = 0.

Assumption 5 (Bounded Initial Value of 𝓖(Φ(t))) We assume that the initial value of 𝓖(Φ(t)) is bounded, i.e., 𝓖(Φ(0)) < ∞.

Based on Assumption (1), (4), we have the following lemma:

Lemma 1 (Gap Bound) There exists a constant κ such that the loss gap function 𝓖(·) and the gradient norm of the strong modality satisfy the following relationship:
‖∇𝓛^a(Φ(t))‖ ≥ κ |𝓖(Φ(t))|. (17)

Proof 1 (Proof of Lemma 1) Since 𝓛^a(·) and 𝓛^v(·) are strongly convex, there exists a constant ρ ∈ (0, 1) satisfying:
‖Φ(t) − Φ*‖ ≤ ρ^t ‖Φ(0) − Φ*‖. (18)

By smoothness, we have:
$$\forall o \in \{a,v\},\ \mathcal{L}^o(\Phi(t)) - \mathcal{L}^o(\Phi^*) \le \frac{L_o}{2}\|\Phi(t) - \Phi^*\|^2 \tag{19}$$
$$\le \frac{L_o}{2}\,\rho^{2t}\,\|\Phi(0) - \Phi^*\|^2 \tag{20}$$

Then we have:
$$\mathcal{G}(\Phi(t)) = \mathcal{L}^a(\Phi(t)) - \mathcal{L}^v(\Phi(t)) \tag{21}$$
$$= \mathcal{L}^a(\Phi(t)) - \mathcal{L}^a(\Phi^*) - \big(\mathcal{L}^v(\Phi(t)) - \mathcal{L}^v(\Phi^*)\big) \tag{22}$$
$$\le \big|\mathcal{L}^a(\Phi(t)) - \mathcal{L}^a(\Phi^*)\big| + \big|\mathcal{L}^v(\Phi(t)) - \mathcal{L}^v(\Phi^*)\big| \tag{23}$$
$$\le \left(\frac{L_a}{2} + \frac{L_v}{2}\right)\rho^{2t}\,\|\Phi(0) - \Phi^*\|^2 \tag{24}$$
$$\le \left(\frac{L_a}{2} + \frac{L_v}{2}\right)\|\Phi(0) - \Phi^*\|^2 \tag{25}$$

By setting $$c = \left(\frac{L_a}{2} + \frac{L_v}{2}\right)\|\Phi(0) - \Phi^*\|^2,$$ we have:
$$|\mathcal{G}(\Phi(t))| \le c \tag{26}$$
Then we have:
$$|\mathcal{G}(\Phi(t))| \ge \frac{1}{c}\,|\mathcal{G}(\Phi(t))|^2 \tag{27}$$

According to Assumption (1), we have:
$$\|\nabla \mathcal{L}^a(\Phi(t))\|^2 \ge \mu_a\,\big|\mathcal{L}^a(\Phi(t)) - \mathcal{L}^a(\Phi^*)\big| \tag{28}$$

Suppose that two modalities achieve the same optimal value, i.e., $$\mathcal{L}^a(\Phi^*) = \mathcal{L}^v(\Phi^*).$$ Then we have:
$$\|\nabla \mathcal{L}^a(\Phi(t))\|^2 \ge 2\mu\,\big|\mathcal{L}^a(\Phi(t)) - \mathcal{L}^a(\Phi^*)\big| \tag{29}$$
$$= 2\mu\,\big|\mathcal{L}^a(\Phi(t)) - \mathcal{L}^v(\Phi^*)\big| \tag{30}$$
$$\ge 2\mu\,\big|\mathcal{L}^a(\Phi(t)) - \mathcal{L}^v(\Phi(t))\big| \tag{31}$$
$$= 2\mu\,|\mathcal{G}(\Phi(t))| \tag{32}$$
$$\ge \frac{2\mu}{c}\,|\mathcal{G}(\Phi(t))|^2 \tag{33}$$

By setting $$\kappa = \sqrt{\frac{2\mu}{c}},$$ we have:
$$\|\nabla \mathcal{L}^a(\Phi(t))\| \ge \kappa\,|\mathcal{G}(\Phi(t))| \tag{34}$$

We have the following theorem:

Theorem 1 (Convergence of $\mathcal{G}$ with Gradient Boosting). Under assumption (2) and (3), if the learning rate is set as $$\eta_t = \frac{\nu}{L_a \beta^2},$$ we have:
$$\mathcal{G}(\Phi(T)) \le \frac{\mathcal{G}(\Phi(0))}{1 + d\,T\,\mathcal{G}(\Phi(0))} = O\!\left(\frac{1}{T}\right), \tag{35}$$
where $$d = \frac{\nu^2 \kappa^2}{2 L_a \beta^2}.$$

Proof 2 (Proof of Theorem 1). Based on the smoothness assumption, we have:
$$\mathcal{L}^a(\Phi(t+1)) \le \mathcal{L}^a(\Phi(t)) + \langle \nabla \mathcal{L}^a(\Phi(t)), \Phi(t+1) - \Phi(t) \rangle + \frac{L_a}{2}\|\Phi(t+1) - \Phi(t)\|^2, \tag{36}$$
$$= \mathcal{L}^a(\Phi(t)) + \eta_t \langle \nabla \mathcal{L}^a(\Phi(t)), h(\Phi(t)) \rangle + \frac{L_a \eta_t^2}{2}\|h(\Phi(t))\|^2, \tag{37}$$
$$\stackrel{(13)}{\le} \mathcal{L}^a(\Phi(t)) - \nu \eta_t \|\nabla \mathcal{L}^a(\Phi(t))\|^2 + \frac{L_a \eta_t^2}{2}\|h(\Phi(t))\|^2, \tag{38}$$
$$\stackrel{(14)}{\le} \mathcal{L}^a(\Phi(t)) - \nu \eta_t \|\nabla \mathcal{L}^a(\Phi(t))\|^2 + \frac{L_a \eta_t^2 \beta^2}{2}\|\nabla \mathcal{L}^a(\Phi(t))\|^2. \tag{39}$$

Considering that the parameters for each modality are independent, we have: 𝓛^v(Φ(t + 1)) = 𝓛^v(Φ(t)). Then we have:

𝓖(Φ(t + 1)) ≤ 𝓖(Φ(t)) − (νη_t − (L_aη_t^2β^2)/2)‖∇𝓛^a(Φ(t))‖^2 (40)

By setting η_t^a = ν/(L_aβ^2), we have:

𝓖(Φ(t + 1)) ≤ 𝓖(Φ(t)) − (ν^2/(2L_aβ^2))‖∇𝓛^a(Φ(t))‖^2 (41)

By using Lemma (1), we have:

𝓖(Φ(t + 1)) ≤ 𝓖(Φ(t)) − (ν^2κ^2/(2L_aβ^2))|𝓖(Φ(t))|^2. (42)

Then we set d = ν^2κ^2/(2L_aβ^2):

1/𝓖(Φ(t + 1)) ≥ 1/(𝓖(Φ(t)) − d|𝓖(Φ(t))|^2) (43)
≥ (1/𝓖(Φ(t)))(1 + d𝓖(Φ(t))) (44)
= 1/𝓖(Φ(t)) + d (45)

By summing up for k = 1, ···, T, we have:

𝓖(Φ(T)) ≤ 𝓖(Φ(0)) / (1 + dT𝓖(Φ(0))) = 𝒪(1/T). (46)

B. Notation Definition

We summarize the notation definition we used in this paper in Table I.

C. Additional Experiments

C.1 Datasets

CREMAD: CREMAD [4] is an audio-visual dataset designed for speech emotion recognition. It comprises 7,442 video clips of 2–3 seconds from 91 actors speaking several short words, which are divided into 6,698 training samples and 744 testing samples. This dataset includes the six most common emotions: angry, happy, sad, neutral, disgust and fear.

KSounds: KSounds [2] dataset is a commonly used dataset containing 31 action categories that can be recognized visually and auditorily, and contains 19,000 10-second video clips from YouTube. This dataset is divided into a training set with 15,000 clips, a validation set with 1,900 clips, and a testing set with 1,900 clips.

NVGesture: NVGesture [29] dataset is a multimodal dataset specifically designed for gesture recognition, containing three types of data modalities, i.e., RGB, optical flow (OF), and Depth. This dataset includes 25 different gesture categories, covering a variety of common gestures. This dataset is divided into 1,050 samples for training and 482 samples for testing.

VGGSound: VGGSound [6] dataset is an audio-visual dataset in the wild, with nearly 200K 10-second video clips. Each sound-emitting object is also visible in the corresponding video clip in this dataset. After filtering out unavailable videos, there are 168,618 videos for training and validation, and 13,954 videos for testing in our experimental settings.

Twitter: Twitter [49] dataset is a dataset designed for multilingual sentiment analysis, which is divided into a training set with 3,197 pairs, a validation set with 1,122 pairs and a testing set with 1,037 pairs.

Table I: Notation Definition

- N: The number of training data.
- K: The number of category labels.
- \mathbf{x}^a / \mathbf{x}^v: Audio/video data point.
- \mathbf{X} = \{(\mathbf{x}_i^a, \mathbf{x}_i^v)\}_{i=1}^N: Training set.
- y_i \in \{0, 1\}^K: Category label of the i-th data.
- \mathbf{Y} = \{y_i\}_{i=1}^N: Category label set.
- Superscript o: o-modality.
- Superscript a/v: Audio/video modality.
- \phi^o(\cdot): Encoder of o-modality.
- \theta^o: Parameters of \phi^o(\cdot).
- \mathbf{u}^o: Feature of \mathbf{x}^o.
- \psi^o(\cdot): Classifier of o-modality.
- \Theta^o: Parameters of \psi^o(\cdot).
- \mathbf{p}^o: Prediction of \mathbf{x}^o.
- \Phi^o = \{\theta^o, \Theta^o\}: Parameter set.
- \ell(\cdot): Cross-entropy loss.
- \mathcal{L}_{\text{CE}}(\cdot): Cross-entropy loss over training.
- \lambda: Hyperparameter used to soften labels.
- \hat{y}_{it}^o: Residual label of \mathbf{x}_i^o at the t-th iteration.
- \epsilon(\cdot): Classification loss.
- \epsilon_{\text{all}}(\cdot): Loss for t classifiers.
- \epsilon_{\text{pre}}(\cdot): Loss for t - 1 classifiers.
- L(\cdot): Objective function.
- \mathcal{L}_{\text{SUB}}(\cdot): Overall objective function.
- n^o: The number of classifiers for o-modality.
- s_t^o: Confidence for o-modality at the t-th iteration.
- \sigma: Coefficient of confidence score.
- \tau: Tolerance score.
- t_N: Adjustment frequency.
- \mathcal{G}(\cdot): Gap function.

This dataset consists of tweets collected from Twitter^2 with three different sentiment labels: positive, negative, and neutral.

Sarcasm: The Sarcasm [3] dataset is specifically designed for sarcasm detection tasks, containing a large amount of text data labeled as sarcastic or non-sarcastic, typically drawn from various sources such as social media, news comments, and conversation data, with each entry clearly labeled for sarcasm. This dataset includes 19,816 pairs for the training set, 2,410 pairs for the validation set, and 2,409 pairs for the testing set.

C.2 Implementations of Toy Experiments

The toy experiment is designed to validate the motivation behind our method. Specifically, it is conducted on the CREMAD dataset, where the task is multimodal classification. Four methods are adopted for comparison: Naive MML, G-Blend [10], MML w/ GB, and Ours. Here, MML w/ GB refers to applying the gradient boosting algorithm to further improve the trained video model obtained from Naive MML, while keeping the audio model fixed. Ours refers to the method employing the sustained boosting algorithm proposed in this paper. Furthermore, ResNet-18 is used as the encoder for all methods, with its parameters randomly initialized. The selection of other hyperparameters is kept consistent with the main experiment.

^2 http://twitter.com

C.3 Influence of Unstable Confidence Score

Since our method relies on the confidence gap to determine whether to add a classifier, fluctuations in confidence scores are indeed an influential factor. We conduct an experiment to explore the fluctuation of confidence score in a scenario with modality noise on the CREMA-D dataset. Specifically, we add Gaussian noise to the video modality to introduce instability in confidence scores and investigate its impact on the algorithm. We design a metric to evaluate the fluctuation of confidence score during training. We first compute the absolute difference in confidence scores between two consecutive iterations, and then take the average over all training rounds:
∀o ∈ {a, v}, s̄^o = (1/n) ∑_{t=2}^{n} |s_t^o − s_{t−1}^o|. (47)
Here, s_t^o denotes the confidence score of o-modality at the t-th iteration, and n denotes the current iteration index in the training process. A higher value of s̄^o indicates more severe fluctuations in confidence scores, reflecting a sharper and less stable prediction behavior. Furthermore, we compare our method with naive MML and ReconBoost [20].

We first compare the model performance in a scenario with modality noise in Table II, where ζ denotes the noise rate, “#CLS” denotes the number of classifiers during training, and in the first column, we also report the fluctuation of confidence score. In fact, as training progressed, we observe that in noisy scenarios, increasing the number of classifiers beyond a certain point no longer led to improved performance on the validation set. This phenomenon became more pronounced as the noise level increased. In other words, the classifiers may overcompensate in scenarios involving modality noise. On the other hand, we also observe that after reaching its optimal performance, the model’s performance gradually degraded without a drastic drop. Therefore, in practical scenarios, an early stopping strategy based on validation set performance can be employed to select the optimal model and avoid classifier overcompensation.

| #CLS | ζ = 0, (s̄ = 0.1428) | ζ = 20, (s̄ = 0.3920) | ζ = 50, (s̄ = 0.4760) |
| ---- | -------------------- | --------------------- | --------------------- |
| 1    | 0.1411               | 0.1411                | 0.2863                |
| 2    | 0.6559               | 0.6129                | 0.6398                |
| 3    | 0.7634               | 0.7608                | 0.7728                |
| 4    | 0.8038               | 0.7917                | 0.8185                |
| 5    | 0.8266               | 0.8118                | 0.8212                |
| 6    | 0.8306               | 0.8293                | 0.8253                |
| 7    | 0.8401               | 0.8333                | 0.8199                |
| 8    | 0.8441               | 0.8266                | 0.8239                |
| 9    | 0.8468               | 0.8293                | 0.8185                |
| 10   | 0.8515               | 0.8280                | 0.8145                |

Furthermore, we compare the performance with competitive baselines including naive MML and ReconBoost [20]. The results are shown in Table III. We can find that compared with the competitive method ReconBoost, our approach consistently achieves superior performance, demonstrating its effectiveness in a scenario with modality noise.

| Method     | ζ = 0      | ζ = 20     | ζ = 50     |
| ---------- | ---------- | ---------- | ---------- |
| Naive MML  | 0.6507     | 0.6425     | 0.6331     |
| ReconBoost | 0.7557     | 0.7215     | 0.7031     |
| Ours       | 0.8515     | 0.8333     | 0.8253     |

## C.4 Computational and Memory Cost at Inference Stage

To demonstrate the practical applicability of our approach, we further provide a comprehensive analysis of its computational and memory overhead at inference stage.^3 Furthermore, we conduct an experiment to analyze the computational and memory cost during the inference phase on the CREMA-D dataset. The baselines include naive MML, PMR [12], AGM [26], MLA [50], ReconBoost [20]. The results are shown in Table IV. We can find that our method achieves superior performance while maintaining competitive inference time. Furthermore, compared with naive MML, our method introduces only 1M additional parameters when 10 classifiers are added. Given the total model size of 23.6M, this corresponds to a relatively small increase of approximately 4%.

| Method     | Accuracy | Inference time (s) |
| ---------- | -------- | ------------------ |
| Naive MML  | 0.6507   | 5.29               |
| PMR        | 0.6659   | 6.59               |
| AGM        | 0.6733   | 5.58               |
| MLA        | 0.7943   | 5.63               |
| ReconBoost | 0.7557   | 5.33               |
| Ours       | 0.8515   | 5.62               |

^3 For computational cost during training, the analysis has been presented in Section 4.5 of the original paper.

## C.5 Selection of Score Function

In our method, the score function is employed to quantify the disparity in classification performance between different modalities, which serves as the basis for determining whether to introduce an additional classifier for the weak modality. Therefore, any metric that effectively captures the difference in classification capabilities across modality-specific models can be adopted as a score function.

To some extent, entropy and loss can reflect the learning status of modality-specific models and thereby serve as proxies for their classification capabilities. To evaluate the effectiveness of using entropy and loss as score functions, we conducted experiments on the CREMA-D dataset, where these two metrics were used to guide the adaptive classifier assignment. The experimental results are presented in Table V, where τ denotes the threshold. Please note that since we use the ratio between different modal metrics to compare against the threshold τ, the value of τ remains consistent across different metric types such as confidence, loss, and entropy. These findings indicate that the proposed method is robust to the choice of score function, consistently achieving comparable results regardless of whether confidence score, entropy, or loss is used.

| Score function          | Accuracy | Threshold τ | #CLS |
| ----------------------- | -------- | ----------- | ---- |
| Confidence score (Ours) | 0.8515   | 0.01        | 10   |
| Entropy                 | 0.8522   | 0.01        | 10   |
| Loss                    | 0.8562   | 0.01        | 10   |

| #CLS | Accuracy | Training time (hrs) | Inference time (s) |
| ---- | -------- | ------------------- | ------------------ |
| 2    | 0.8159   | 1.68                | 5.31               |
| 4    | 0.8387   | 1.72                | 5.38               |
| 6    | 0.8441   | 1.78                | 5.46               |
| 8    | 0.8468   | 1.86                | 5.53               |
| 10   | 0.8515   | 1.98                | 5.62               |

C.6 Computational Cost vs. The Number of Classifiers

The computational overhead—including both training and inference time—is indeed influenced by the number of classifiers. Since our method dynamically adds classifiers during training, we incorporate a threshold to limit the maximum number of classifiers. This allows us to systematically investigate the trade-off between computational cost and model performance as a function of classifier quantity.

Specifically, we conduct experiments on the CREMA-D dataset, where the maximum number of weak modality classifiers is constrained to not exceed a predefined threshold M. We vary M across the set {2, 4, 6, 8, 10}, and for each setting, we report the training time, inference time, and the corresponding classification performance.

The experimental results are summarized in Table VI. We observe that as the number of classifiers increases, both training time and inference time increase accordingly, while performance also improves to a certain extent. This indicates that incorporating more classifiers can indeed enhance model performance, but at the cost of increased computational overhead.