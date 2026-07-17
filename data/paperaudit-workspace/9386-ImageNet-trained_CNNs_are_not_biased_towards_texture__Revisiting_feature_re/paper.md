
## Abstract


ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression
Tom Burgert1,2, Oliver Stoll1,2, Paolo Rota3, Begüm Demir1,2
BIFOLD1, TU Berlin2, University of Trento3
{t.burgert,o.stoll,demir}@tu-berlin.de, paolo.rota@unitn.it

Abstract
The hypothesis that Convolutional Neural Networks (CNNs) are inherently texture-biased has shaped much of the discourse on feature use in deep learning. We revisit this hypothesis by examining limitations in the cue-conflict experiment by Geirhos et al. To address these limitations, we propose a domain-agnostic framework that quantifies feature reliance through systematic suppression of shape, texture, and color cues, avoiding the confounds of forced-choice conflicts. Our evaluation under controlled suppression conditions suggests that CNNs often rely more on local shape features than texture, challenging the prevailing notion of an inherent texture bias in many standard architectures and training regimes. Nonetheless, this reliance can be substantially mitigated through modern training strategies or architectures (ConvNeXt, ViTs). We further extend the analysis across computer vision, medical imaging, and remote sensing, revealing that reliance patterns differ systematically: computer vision models prioritize shape, medical imaging models emphasize color, and remote sensing models exhibit a stronger reliance on texture. Code is available at https://github.com/tomburgert/feature-reliance.


## Introduction


1 Introduction
Convolutional neural networks (CNNs) have played a central role in the development of deep learning models for visual recognition [1], [2], [3], [4]. Their success across a range of computer vision (CV) benchmarks has contributed to the perception that they acquire perceptual representations resembling those of humans [5], [6], [7]. However, a growing body of work suggests that CNNs may process visual information in fundamentally different ways [8], [9], [10]. Much prior work, including influential studies [8], argued CNNs are biased towards texture features in contrast to human shape reliance. However, growing evidence and the present work challenge this binary view, proposing CNN feature reliance is more nuanced and context-dependent than previously assumed.

In the cue-conflict experiment, images are synthesized by combining the shape of one object class with the texture of another, using neural style transfer techniques [12]. Models and humans are then presented with these hybrid images, and their predictions are analyzed to infer which visual cues they rely on. The observed divergence, with CNNs favoring texture and humans favoring shape, has become a dominant narrative for understanding human–machine perceptual differences and has inspired a wide range of follow-up studies [13], [14], [15], [16], [17].

Although influential, the cue-conflict experiment is based on assumptions that may limit the generalizability and clarity of its findings. Conceptually, it reduces feature reliance to a binary choice between shape and texture, overlooking other potentially informative cues such as color, and tends to link salience with reliance implicitly. Methodologically, the generated stimuli entangle unintentionally

Cue-Conflict Setup by Geirhos et al. [8] | Evaluating Feature Reliance through Suppression

Figure 1: Comparison of cue-conflict setup [8] (left) and our suppression-based framework (right). While Geirhos et al. infer reliance through preference on hybrid images, our framework directly quantifies reliance by measuring accuracy under systematic suppression of texture, shape, or color.

Cue-conflict stimuli conflate multiple features, introduce texture cues across the image in a spatially unbalanced manner, and rely on shape-based response interfaces that may bias human judgments. As discussed further in Section 3, these conceptual and methodological limitations complicate conclusions about the feature use of models and humans.

In this work, we argue for a conceptual shift: from analyzing feature bias through forced-choice conflicts to assessing feature reliance through targeted suppression. This conceptual distinction reframes how feature preferences and reliance should be evaluated. A model may prefer a certain cue in conflict, not because it is more predictive, but because it is more salient. Conversely, a model may rely heavily on a feature in natural settings, even if it does not dominate in cue-conflict scenarios. To complement existing cue-conflict protocols, we propose a domain-agnostic evaluation framework that quantifies performance degradation under systematic suppression of individual feature types. While avoiding some limitations of adversarial-style inputs, our method introduces its own challenges related to transformation artifacts and incomplete suppression. By isolating individual feature contributions, our framework offers a more reliable basis for interpreting model decisions and comparing representational strategies, both between humans and neural networks, and across model architectures and domains.

Our main contributions are as follows:
(1) We present a re-examination of Geirhos et al.'s cue-conflict experiment, highlighting aspects in their evaluation protocol that may limit its generalizability.
(2) We introduce a domain-agnostic framework for evaluating feature reliance through targeted feature suppression, enabling cleaner measurement of model dependence on individual visual cues without requiring conflicting cue setups.
(3) Using the proposed framework, we systematically compare human and model feature reliance under controlled conditions. Our results challenge the texture bias hypothesis [8] by showing that CNNs are not inherently texture-biased; instead, they only exhibit a pronounced sensitivity to local shape, which can be mitigated through modern training strategies. Notably, models trained with vision-language supervision most closely match human behavior.
(4) We apply the same framework to assess domain-specific differences in feature reliance, showing that models trained on CV, remote sensing (RS), and medical imaging (MI) datasets prioritize distinct visual cues depending on domain characteristics.


## Related Work


2 Related Work

Understanding which features deep neural networks rely on for image classification has been a long-standing research question. While early interpretations of CNNs assumed a hierarchical buildup from


![Figure 6](image_6.jpg)
<!-- Caption: Related Work, index 6 -->



![Figure 7](image_7.jpg)
<!-- Caption: Related Work, index 7 -->



![Figure 8](image_8.jpg)
<!-- Caption: Related Work, index 8 -->


While CNNs are believed to learn a hierarchy from low-level edges to complex shape representations [5], [6], [7], more recent studies have challenged this view, suggesting that CNNs often rely disproportionately on local texture rather than global shape [9], [10], [8], [18]. Geirhos et al. [8] formalized this observation as the texture bias hypothesis, using a cue-conflict protocol to reveal divergent feature preferences between humans and CNNs.

Subsequent work investigated factors shaping feature reliance beyond architecture. Hermann et al. [13] showed that texture bias in CNNs arises primarily from training objectives and augmentations, with techniques like blurring and cropping increasing shape bias more than architectural changes. Although shape features are present in deeper layers [15], [14], they are not consistently used during classification. Transformer-based models and vision-language models have shifted this discussion. Vision transformers (ViTs) exhibit reduced texture bias compared to some CNNs due to their global attention mechanisms [19], [20]. Current understanding of vision-language models’ influence on shape bias remains under active investigation, with limited definitive conclusions drawn thus far.

Various methods have attempted to enforce shape bias or suppress texture cues for improved robustness, including anisotropic filtering [22], edge encoding [23], style disentanglement [24], [25], and shape-focused augmentations [26], [27]. However, stylization alone may improve robustness independent of shape bias [28], and neither shape nor texture bias reliably predicts generalization [16]. These findings have motivated integrative approaches that combine diverse feature biases. Joint supervision [29], ensembles [30], and adaptive recombination [31] aim to harness complementary features. Ge et al. [32] and Jain et al. [17] show that disentangling and combining shape, texture, and color improve robustness and interpretability. Nonetheless, Lucieri et al. [33] caution that in domains like MI, cue entanglement is essential and biasing towards shape may be counterproductive.

Efforts to increase shape bias are often motivated by the broader goal of human-model alignment. Geirhos et al. [34], [35] show that even robust models exhibit error patterns that diverge from humans, revealing a persistent consistency gap. Muttenthaler et al. [36] further argue that alignment with human conceptual structure depends more on training signals than model scale, indicating that robustness and shape bias alone are insufficient proxies for human-like perception.


## Motivation


3 Rethinking Texture Bias: A Critical Look at Cue-Conflict Evaluation

The hypothesis that CNNs trained on ImageNet are biased towards texture was popularized by Geirhos et al. [8], who introduced a cue-conflict evaluation protocol. In this setup, images were generated by neural style transfer [12], combining the shape content (cue) of one class with the texture content (cue) of another. Predictions from both humans and CNNs on these images were then used to infer whether classification decisions were driven more by shape or texture features. Over time, the cue-conflict evaluation protocol has become a de facto standard for assessing feature bias in deep neural networks. While impactful, this protocol introduced several assumptions and limitations that have received limited attention. Conceptually, the protocol frames feature reliance as a binary shape-or-texture choice, which may overlook other cues such as color and conflate salience with dependence. In addition, the stylized stimuli constrain the evaluation of feature bias to naturalistic images with a similar set of classes and cannot be generalized across datasets (e.g., flower classification) or domains (e.g., RS, MI). Beyond these conceptual limitations, the cue-conflict protocol exhibits three methodological concerns in its design and implementation:

(i) Lack of Feature Isolation. The texture cues within the cue-conflict images also preserved information beyond texture, including color and local shape structures (e.g., contours and parts of silhouettes). As a result, the synthesized texture cue was not a pure representation of texture but a composite of multiple features, making it difficult to attribute classification behavior to texture alone. An example can be seen in Figure 2a.

(ii) Overloaded Texture Class Signals. The protocol consistently inserted texture cues not only into the object region but also into the image background. Since CNNs aggregate local statistics across spatial positions, this broad spatial distribution increased the signal strength of the texture class relative to the shape class. This spatial imbalance systematically biases CNNs towards texture-based decisions, not because of an intrinsic preference but due to the dominant spatial availability of the texture signal. An example can be seen in Figure 2b.

(iii) Human Interface Bias Towards Shape. Participants in the human experiments selected the image class by clicking on buttons labeled with icons representing each category. These icons represented global shape characteristics (i.e., silhouettes), potentially guiding participants toward shape-based choices.

towards matching shape features in the cue-conflict image with the icon. This response format potentially introduces bias towards shape decisions, especially when participants were unsure which feature to prioritize. The used icons are visualized in Figure 2c.

(a) [Image showing boat shape cue merged with chair texture cue]
(b) [Image showing airplane shape cue merged with clock texture cue]
(c) [Four icons showing: airplane, car, cat, and bird silhouettes in a 2x2 grid]

Figure 2: Example images taken from the cue-conflict dataset [8]. (a) Boat shape cue merged with chair texture cue. (b) Airplane shape cue merged with clock texture cue. (c) Icons of the human interface to select classes.

These design choices may inadvertently influence CNNs towards texture-driven decisions and humans towards shape-based judgments and complicate drawing definitive conclusions about the actual feature reliance of models.


## Method


4 A Domain-Agnostic Framework for Feature Reliance

Accurately assessing how deep neural networks rely on different visual features remains a central challenge in understanding their behavior. While the cue-conflict evaluation protocol [8] introduced a reliance test based on feature bias, it imposes conceptual and methodological constraints that limit its generalizability. Rather than forcing models to choose between shape and texture, we propose to assess their reliance on individual feature types by systematically suppressing them and measuring the resulting impact on classification performance. This shift enables a more flexible, generalizable, and semantically grounded analysis of feature use in neural networks.

To evaluate the reliance of deep neural networks on individual visual features, we employ a set of image transformations that selectively suppress shape, texture, or color information while minimally affecting the remaining features. Each transformation is chosen for its ability to target a specific feature class. We define three feature types:

• Shape refers to information carried by spatial arrangement and structural contours, including both global (object outline) and local (part-level) shape.
• Texture is defined by repetitive patterns, high-frequency local variations, and fine-grained surface details.
• Color denotes chromatic information independent of spatial layout or texture.

For each feature type, we include two complementary transformations that differ in their suppression mechanisms and preservation profiles, offering distinct but comparable perspectives on the targeted feature. The transformations are summarized in Table 1 and briefly described in the following. Patch Shuffle [37, 28] randomizes spatial positions of patches, preserving some texture but disrupting global shape. Patch Rotation rotates individual patches, which may partially disrupt both shape and local texture cues due to edge discontinuities introduced within patches. Both transformations primarily disrupt shape but can unintentionally affect texture, complicating a clean feature separation. Bilateral Filtering [38] and Gaussian Blur reduce texture by smoothing high-frequency details, with the former preserving edges more effectively. Grayscale removes chromatic cues entirely, while Channel Shuffle disrupts color correlations without altering intensity. In the following, we validate the suppression effects of these transformations using quantitative metrics.


![Figure 13](image_13.jpg)
<!-- Caption: Method, index 13 -->



![Figure 14](image_14.jpg)
<!-- Caption: Method, index 14 -->



![Figure 15](image_15.jpg)
<!-- Caption: Method, index 15 -->


Table 1: Feature-suppressing transformations used in this work. Each feature is suppressed using two transformations with differing strengths.

| Feature Type | Transformation 1 | Transformation 2 |
| ------------ | ---------------- | ---------------- |
| Shape        | Patch Shuffle    | Patch Rotation   |
| Texture      | Bilateral Filter | Gaussian Blur    |
| Color        | Grayscale        | Channel Shuffle  |


Table 2: Quantitative validation of suppression transformations across 800 images of ImageNet. Each transformation is used with a fixed parameter setting (see Param ID legend below). Values report normalized metric scores. Arrows indicate desired direction: ↑ higher is better, ↓ lower is better.

| Transformation        | Param ID | Texture↓ | Shape↑ | LV↓   | HFE↓  | ESSIM↑ | GC↑   |
| --------------------- | -------- | -------- | ------ | ----- | ----- | ------ | ----- |
| *Texture-Suppressing* |          |          |        |       |       |        |       |
| Bilateral Filter      | A        | 0.521    | 0.796  | 0.548 | 0.493 | 0.737  | 0.855 |
| Box Blur              | B        | 0.193    | 0.363  | 0.237 | 0.148 | 0.436  | 0.289 |
| Gaussian Blur         | C        | 0.349    | 0.662  | 0.392 | 0.306 | 0.744  | 0.579 |
| Median Filter         | D        | 0.357    | 0.506  | 0.399 | 0.316 | 0.584  | 0.429 |
| NLMeans Denoising     | E        | 0.706    | 0.797  | 0.723 | 0.690 | 0.730  | 0.864 |
| Transformation        | Param ID | Texture↑ | Shape↓ | LV↑   | HFE↑  | ESSIM↓ | GC↓   |
| *Shape-Suppressing*   |          |          |        |       |       |        |       |
| Patch Shuffle         | F        | 1.000    | 0.176  | 1.000 | 1.000 | 0.205  | 0.147 |
| Patch Rotation        | F        | 1.000    | 0.293  | 1.000 | 1.000 | 0.339  | 0.247 |

Legend: A: $d=11$, $\sigma_c=170$, $\sigma_s=75$; B: $k=11$; C: $k=11$, $\sigma=2.0$; D: $k=11$; E: $h=20$, $tws=11$, $sws=11$; F: $grid=6$.

4.1 Quantitative Validation of Suppression Transformations

While the individual transformations used in this work are not novel, their selection for targeted feature suppression requires empirical justification. To validate that each transformation suppresses the intended visual feature (e.g., texture, shape) while preserving others, we quantify their effects using four metrics: Local Variance (LV) [39] and High-Frequency Energy (HFE) [40] to assess texture suppression, and Edge-SSIM (ESSIM) [41] and Gradient Correlation (GC) to measure shape preservation. All metrics are normalized to the range [0, 1] by dividing by the scores of the unsuppressed image. Higher values of ESSIM and GC indicate better preservation of edge and structural information, while lower values of LV and HFE reflect stronger suppression of texture features. Further, we compute a harmonic mean across the two texture metrics (Texture) and the two shape metrics (Shape) for each transformation.

We test the effectiveness of the feature suppression transformations across 800 sampled images from the ImageNet validation set. For each transformation, we evaluate a representative configuration chosen to balance suppression of the target feature and preservation of others. These settings, such as kernel size or smoothing strength, are indexed by Param IDs in Table 2, with details listed below the table. A full ablation of parameter settings is provided in the supplemental material (see Section D). In addition to our selected texture suppression transformations, we also compare common alternatives such as Non-Local Means Denoising [42], Box blur, and Median filtering [43] to ensure a fair comparison across standard smoothing techniques. Among texture-suppressing methods, bilateral filtering yields the most balanced trade-off between reducing texture (LV: 0.54, HFE: 0.49) and preserving shape (ESSIM: 0.74, GC: 0.85). Gaussian Blur suppresses texture more uniformly but leads to a greater loss of shape information. Box blur and median filtering remove texture strongly, but at a substantial cost to shape preservation. For shape suppression, we evaluate Patch Shuffle and Patch Rotation with a grid size of 6. These transformations preserve texture but substantially disrupt structural integrity, making them suitable for assessing shape reliance. To complement the quantitative evaluation, qualitative visual examples of the suppression effects are provided in the supplemental material (see Section C).


## Experiments


# 5 Experiments

## 5.1 Experiment I: Human vs. CNNs Feature Reliance

Experimental Setup. To compare human and model reliance on different visual features, we designed a controlled experiment inspired by Geirhos et al. [34, 8]. The ImageNet16-like dataset was constructed by selecting 50 images per category from the ImageNet validation set without manual adjustment. Categories with fewer confident predictions were left as is, leading to class imbalance. Models were evaluated using their default pretrained weights without fine-tuning to ensure fair and consistent comparisons across architectures.

Humans were presented with image stimuli in randomized order under one of five conditions: original, global shape suppression, local shape suppression, texture suppression, or color suppression. Each feature was suppressed via a single transformation with fixed hyperparameters: Patch Shuffle with grid size 3 (global shape), grid size 6 (local shape), bilateral filtering with d=12, σ_color=170 and σ_space=75 (texture), and grayscale conversion (color). See Section 4.1 for justification. Each participant saw only one randomly chosen version of each image to avoid learning effects. The five suppression conditions of one image were split across groups of five participants to ensure balanced coverage. Only twenty participants completed the study, with each shown a random subset of image versions under different suppression conditions, limiting exposure per condition. Trials included brief 200 ms image presentations with masking to restrict perceptual processing. Participants answered via an alphabetical grid interface presenting 16 categories, risking interface-induced bias and potential random guessing due to limited trial counts per condition. An additional "not clear" button was available for unrecognizable stimuli. Attention checks were administered every 100 trials, and failed trials were excluded. Additional details and interface screenshots can be found in the supplemental material.

Model evaluation mirrored the human protocol, evaluating their performance under the same five suppression conditions using the identical image set shown to humans. Model predictions were determined using the argmax class across all subclasses without thresholding to ensure standard classification metrics and facilitate unbiased, consistent evaluation aligned with common practice. This procedure was chosen heuristically, complementary results using argmax to define class predictions are reported in the supplemental material and show nearly identical reliance profiles.

We evaluated several architectures: ResNet50-standard, trained from scratch with basic augmentations, and ResNet50-sota, trained with a modern recipe [46]. Additional CNNs include MobileNetV3 [47], EfficientNet [3], EfficientNetV2 [48], ConvMixer [49], ConvNeXt [4], and ConvNeXtV2 [50]. Transformer-based models include ViT [51], DeiT [52], SwinTransformer [53], and CLIP-ViT [54]. All models except ResNet50-standard were obtained as pretrained checkpoints from the timm library [55]. The detailed training procedures can be found in the supplemental material.

Results. Figure 3 presents a comparative overview of the performance of humans and CNNs under feature suppression, plotted as the relative accuracy (i.e., accuracy under suppression divided by baseline accuracy on original images). Separate subplots show results for each suppressed feature type. We highlight three representative CNNs: ResNet50-standard, ResNet50-sota, and ConvNeXtV2 alongside human performance. The results show that CNNs are not strongly reliant on texture: under texture suppression, ResNet50-standard retains 80% of its original performance, close to performance under global shape suppression (83%). The highest vulnerability is observed under local shape suppression, where accuracy drops to just 28%. Humans exhibit a similar reliance profile with local shape suppression being most disruptive, but show higher robustness to it (76% retained accuracy). Interestingly, modern training strategies substantially mitigate this effect: the ResNet50-sota reaches 62% under local shape suppression, and ConvNeXtV2 improves further to 65%. While modern training regimes appear to reduce local shape reliance in some CNN architectures, this effect is not consistent across all models. Certain architectures continue to show pronounced dependence on local shape cues regardless of training improvements, indicating architecture and capacity also contribute significantly to feature reliance. Statistical significance tests confirming these differences are reported in the supplemental material.

Broadening the analysis to a wider range of architectures (Table 3), we observe that several models trained with state-of-the-art recipes exhibit a more balanced reliance profile. However, this trend is not universal: ConvMixer, EfficientNet, and MobileNet variants retain a strong dependence on local shape, indicating that improved training alone does not guarantee human-like feature use and that architectural inductive biases or capacity limitations may still play a role. Among transformer-

(a) Global Shape | (b) Local Shape | (c) Texture | (d) Color

[Four scatter plots showing relative accuracy under different feature suppression methods, with data points for Humans, ResNet50-standard, ResNet50-sota, and ConvNeXtV2]

Figure 3: Relative accuracy under feature suppression for human observers and three CNNs—ResNet50-standard, ResNet50-sota, ConvNeXtV2—on the curated ImageNet16 dataset. Each subplot shows performance under suppression of a specific feature: (a) global shape via Patch Shuffle (grid = 3); (b) local shape via Patch Shuffle (grid = 6); (c) texture via bilateral filtering; and (d) color via grayscale.

Table 3: Relative accuracy (accuracy under suppression divided by accuracy on original images) for each feature-suppression type across models and human observers.

Architecture | Global Shape | Local Shape | Texture | Color | Original | #Params
Humans | 0.965 | 0.763 | 0.979 | 0.999 | 0.969 | –
ResNet50-standard [2] | 0.832 | 0.276 | 0.795 | 0.924 | 0.954 | 25.6M
ResNet50-sota [46] | 0.943 | 0.618 | 0.867 | 0.948 | 0.931 | 25.6M
ConvNeXt [4] | 0.938 | 0.606 | 0.910 | 0.961 | 0.934 | 28.6M
ConvNeXtV2 [50] | 0.949 | 0.647 | 0.925 | 0.969 | 0.940 | 28.6M
EfficientNet [3] | 0.870 | 0.240 | 0.892 | 0.987 | 0.856 | 30.0M
EfficientNetV2 [48] | 0.926 | 0.423 | 0.897 | 0.957 | 0.932 | 24.0M
MobileNetV3 [47] | 0.795 | 0.217 | 0.761 | 0.859 | 0.881 | 5.4M
ConvMixer [49] | 0.920 | 0.437 | 0.815 | 0.891 | 0.874 | 21.1M
ViT [51] | 0.930 | 0.636 | 0.921 | 0.977 | 0.929 | 86.6M
DeiT [52] | 0.938 | 0.730 | 0.926 | 0.969 | 0.932 | 86.6M
Swin [53] | 0.924 | 0.713 | 0.906 | 0.941 | 0.945 | 87.8M
CLIP ViT [54] | 0.959 | 0.758 | 0.949 | 0.984 | 0.936 | 86.6M

Among transformer-based models, the ViT demonstrates a feature reliance profile similar to ResNet50-sota across all suppression conditions, challenging the notion that transformers are inherently more shape-oriented than CNNs. The CLIP model shows better overall robustness under feature suppression conditions relative to other models, indicating that vision-language supervision influences feature utilization. However, alignment with human patterns remains partial and domain-dependent, warranting further investigation. This may reflect the effect of contrastive vision-language training, which prioritizes alignment with high-level semantic concepts over low-level visual cues.

These findings challenge the texture bias hypothesis popularized by Geirhos et al. [8] as a fixed inductive bias of CNNs. Instead, the observed behavior in the cue-conflict experiment may have reflected a dominant reliance on local shape features, rather than an inherent texture bias.

5.2 Experiment II: Domain-specific Feature Reliance

While Section 5.1 focuses on comparing feature reliance between humans and CNNs on a fixed benchmark, this section explores how reliance on shape, texture, and color varies across domains. The same suppression-based framework introduced earlier is applied to three representative visual domains: CV, MI, and RS. In each case, we fix the architecture to a ResNet50 and apply the standard training protocol, including only the data augmentation techniques: random resized crop and horizontal flip. For CV datasets, we either train from scratch or initialize models with ImageNet-pretrained weights (standard training protocol) and then fine-tune on the respective datasets. For MI and RS, we train from scratch to allow a disentangled comparison across domains. Additional results for MI and RS with pretrained models to simulate operational scenarios can be found in the supplemental material. Details about the hyperparameters, as well as an overview of the corresponding validation accuracies, are provided in the supplemental material. To reduce redundancy, we report results


![Figure 22](image_22.jpg)
<!-- Caption: Experiments, index 22 -->



![Figure 23](image_23.jpg)
<!-- Caption: Experiments, index 23 -->



![Figure 24](image_24.jpg)
<!-- Caption: Experiments, index 24 -->



![Figure 25](image_25.jpg)
<!-- Caption: Experiments, index 25 -->


(a) CV Shape Suppression
Graph showing Accuracy (Relative to No Suppression) vs. Grid Size (0–12) with multiple colored lines representing:
- ImageNet (pretrained)
- Caltech101 (pretrained)
- Flowers102 (pretrained)
- Oxford-IIIT-Pet (pretrained)
- STL10 (pretrained)
All lines show declining accuracy as grid size increases, starting near 1.0 and dropping to approximately 0.0–0.2 at grid size 12.

(b) CV Texture Suppression
Graph showing Accuracy (Relative to No Suppression) vs. σ (0–14) with multiple colored lines representing:
- ImageNet (pretrained)
- Caltech101 (pretrained)
- Flowers102 (pretrained)
- Oxford-IIIT-Pet (pretrained)
- STL10 (pretrained)
Lines show a gradual decline from 1.0 to approximately 0.4–0.8 as σ increases to 14.

(c) CV Color Suppression
Graph showing Accuracy (Relative to No Suppression) vs. α (0–1.0) with multiple colored lines representing:
- ImageNet (pretrained)
- Caltech101 (pretrained)
- Flowers102 (pretrained)
- Oxford-IIIT-Pet (pretrained)
- STL10 (pretrained)
Lines show varying patterns with some declining more steeply than others as α increases.

(d) MI Shape Suppression
Graph showing Accuracy (Relative to No Suppression) vs. Grid Size (0–12) with multiple colored lines representing:
- BloodMNIST (from scratch)
- BreastMNIST (from scratch)
- DermaMNIST (from scratch)
- OCTMNIST (from scratch)
- PathMNIST (from scratch)
Lines show varying decline patterns, with some datasets more robust to shape suppression than others.

(e) MI Texture Suppression
Graph showing Accuracy (Relative to No Suppression) vs. σ (0–14) with multiple colored lines representing:
- BloodMNIST (from scratch)
- BreastMNIST (from scratch)
- DermaMNIST (from scratch)
- OCTMNIST (from scratch)
- PathMNIST (from scratch)
Lines show relatively stable performance with gradual decline as σ increases.

(f) MI Color Suppression
Graph showing Accuracy (Relative to No Suppression) vs. α (0–1.0) with multiple colored lines representing:
- BloodMNIST (from scratch)
- BreastMNIST (from scratch)
- DermaMNIST (from scratch)
- OCTMNIST (from scratch)
- PathMNIST (from scratch)
Lines show dramatic differences in sensitivity to color suppression across datasets.

(g) RS Shape Suppression
Graph showing Accuracy (Relative to No Suppression) vs. Grid Size (0–12) with multiple colored lines representing:
- AID (from scratch)
- EuroSAT (from scratch)
- NWPU (from scratch)
- RESISC45 (from scratch)
- UC Merced (from scratch)
Lines show gradual decline from 1.0 to approximately 0.2–0.4 as grid size increases.

(h) RS Texture Suppression
Graph showing Accuracy (Relative to No Suppression) vs. σ (0–14) with multiple colored lines representing:
- AID (from scratch)
- EuroSAT (from scratch)
- NWPU (from scratch)
- RESISC45 (from scratch)
- UC Merced (from scratch)
Lines show decline from 1.0 to approximately 0.4–0.6 as σ increases.

(i) RS Color Suppression
Graph showing Accuracy (Relative to No Suppression) vs. α (0–1.0) with multiple colored lines representing:
- AID (from scratch)
- EuroSAT (from scratch)
- NWPU (from scratch)
- RESISC45 (from scratch)
- UC Merced (from scratch)
Lines show varying patterns with most declining as α increases.

Figure 4: Feature suppression results across three domains. Top row (a–c): ResNet50 pretrained on ImageNet and fine-tuned on CV datasets. Middle row (d–f): ResNet50 trained from scratch on MI datasets from MedMNIST v2. Bottom row (g–i): ResNet50 trained from scratch on high-resolution RS datasets. Columns correspond to: (a, d, g) shape suppression (Patch Shuffle), (b, e, h) texture suppression (Bilateral Filter), and (c, f, i) color suppression (Grayscale).

We use one representative suppression technique per feature type in the main paper. Results using alternative suppression methods per feature type are included in the supplemental material and exhibit qualitatively similar patterns across domains.

To visualize domain-specific suppression sensitivity, we present selected datasets in Figure 4, illustrating general trends of feature reliance under suppression. While most datasets show monotonic accuracy declines with feature suppression, some exhibit atypical behaviors or plateaus. Chance-level accuracies were rescaled to zero and baseline accuracies to one to enable cross-domain visual comparison, though this normalization masks inherent reliability differences. Figure 5 aggregates suppression trends but should be interpreted cautiously given dataset-level heterogeneity.

Computer Vision (CV).
Figure 4a–c shows suppression results for five standard CV benchmarks (ImageNet [45], Caltech101 [56], Flowers102 [57], Oxford-IIIT-Pet [58], STL10 [59]). Across datasets, we observe that shape suppression induces the strongest performance degradation, especially as the patch-shuffle grid size increases. This confirms a pronounced reliance on local shape information in pretrained CNNs. In contrast, texture suppression via bilateral filtering has minimal effect, and color suppression through grayscale conversion yields only minor degradation, indicating that CNNs fine-tuned on these datasets are largely robust to the removal of texture and color cues. These results are consistent with our human comparison study and suggest that local shape continues to dominate feature reliance in natural image classification tasks. For completeness, the supplemental material reports results using alternative suppression methods per feature type that exhibit qualitatively similar patterns across domains.


![Figure 29](image_29.jpg)
<!-- Caption: Experiments, index 29 -->



![Figure 30](image_30.jpg)
<!-- Caption: Experiments, index 30 -->



![Figure 31](image_31.jpg)
<!-- Caption: Experiments, index 31 -->



![Figure 32](image_32.jpg)
<!-- Caption: Experiments, index 32 -->



![Figure 33](image_33.jpg)
<!-- Caption: Experiments, index 33 -->



![Figure 34](image_34.jpg)
<!-- Caption: Experiments, index 34 -->



![Figure 35](image_35.jpg)
<!-- Caption: Experiments, index 35 -->



![Figure 36](image_36.jpg)
<!-- Caption: Experiments, index 36 -->



![Figure 37](image_37.jpg)
<!-- Caption: Experiments, index 37 -->


(a) Shape Suppression | (b) Texture Suppression | (c) Color Suppression

Figure 5: Domain-averaged feature suppression curves for CV, MI, and RS. (a) Shape suppression via Patch Shuffle. (b) Texture suppression via bilateral filtering. (c) Color suppression via grayscale.
It includes results for models trained from scratch as well as a class-wise analysis for ImageNet, which confirms that the global reliance patterns are consistent across categories.

Medical Imaging (MI). Figure 4d–f summarizes results on five MedMNIST-v2 datasets with differing sample sizes and class distributions. Data heterogeneity and imbalance possibly contribute to observed suppression effect variability. Detailed statistical tests and validation splits are not reported, limiting interpretability of the results. While shape suppression degrades performance, the impact is generally less pronounced, and texture suppression yields moderate performance drops in datasets such as PathMNIST and BloodMNIST, but relatively little effect in RetinaMNIST and DermaMNIST. By contrast, color suppression induces a substantial decline in classification accuracy for most datasets, reflecting the strong diagnostic role of chromatic cues, except in ChestMNIST, which contains only grayscale images. Taken together, these results suggest that feature reliance in MI varies substantially across datasets, with a common trend towards greater dependence on color information.

Remote Sensing (RS). Figure 4g–i reports suppression curves for five very-high-resolution RGB datasets: UCMerced [61], RSD46-WHU [62], DeepGlobe [63], PatternNet [64], and AID [65]. As in MI, shape suppression impacts performance, but the degradation is less pronounced than in the CV domain, indicating lower reliance on local shape. In contrast to CV and MI, texture suppression leads to substantial performance degradation across all datasets, suggesting that fine-grained surface patterns are critical for RS classification. Surprisingly, color suppression also results in notable performance drops, despite the use of RGB imagery only. This likely reflects strong correlations between chromatic cues and semantic land cover categories. Overall, RS models exhibit a pronounced reliance on texture and color, and comparatively less dependence on local shape, reflecting the distinct statistical structure and spatial semantics of RS imagery.

Cross-Domain Comparison. To synthesize these observations, Figure 5 presents the domain-averaged suppression curves for each feature, including 1-sigma error bars. Three clear trends emerge. First, CV models are most reliant on local shape, especially when trained from scratch, while ImageNet pretraining induces slightly greater robustness. Second, MI models exhibit stronger dependence on color, consistent with the nature of some medical tasks (e.g., in dermatology, histopathology), which often require interpreting chromatic cues. Third, RS models exhibit the highest texture reliance among the three tested domains. This may reflect the nature of many RS classes that are defined by texture-like patterns (e.g., fields, residential areas), rather than by distinct global contours. These patterns confirm that feature reliance is shaped not only by architecture and training regime, but also by the visual and semantic properties of the task or domain.

Finally, to validate the observed feature reliance patterns, we conduct complementary experiments on CV datasets with simultaneous suppression of two features (see Section I.5 in the supplemental material). Results confirm the trends of single-feature suppression: performance is highest when only shape is preserved, reduced when only texture remains, and nearly lost when only color is available. In summary, the findings highlight that domain characteristics, alongside architecture and training regime, play a crucial role in shaping feature reliance. While prior work emphasized architecture-induced biases, our results suggest that data properties equally govern the perceptual strategies that models adopt.


![Figure 43](image_43.jpg)
<!-- Caption: Experiments, index 43 -->



![Figure 44](image_44.jpg)
<!-- Caption: Experiments, index 44 -->



![Figure 45](image_45.jpg)
<!-- Caption: Experiments, index 45 -->



## Conclusion


# 6 Conclusion

This paper revisited the widely cited claim that CNNs trained on ImageNet are inherently biased towards texture. We identify critical conceptual and methodological limitations in the cue-conflict experiment popularized by Geirhos et al. [8] that support this hypothesis. Further, we propose a new framework for evaluating feature reliance based on targeted suppression rather than forced-choice preference. Using this framework, we find no evidence for an inherent texture bias in CNNs, but instead observed a pronounced reliance on local shape features. Nonetheless, we show that this reliance can be substantially mitigated through modern training strategies. Across domains, we find that feature reliance varies substantially: CV models prioritize shape, MI models rely more evenly on color, and RS models exhibit strong texture sensitivity. These findings add to growing evidence that architectural biases alone do not determine CNN feature reliance, which also depends heavily on training objectives and domain characteristics. Understanding this flexibility informs future model design toward improved alignment with human perception strategies. At the same time, the relative contributions of architectural components and training strategies to these reliance patterns remain to be systematically evaluated.

Limitations. Our framework relies on operational definitions of shape, texture, and color based on specific transformations, but features are continuous and interdependent, limiting perfect isolation. In practice, suppression only reduces rather than eliminates features: texture suppression can leave residual low-level features perceptible as texture, while shape suppression does not fully remove all shape cues. This reflects the inherent trade-off of reducing one feature while preserving others, making absolute removal unattainable. The applied suppression techniques may also introduce artifacts that affect model behavior independently of the targeted features (e.g., block-like structures from Patch Shuffle, smoothing from filtering). Further, the results obtained with pretrained models may reflect effects of similarities between suppression transformations and augmentation techniques (e.g., Cutout and Patch Shuffle). Finally, our human experiments employed a controlled forced-choice design with brief exposures and a limited set of categories to ensure comparability. While necessary for experimental control, these constraints may not fully reflect the richness and adaptability of human visual perception in real-world settings.


## Appendix


# Acknowledgements

We thank Johanna Vielhaben and Genc Hoxha for suggesting the addition of quantitative experiments to validate the feature suppression techniques, Liliann Lehrke for valuable guidance on the design of the human study, and Christopher Olk for helpful discussions in selecting the title. This work was partly supported by EU Horizon projects ELIAS (No. 101120237) and ELLIOT (No. 101214398). The authors declare no conflicts of interest. The funding agencies had no role in study design, data collection, analysis, or manuscript preparation.


## References


# References

[1] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. ImageNet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, volume 25, 2012.

[2] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

[3] Mingxing Tan and Quoc Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. In Proceedings of the International Conference on Machine Learning, pages 6105–6114, 2019.

[4] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A ConvNet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11976–11986, 2022.

[5] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. Nature, 521(7553):436–445, 2015.

[6] Jonas Kubilius, Stefania Bracci, and Hans P. Op de Beeck. Deep neural networks as a computational model for human shape sensitivity. PLOS Computational Biology, 12(4):e1004896, 2016. Publisher: Public Library of Science, San Francisco, CA, USA.

[7] Samuel Ritter, David G. T. Barrett, Adam Santoro, and Matt M. Botvinick. Cognitive psychology for deep neural networks: A shape bias case study. In International Conference on Machine Learning, pages 2940–2949, 2017.
[8] Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A. Wichmann, and Wieland Brendel. ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. In International Conference on Learning Representations, 2019.
[9] Nicholas Baker, Hongjing Lu, Gennady Erlikhman, and Philip J. Kellman. Deep convolutional networks do not classify based on global object shape. PLOS Computational Biology, 14(12):e1006613, 2018. Publisher: Public Library of Science, San Francisco, CA, USA.
[10] Wieland Brendel and Matthias Bethge. Approximating CNNs with Bag-of-Local-Features models works surprisingly well on ImageNet. In International Conference on Learning Representations, 2019.
[11] Barbara Landau, Linda B. Smith, and Susan S. Jones. The importance of shape in early lexical learning. Cognitive Development, 3(3):299–321, 1988.
[12] Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2414–2423, 2016.
[13] Katherine Hermann, Ting Chen, and Simon Kornblith. The Origins and Prevalence of Texture Bias in Convolutional Neural Networks. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 19000–19015, 2020.
[14] Katherine Hermann and Andrew Lampinen. What shapes feature representations? Exploring datasets, architectures, and training. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 9995–10006, 2020.
[15] Md. Amirul Islam, Matthew Kowal, Patrick Esser, Sen Jia, Björn Ommer, Konstantinos G. Derpanis, and Neil Bruce. Shape or texture: Understanding discriminative features in CNNs. In International Conference on Learning Representations, 2021.
[16] Paul Gavrikov and Janis Keuper. Can biases in ImageNet models explain generalization? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22184–22194, 2024.
[17] Saachi Jain, Dimitris Tsipras, and Aleksander Madry. Combining diverse feature priors. In International Conference on Machine Learning, pages 9802–9832, 2022.
[18] Ajay Subramanian, Elena Sizikova, Najib Majaj, and Denis Pelli. Spatial-frequency channels, shape bias, and adversarial robustness. In Advances in Neural Information Processing Systems, volume 36, pages 4137–4149, 2023.
[19] Muhammad Muzammal Naseer, Kanchana Ranasinghe, Salman H. Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Intriguing properties of vision transformers. Advances in Neural Information Processing Systems, 34:23296–23308, 2021.
[20] Shikhar Tuli, Ishita Dasgupta, Erin Grant, and Thomas L. Griffiths. Are convolutional neural networks or transformers more like human vision? arXiv preprint arXiv:2105.07197, 2021.
[21] Paul Gavrikov, Jovita Lukasik, Steffen Jung, Robert Geirhos, Muhammad Jehanzeb Mirza, Margret Keuper, and Janis Keuper. Can we talk models into seeing the world differently? In International Conference on Learning Representations, 2025.
[22] Shlok Mishra, Anshul Shah, Ankan Bansal, Janit Anjaria, Jonghyun Choi, Abhinav Shrivastava, Abhishek Sharma, and David Jacobs. Learning visual representations for transfer learning by suppressing texture. In British Machine Vision Conference, 2022.

[23] Narges Honarvar Nazari and Adriana Kovashka. The Role of Shape for Domain Generalization on Sparsely-Textured Images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5120–5130, 2022.
[24] Hyeonseob Nam, Hyunjae Lee, Jongchan Park, Wonjun Yoon, and Donggeun Yoo. Reducing Domain Gap by Reducing Style Bias. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8690–8699, 2021.
[25] Dhruva Kashyap, Sumukh K. Aithal, Rakshith C., and Natarajan Subramanyam. Towards Domain Adversarial Methods to Mitigate Texture Bias. In International Conference on Learning Representations Workshop, 2022.
[26] Sangjun Lee, Inwoo Hwang, Gi-Cheon Kang, and Byoung-Tak Zhang. Improving Robustness to Texture Bias via Shape-Focused Augmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 4323–4331, 2022.
[27] Aditya Tripathi, Rishabh Singh, Anirban Chakraborty, and Pradeep Shenoy. Edges to Shapes to Concepts: Adversarial Augmentation for Robust Vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24470–24479, 2023.
[28] Chaithanya Kumar Mummadi, Ranjitha Subramaniam, Robin Hutmacher, Julien Vitay, Volker Fischer, and Jan Hendrik Metzen. Does Enhanced Shape Bias Improve Neural Network Robustness to Common Corruptions? In International Conference on Learning Representations, 2021.
[29] Yingwei Li, Qihang Yu, Mingxing Tan, Jieru Mei, Peng Tang, Wei Shen, Alan L. Yuille, and Cihang Xie. Shape-Texture Debiased Neural Network Training. In International Conference on Learning Representations, 2021.
[30] Kenneth T. Co, Luis Muñoz-González, Leslie Kanthan, Ben Glocker, and Emil C. Lupu. Universal Adversarial Robustness of Texture and Shape-Biased Models. In IEEE International Conference on Image Processing, pages 799–803, 2021.
[31] Xinkuan Qiu, Meina Kan, Yongbin Zhou, Yanchao Bi, and Shiguang Shan. Shape-Biased CNNs Are Not Always Superior in Out-of-Distribution Robustness. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2326–2335, 2024.
[32] Yunhao Ge, Yao Xiao, Zhi Xu, Xingrui Wang, and Laurent Itti. Contributions of Shape, Texture, and Color in Visual Recognition. In European Conference on Computer Vision, pages 369–386, 2022.
[33] Adriano Lucieri, Fabian Schmeisser, Christoph Peter Balada, Shoaib Ahmed Siddiqui, Andreas Dengel, and Sheraz Ahmed. Revisiting the Shape-Bias of Deep Learning for Dermoscopic Skin Lesion Classification. In Annual Conference on Medical Image Understanding and Analysis, pages 46–61, 2022.
[34] Robert Geirhos, Carlos R. M. Temme, Jonas Rauber, Heiko H. Schütt, Matthias Bethge, and Felix A. Wichmann. Generalisation in Humans and Deep Neural Networks. Advances in Neural Information Processing Systems, 31, 2018.
[35] Robert Geirhos, Kantharaju Narayanappa, Benjamin Mitzkus, Tizian Thieringer, Matthias Bethge, Felix A. Wichmann, and Wieland Brendel. Partial Success in Closing the Gap Between Human and Machine Vision. Advances in Neural Information Processing Systems, 34:23885–23899, 2021.
[36] Lukas Muttenthaler, Jonas Dippel, Lorenz Linhardt, Robert A. Vandermeulen, and Simon Kornblith. Human Alignment of Neural Network Representations. In International Conference on Learning Representations, 2023.
[37] Tiange Luo, Tianle Cai, Mengxiao Zhang, Siyu Chen, Di He, and Liwei Wang. Defective Convolutional Networks. arXiv preprint arXiv:1911.08432, 2019.
[38] Carlo Tomasi and Roberto Manduchi. Bilateral Filtering for Gray and Color Images. In International Conference on Computer Vision, pages 839–846. IEEE, 1998.

[39] Robert M. Haralick, Karthikeyan Shanmugam, and Its'hak Dinstein. Textural features for image classification. IEEE Transactions on Systems, Man, and Cybernetics, 3(6):610–621, 1973.

[40] Rafael C. Gonzalez and Paul Wintz. Digital image processing. Addison-Wesley, 1987.

[41] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.

[42] Antoni Buades, Bartomeu Coll, and J.-M. Morel. A non-local algorithm for image denoising. In IEEE Computer Society Conference on Computer Vision and Pattern Recognition, volume 2, pages 60–65, 2005.

[43] Thomas S. Huang, G. J. Yang, and G. Y. Tang. A fast two-dimensional median filtering algorithm. IEEE Transactions on Acoustics, Speech, and Signal Processing, 27(1):13–18, 1979.

[44] George A. Miller. WordNet: a lexical database for English. Communications of the ACM, 38(11):39–41, 1995.

[45] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.

[46] Ross Wightman, Hugo Touvron, and Hervé Jégou. ResNet strikes back: An improved training procedure in timm. In Advances in Neural Information Processing Systems Workshop, 2021.

[47] Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, and others. Searching for MobileNetV3. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1314–1324, 2019.

[48] Mingxing Tan and Quoc Le. EfficientNetV2: Smaller models and faster training. In International Conference on Machine Learning, pages 10096–10106, 2021.

[49] Asher Trockman and J. Zico Kolter. Patches Are All You Need? Transactions on Machine Learning Research, 2023. ISSN 2835-8856.

[50] Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, and Saining Xie. ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16133–16142, 2023.

[51] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

[52] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In International Conference on Machine Learning, pages 10347–10357, 2021.

[53] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin Transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10012–10022, 2021.

[54] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, and others. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763, 2021.

[55] Ross Wightman. PyTorch Image Models, 2019. URL https://github.com/rwightman/pytorch-image-models. Publication Title: GitHub repository.

[56] Li Fei-Fei, R. Fergus, and P. Perona. Learning Generative Visual Models from Few Training Examples: An Incremental Bayesian Approach Tested on 101 Object Categories. In *Conference on Computer Vision and Pattern Recognition Workshops*, pages 178–178, 2004.

[57] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In *Indian Conference on Computer Vision, Graphics & Image Processing*, pages 722–729. IEEE, 2008.

[58] Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In *IEEE Conference on Computer Vision and Pattern Recognition*, pages 3498–3505, 2012.

[59] Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. In *Proceedings of the International Conference on Artificial Intelligence and Statistics*, pages 215–223. JMLR Workshop and Conference Proceedings, 2011.

[60] Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, and Bingbing Ni. MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*, 10(1):41, 2023.

[61] Yi Yang and Shawn Newsam. Bag-of-visual-words and spatial extensions for land-use classification. In *Proceedings of the 18th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems*, pages 270–279, 2010.

[62] Yang Long, Yiping Gong, Zhifeng Xiao, and Qing Liu. Accurate object localization in remote sensing images based on convolutional neural networks. *IEEE Transactions on Geoscience and Remote Sensing*, 55(5):2486–2498, 2017.

[63] Ilke Demir, Krzysztof Koperski, David Lindenbaum, Guan Pang, Jing Huang, Saikat Basu, Forest Hughes, Devis Tuia, and Ramesh Raskar. DeepGlobe 2018: A challenge to parse the earth through satellite images. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops*, pages 172–181, 2018.

[64] Weixun Zhou, Shawn Newsam, Congmin Li, and Zhenfeng Shao. PatternNet: A benchmark dataset for performance evaluation of remote sensing image retrieval. *ISPRS Journal of Photogrammetry and Remote Sensing*, 145:197–209, 2018.

[65] Gui-Song Xia, Jingwen Hu, Fan Hu, Baoguang Shi, Xiang Bai, Yanfei Zhong, Liangpei Zhang, and Xiaoqiang Lu. AID: A benchmark data set for performance evaluation of aerial scene classification. *IEEE Transactions on Geoscience and Remote Sensing*, 55(7):3965–3981, 2017.

[66] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C. J. Carey, Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17:261–272, 2020.