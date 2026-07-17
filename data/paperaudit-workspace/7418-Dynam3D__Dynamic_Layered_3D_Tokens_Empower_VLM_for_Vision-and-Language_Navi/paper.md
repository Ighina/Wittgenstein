
## Abstract


Dynam3D: Dynamic Layered 3D Tokens Empower VLM for Vision-and-Language Navigation

Zihan Wang, Seungjun Lee, Gim Hee Lee
School of Computing, National University of Singapore
zihan.wang@u.nus.edu, gimhee.lee@nus.edu.sg

Abstract

Vision-and-Language Navigation (VLN) is a core task where embodied agents leverage their spatial mobility to navigate in 3D environments toward designated destinations based on natural language instructions. Recently, video-language large models (Video-VLMs) with strong generalization capabilities and rich commonsense knowledge have shown remarkable performance when applied to VLN tasks. However, these models still encounter the following challenges when applied to real-world 3D navigation: 1) insufficient understanding of 3D geometry and spatial semantics; 2) limited capacity for large-scale exploration and long-term environmental memory; 3) poor adaptability to dynamic and changing environments. To address these limitations, we propose Dynam3D, a dynamic layered 3D representation model that leverages language-aligned, generalizable, and hierarchical 3D representations as visual input to train 3D-VLMs for navigation action prediction. Given posed RGB-D images, our Dynam3D projects 2D CLIP features into 3D space and constructs multi-level 3D patch–instance–zone representations for 3D geometric and semantic understanding with a dynamic and layer-wise update strategy. Our Dynam3D is capable of online encoding and localization of 3D instances, and dynamically updates them in changing environments to provide large-scale exploration and long-term memory capabilities for navigation. By leveraging large-scale 3D-language pretraining and task-specific adaptation, our Dynam3D achieves competitive performance on VLN benchmarks including R2R-CE, REVERIE-CE, and NavRAG-CE under monocular settings. Furthermore, experiments on pre-exploration, lifelong memory, and preliminary real-world robot tests demonstrate the potential applicability of our approach. The code is available at https://github.com/MrZihan/Dynam3D.


## Introduction


1 Introduction

Vision-and-language navigation (VLN) tasks [1–4] require agents to integrate three core capabilities: 1) understanding natural language instructions, 2) exploring environments and localizing targets or destinations, and 3) planning and executing navigation actions. As illustrated in Figure 1(a), recent works [5–7] have predominantly focused on using video-based large models [8–10] to develop monocular VLN systems. This is due to the practical constraint that most robots are equipped with monocular cameras instead of panoramic cameras. These models pre-trained on large-scale internet data demonstrate strong language understanding and multimodal reasoning abilities, which enable effective instruction following and continuous prediction of navigation actions toward the destination.

Despite these recent advances, several limitations still remain: 1) Video-based models struggle to capture spatial geometry and semantics in large-scale 3D environments. Our experiments reveal that this significantly hinders the ability of these models to explore extensively and correct errors effectively. 2) These models lack mechanisms for structured scene memory. This prevents the use of

(a) Video-based VLM for Navigation.
(b) Our 3D Patch-Instance-Zone VLM for Navigation.
Figure 1: Different vision-language large models for monocular VLN tasks. Compared to previous video-based representations (a), our Dynam3D (b) adopts dynamic hierarchical 3D representations offering advantages in spatial geometry and semantic understanding.

It relies on pre-exploration knowledge and limits the potential for lifelong learning. 3) Representations derived from historical frames are inadequate for dynamically changing 3D scenes, where frequent object and human movements lead to performance drop.

We propose Dynam3D to alleviate the limitations mentioned above. As illustrated in Figure 1(b), our Dynam3D is a 3D-language model with dynamic layered 3D representations for vision-and-language navigation. To encode 3D environments, we extract patch-level 2D features using CLIP [11] and project them into 3D space via depth maps and camera poses. Our Dynam3D employs FastSAM [12] to generate 2D instance masks, and aggregates patch features within each mask into instance-level representations. A 3D instance merging discriminator aligns 2D instances with existing 3D instances based on geometry and semantics to enable dynamic updates of 3D instance representations. While previous methods including online [13] and language-guided [14] 3D segmentation approaches have primarily focused on accurate mask generation and incremental updates, recent advances have also begun exploring semantic alignment in instance representations. Our Dynam3D builds upon these efforts by additionally leveraging large-scale 3D-language pretraining and hierarchical 3D patch-instance-zone representations to enhance scene understanding. However, comprehensive evaluation against prior semantic alignment techniques is needed to quantify the improvements.

Furthermore, our Dynam3D aggregates 3D instance features within spatial zones to facilitate understanding of large-scale environments. As a result, this enables high-level comprehension of layouts, e.g., bedrooms, kitchens, etc., which instance-level features alone cannot capture. Our Dynam3D updates the scene dynamically with this hierarchical patch–instance–zone representation: outdated patch features are removed when a new RGB-D observation arrives, and new features are projected and propagated across the representation layers (patch–instance–zone) for change adaptation. These features enable our Dynam3D to maintain a lifelong and dynamic environmental memory that can significantly improve navigation performance in real-world deployments.

We further introduce a generalizable feature field model [15] to render 3D patch features over an agent-centric panoramic scope for the enrichment of local geometric and semantic perception. These rendered 3D patch features combined with instance and zone representations serve as visual input to the 3D Vision-Language Model (VLM). Given language instructions and action history, the 3D-VLM directly predicts navigation actions, e.g., turn θ degrees, move forward d cm, or stop.

In summary, our main contributions include:
• We propose Dynam3D, a multi-level patch-instance-zone 3D representation model that performs online 3D instance and zone-level encoding and real-time hierarchical updates in dynamic environments.
• We introduce a 3D Vision-Language Model that integrates 3D patch features from generalizable feature fields and 3D instance-zone features from our Dynam3D. This balances fine-grained geometry and global spatial layout for navigation planning.
• Our monocular VLN system achieves competitive performance on benchmarks including R2R-CE, REVERIE-CE, and NavRAG-CE. The results indicate promising potential for pre-exploration, lifelong memory, and preliminary real-world application, although further validation is necessary.


![Figure 8](image_8.jpg)
<!-- Caption: Introduction, index 8 -->



## Related Work


## 2 Related Work

Vision-and-Language Navigation. Vision-and-Language Navigation (VLN) [1, 3, 2, 16–19] requires the agent to understand complex natural language instructions and navigate to the described destination. In contrast to early works [16–18, 20, 21] which primarily concentrate on training and evaluating models within discrete environment simulators [22, 1, 2] (i.e., move on the pre-defined navigation connectivity graph, equipped with panoramic RGB-D camera), recent researches have increasingly emphasized navigation in continuous environment simulators [23–27] and the real-world deployment of monocular VLN systems [5, 28, 15, 6, 7, 29, 30]. For monocular VLN on continuous environment simulators, the agent equips only a forward-facing monocular RGB-D camera, and uses low-level actions to navigate. Recent works [31–35, 29] have adapted 2D-VLMs to VLN tasks, demonstrating notable performance improvements. Extensions like NaVid [5], Uni-NaVid [6], and NaVILA [7] exploit video-based large models for monocular VLN with real-world relevance. While these methods contribute significantly, some still face challenges in fine-grained geometry and large-scale spatial layout comprehension, affecting object localization and planning. Several concurrent approaches also explore 3D-VLMs for dynamic environment navigation. Our Dynam3D introduces a hierarchical multi-level 3D representation with dynamic updates, further advancing these directions, though comparative assessment with similar recent models remains ongoing.

3D Vision-Language Models. Inspired by the development of 2D-VLM [36, 37, 8–10], recent works integrate 3D inputs, the point clouds [38–40] or multi-view images [41–43] to enable 3D scene reasoning for 3D-VLMs. These approaches differ primarily in scene representation: LL3DA [38] encodes full-scene point clouds directly; LEO [40] and Chat-Scene [39] decompose scene point clouds into object-level segments and encode corresponding features. 3D-LLM [42] and Scene-LLM [41] begin with multi-view images, apply 2D object segmentation, and aggregate CLIP features into pixel-aligned 3D points. LLaVA-3D [44] builds on a pretrained 2D VLM [37] to embed 2D patches into 3D voxels via multi-view inputs and 3D positional embeddings. This enables fast adaptation to 3D tasks while maintaining strong 2D perception. However, current 3D-VLMs face fundamental challenges in large-scale unseen and dynamic tasks such as embodied navigation. Full-scene point cloud or voxel-based representations are impractical for real-time reasoning in unseen environments. Existing models lack mechanisms for incremental updates, which make it difficult to revise or discard outdated scene information in dynamic contexts. Moreover, they struggle to balance the computational trade-off between global spatial layout and fine-grained geometric semantics. In this context, we propose Dynam3D, a 3D-VLM model that is better adapted for such dynamic embodied tasks.


## Method


## 3 Our Method

Overview. Figure 2 shows the framework of our Dynam3D for vision-and-language navigation. The framework takes the posed monocular RGB and depth images as input, and outputs atomic navigation actions such as turning, moving forward, stopping etc. Our Dynam3D maintains a set of patch feature points to encode the generalizable feature field [15] used to render the panoramic 3D patch tokens of the agent. Furthermore, our Dynam3D layer-by-layer encodes and updates 3D instance representations and large-scale cube zone representations for multi-level scene understanding and target localization (cf. Section 3.1). These multi-level 3D tokens, navigation instructions and history actions are then fed into a 3D-VLM for next action prediction (cf. Section 3.2).

### 3.1 Dynamic Layered 3D Representation Model

We first design and pre-train a multi-level 3D representation model to acquire language-aligned 3D representations encompassing both fine-grained details and global layouts.

Encoding the Patch Feature Points. To memorize the geometry and semantics of 3D environments, we follow HNR [45] and g3D-LF [15] in using CLIP-ViT-L/14@336px [11] as the encoder for RGB images to extract 2D patch features {g_{t,i} ∈ R^{768}}_{i=1}^I. g_{t,i} denotes the i-th patch feature of the 2D feature map extracted from t-th frame observed by the agent and I = 24 × 24. The patch features {g_{t,i}}_{i=1}^I are then projected to the corresponding 3D world coordinates {P_{t,i}}_{i=1}^I using the depth map

Figure 2: The architecture of our Dynam3D framework. Our Dynam3D takes posed monocular RGB and depth images as input and outputs atomic navigation actions. It encodes and updates multi-level 3D representations for scene understanding and target localization. The 3D tokens, navigation instructions, and history actions are then consolidated into the 3D-VLM for next action prediction.

and camera parameters. For each feature $$\mathbf{g}_{t,i}$$, the observed horizontal orientation $$\theta_{t,i}$$ and the regional size $$s_{t,i}$$ are also calculated and stored to enhance the spatial representation. The set of feature points $$\mathcal{M}$$ can therefore be updated online as:
$$\mathcal{M}_t = \mathcal{M}_{t-1} \cup \{[\mathbf{g}_{t,i}, P_{t,i}, \theta_{t,i}, s_{t,i}]\}_{i=1}^{I}.$$
(1)

Updating the Patch Feature Points. As shown in Figure 3, we employ the Frustum Culling strategy to dynamically update the feature points set $$\mathcal{M}$$ by discarding outdated features and incorporating new ones, which differs from previous methods [46, 45, 28, 15] that simply add new feature points regardless of object motion or removal. Specifically, after obtaining the observed depth image $$\mathbf{D}_t \in \mathbb{R}^{H \times W}$$, the frustum culling strategy transforms the 3D world coordinate $$P_w \in \mathcal{M}$$ of each feature point into the pixel coordinate of the depth image using the camera pose $$[\mathbf{R}, \mathbf{T}]$$ and camera intrinsics $$\mathbf{K}$$ as follows:
$$P_c^{\top} = \begin{bmatrix} x_c \\ y_c \\ z_c \end{bmatrix} = \mathbf{R}P_w^{\top} + \mathbf{T}, \quad \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{z_c}\mathbf{K}\begin{bmatrix} x_c \\ y_c \\ z_c \end{bmatrix},$$
FrustumCulling$$(P_w)$$, if $$0 < z_c < \min(d_{u,v} + \delta, \Delta)$$, $$0 < u < W$$, and $$0 < v < H$$.
(2)

$$d_{h,w}$$ denotes the depth value in row $$h$$ and column $$w$$ of the depth image $$\mathbf{D}_t \in \mathbb{R}^{H \times W}$$. A feature point $$P_w$$ is removed from the feature points set $$\mathcal{M}$$ by the FrustumCulling$$(\cdot)$$ function when $$0 < z_c < \min(d_{u,v} + \delta, \Delta)$$, $$0 < u < W$$ and $$0 < v < H$$, where $$\delta$$ is a noise threshold and $$\Delta$$ is the farthest culling distance. The frustum culling is first applied and followed by adding the new feature points when an RGB-D observation is obtained.

Dynamically Encoding 3D Instance Representations. Due to the overwhelming volume of 3D patch features, a direct employment as visual input to 3D-VLM is computationally and economically impractical. In contrast to voxel-level pooling approaches, e.g., LLaVA-3D [44], our Dynam3D encodes features at the 3D instance level since target localization in navigation instructions is mostly described in terms of object instances. As illustrated in Figure 2, FastSAM [12] rapidly segments


![Figure 15](image_15.jpg)
<!-- Caption: Method, index 15 -->


Figure 3: Left: Illustration of the feature points update and frustum culling strategy. Right: The supervision of feature distillation and 3D-language contrastive learning for our Dynam3D model.

We segment the observed RGB image into a set of 2D instance masks. Within each mask, a transformer-based instance encoder aggregates the corresponding patch features {g_m}_{m=1}^M with positional embeddings {p_m}_{m=1}^M into a compact instance-level representation ℰ using a learnable token q as query:

p_m = MLP([P_m − Average({P_m}_{m=1}^M), s_m, cos(θ_m), sin(θ_m)]).

ℰ = InstanceEncoder(q, {g_m ⊕ p_m}_{m=1}^M). (3)

In contrast to simple 2D instance representations, 3D instances require both multi-view and geometric consistency, enabling the agent to identify the same instance across different views. To this end, we train a Merging Discriminator to integrate 2D instance representations into consistent 3D instances, as shown in Figure 2. Initially, each 2D instance is treated as a new 3D instance. At each subsequent step, for every new 2D instance, the Top-K nearest existing 3D instances are retrieved. The Merging Discriminator evaluates each 2D–3D instance candidate pair using semantic and geometric encodings to determine correspondence. If no match is found among the Top-K candidates, a new 3D instance is created. Otherwise, the 2D instance is merged with the most similar 3D instance by concatenating their patch features and updating the 3D instance representation through the instance encoder. The 3D instance representation is updated incrementally by integrating new patch features only if they are verified to match existing instance semantics and spatial locations. However, outdated patches are removed if they fail the Frustum Culling criteria, potentially causing partial feature loss. Notably, when multiple 2D instances merge into a 3D instance, care is taken to avoid redundant patch aggregation to prevent feature inflation. We discard the entire 3D instance representation only when all associated patches are removed due to occlusion or scene changes.

The Merging Discriminator is trained on a dataset exceeding 5K rooms with annotated 3D instance segmentation data from ScanNet [47], HM3D [48], Matterport3D [22], and 3RScan [49]. Each patch is assigned a ground-truth instance ID by nearest neighbor matching to annotated points. For training, the discriminator outputs a binary classification indicating whether a 2D-3D instance pair corresponds to the same object instance. This training encourages robust multi-view instance feature consistency despite partial feature updates.

𝓛_segm = (1/J) ∑_{j=1}^{J} ∑_{k=1}^{K} CrossEntropy(MergingDiscriminator(ℰ_j^{2D}, ℰ_k^{3D}, D_{j,k}), 𝒢_{j,k}). (4)

The function MergingDiscriminator(·) is an MLP network which takes as input the 2D instance features ℰ_j^{2D}, 3D instance features ℰ_k^{3D} and their Euclidean distance D_{j,k}, and outputs a two-dimensional logit vector. After extensive pre-training, the function MergingDiscriminator(·) efficiently integrates 2D instances into existing 3D instances to maintain multi-view and geometrically consistent 3D representations that can be updated.

Feature Distillation and Language Alignment for 3D Instances. To align 3D instances with language semantics, we leverage contrastive learning on large-scale 3D–language pairs from SceneVerse [50] and g3D-LF [15]. Given a 3D instance feature ℰ_i and its corresponding annotated language description feature T_i extracted from the CLIP text encoder, we treat T_i as the positive sample


![Figure 18](image_18.jpg)
<!-- Caption: Method, index 18 -->



![Figure 19](image_19.jpg)
<!-- Caption: Method, index 19 -->


and descriptions of other instances serve as negatives:

$$\mathcal{L}_{\text{instance\_text}} = \frac{1}{I} \sum_{i=1}^{I} \text{CrossEntropy}\left(\left\{\text{CosSim}(\mathcal{O}_i, T_j)/\tau\right\}_{j=1}^{J}, i\right).$$
(5)

However, the generalization ability is limited since the scale of 3D-language data remains substantially smaller than that of image-language datasets: millions vs. billions [11]. We thus further enhance generalization by distilling visual knowledge from CLIP [11] into our Dynam3D model:

$$\mathcal{L}_{\text{instance\_distillation}} = \frac{1}{I} \sum_{i=1}^{I} \text{CrossEntropy}\left(\left\{\text{CosSim}(\mathcal{O}_i, \mathcal{O}_j^{gt})/\tau\right\}_{j=1}^{J}, i\right).$$
(6)

To obtain the ground-truth instance feature $\mathcal{O}_j^{gt}$ for distillation, we apply FastSAM to generate 2D instance masks and adopt the Masked Average Pooling (MAP) strategy from Feature Splatting [51] to average-pool patch-level features within each instance mask and obtain $\mathcal{O}_j^{gt}$. However, we observe that the instance-level features extracted in this strategy are interfered by noise from the overall image background. The ground-truth instance features of the same 3D instance obtained from different views exhibit a significant gap, which greatly affects the effectiveness of distillation since one of our goals is to achieve multi-view consistency in the representation of 3D instances. Consequently, we propose a strategy of Subspace Contrastive Learning:

$$\mathcal{L}_{\text{subspace\_distillation}} = \frac{1}{I} \sum_{i=1}^{I} \text{CrossEntropy}\left(\left\{\text{CosSim}\big((\mathcal{O}_i - \mathcal{V}_j), (\mathcal{O}_j^{gt} - \mathcal{V}_j)\big)/\tau\right\}_{j=1}^{J}, i\right),$$
(7)

where $\mathcal{V}_j$ is computed by average pooling all patch features within the given 2D view to yield the local semantic center of this view, i.e., semantic subspace. In Equation 6, instance features are optimized by maximizing cosine similarity with respect to the origin of the CLIP semantic space as the anchor. As a result, positive samples are pulled closer and negative samples are pushed farther apart. However, ground-truth bias of different views can impede this contrastive process. In Equation 7, we replace the origin anchor with the semantic center $\mathcal{V}_j$ of the view to mitigate the bias effect, impose a stronger optimization constraint, and promote a sparser feature space with improved representational capacity.

Feature Distillation and Language Alignment for 3D Zones. As shown in Figure 2 and Figure 3, we introduce the zone-level representations $\mathcal{Z}$ to further capture coarse-grained spatial layout context. Specifically, our Dynam3D partitions the 3D world coordinate space into uniform cubic zones (each spanning several cubic meters) and employs a zone encoder to aggregate the instance-level features $\mathcal{O}$ within each zone to obtain $\mathcal{Z}$. The encoding process is similar to Equation 3. For feature distillation at the zone level, our Dynam3D adopts a relatively simple strategy: it uses a zone encoder to aggregate 3D instances that belong to the same 2D view, and then aligns the aggregated zone representation $\mathcal{Z}$ with the CLIP feature of the entire 2D view. Although the aggregated instances do not strictly come from the same cube zone, this approach ensures the quality of the distilled ground-truth features. For zone-level language alignment, we follow g3D-LF [15] to use Fine-grained Contrastive Learning for long-text contrastive supervision. Specifically, we compute an affinity matrix between the instance representations within a zone and the long-text representations to measure similarity, and then perform contrastive learning across different zones and texts.

3.2 3D Vision-Language Model for Navigation

As illustrated in Figure 2, Dynam3D constructs hierarchical 3D representations, spanning from fine-grained object instances to large-scale environmental zones. Leveraging these multi-level 3D representations as perceptual inputs, we introduce a dedicated 3D Vision-Language Model (3D-VLM) tailored for VLN tasks.

Encoding Panoramic 3D Patch Tokens via Generalizable Feature Fields. To effectively capture fine-grained geometric and semantic information within the surrounding panorama of the agent, we build upon the approach of g3D-LF [15] and adopt a generalizable feature field model to predict agent-centric 3D patch tokens. Specifically, we uniformly sample 12×48 rays covering a 90° vertical and 360° horizontal field-of-view around the agent, rendering both the 3D patch features $\hat{\mathbf{g}}$ and their corresponding depth estimates. These features, with positional embeddings, provide rich and spatially grounded representations of the scene geometry and semantics from the egocentric viewpoint.

Multimodal Reasoning and Action Prediction. To balance multimodal reasoning capabilities with computational efficiency, the 3.8 billion-parameter LLaVA-Phi-3-mini [52, 53] is integrated into

the proposed 3D-VLM framework. Since the 3D tokens (patch-instance-zone) are aligned with the semantic space of CLIP-ViT-L/14@336px [11], the strong multimodal understanding and reasoning abilities of this 2D-VLM can be effectively transferred to the 3D domain.

As shown in Figure 2, the input and output format of our 3D-VLM is:

Input: <user> {patch_tokens}{instance_tokens}{zone_tokens}{instruction_tokens} {history_action_tokens} <end> <assistant>

Output: Next action: 1) Turn left θ degree. 2) Turn right θ degree. 3) Forward d cm. 4) Stop.

<user> is a special token in LLaVA [36] used to indicate that the following tokens are context. <end> marks the end of a sequence. <assistant> indicates that the following tokens are the response of the model. To encode the relative positional relationship between 3D tokens and the agent, the relative coordinates [x_c, y_c, z_c], i.e., camera coordinates of each 3D token to the agent, are calculated along with the relative distance D_c and the relative horizontal angle θ_c. [x_c, y_c, z_c, D_c, cos(θ_c), sin(θ_c)] of each token are then fed into an MLP network to generate the corresponding positional embeddings.

The 3D patch tokens {patch_tokens} rendered from the generalizable feature field are organized in a row-major order of 12×48 tokens, starting from the rays directly behind the agent and proceeding clockwise. This strategy is similar to that used in the pre-trained LLaVA-Phi-3-mini model [52, 53] when handling a single-view image. The instance tokens {instance_tokens} and zone tokens {zone_tokens} are sorted by their Euclidean distance to the agent from nearest to farthest. As shown in Figure 2, 3D-VLM outputs atomic actions with turning angles or movement distances. The history actions {history_action_tokens} store the four most recent action texts, padding with the special token <none> if fewer than four are available.


## Experiments


## 4 Experiments

### 4.1 Comparison with SOTA Methods

As shown in Tables 1 and 2, we evaluate the navigation performance of our Dynam3D across three distinct continuous-environment VLN benchmarks. Specifically, the R2R-CE dataset (Table 1) provides step-by-step and following instructions. Compared to prior state-of-the-art methods, e.g., g3D-LF and Uni-NaVid, our Dynam3D achieves modest improvements of approximately 1-2% in navigation success rate (SR), aligning with expected margins given the task difficulty. Although Dynam3D uses a large model, it maintains a smaller parameter footprint (3.8B vs. 7B) relative to the video-based Uni-NaVid.

To provide fair comparison on the more challenging and realistic benchmarks such as REVERIE-CE with coarse-grained instructions and NavRAG-CE requiring complex user demand understanding, we retrained NaVid and g3D-LF on our training dataset for evaluation (Table 2). Our Dynam3D demonstrates incremental improvements, outperforming NaVid by approximately 3-5% in Success Rate (SR) on REVERIE-CE and NavRAG-CE. The detailed experimental setup is described in the supplementary materials.

Table 1: Evaluation of VLN on R2R-CE with monocular setting. ∗ denotes zero-shot method.

| Methods             | LLM | Scene Representation           | R2R-CE Val<br/>NE↓ | R2R-CE Val<br/>OSR↑ | R2R-CE Val<br/>SR↑ | R2R-CE Val<br/>SPL↑ | R2R-CE Test<br/>NE↓ | R2R-CE Test<br/>OSR↑ | R2R-CE Test<br/>SR↑ | R2R-CE Test<br/>SPL↑ |
| ------------------- | --- | ------------------------------ | ------------------ | ------------------- | ------------------ | ------------------- | ------------------- | -------------------- | ------------------- | -------------------- |
| CM² [54]            | ×   | Semantic Map                   | 7.02               | 41.5                | 34.3               | 27.6                | 7.7                 | 39                   | 31                  | 24                   |
| WS-MGMap [55]       | ×   | Multi-Granularity Semantic Map | 6.28               | 47.6                | 38.9               | 34.3                | 7.11                | 45                   | 35                  | 28                   |
| InstructNav∗ [56]   | ✓   | Semantic Value Map             | 6.89               | -                   | 31                 | 24                  | -                   | -                    | -                   | -                    |
| AO-Planner∗ [57]    | ✓   | Visual Affordance Prompts      | 6.95               | 38.3                | 25.5               | 16.6                | -                   | -                    | -                   | -                    |
| NaVid [5]           | ✓   | Video Frames                   | 5.47               | 49.1                | 37.4               | 35.9                | -                   | -                    | -                   | -                    |
| VLN-3DFF [28]       | ×   | Feature Fields                 | 5.95               | 55.8                | 44.9               | 30.4                | 6.24                | 54.4                 | 43.7                | 28.9                 |
| g3D-LF [15]         | ×   | Feature Fields                 | 5.70               | 59.5                | 47.2               | 34.6                | 6.00                | 57.5                 | 46.3                | 32.2                 |
| Uni-NaVid [6]       | ✓   | Multi-Granularity Video Frames | 5.58               | 53.3                | 47.0               | 42.7                | -                   | -                    | -                   | -                    |
| Dynam3D (Ours)      | ✓   | 3D Patch-Instance-Zone Tokens  | 5.34               | 62.1                | 52.9               | 45.7                | 5.53                | 60.4                 | 51.4                | 44.8                 |

### 4.2 Experiments on Pre-exploration and Lifelong Memory

As shown in Table 3, we additionally evaluate the performance under the Pre-exploration and Lifelong Memory settings to further demonstrate the advantages of our Dynam3D. The pre-explored panoramic

Table 2: Evaluation of VLN on REVERIE-CE and NavRAG-CE with monocular setting. * denotes zero-shot method.

| Methods             | LLM | Scene Representation          | REVERIE-CE Val NE↓ | REVERIE-CE Val OSR↑ | REVERIE-CE Val SR↑ | REVERIE-CE Val SPL↑ | NavRAG-CE Val NE↓ | NavRAG-CE Val OSR↑ | NavRAG-CE Val SR↑ | NavRAG-CE Val SPL↑ |
| ------------------- | --- | ----------------------------- | ------------------ | ------------------- | ------------------ | ------------------- | ----------------- | ------------------ | ----------------- | ------------------ |
| InstructNav* [56]   | ✓   | Semantic Value Map            | 7.44               | 31.5                | 25.2               | 19.1               | 9.83              | 24.1               | 17.4              | 10.9               |
| NaVid [5]           | ✓   | Video Frames                  | 6.74               | 36.3                | 26.6               | 20.8               | 9.35              | 29.6               | 19.4              | 13.9               |
| g3D-LF [15]         | ×   | Feature Fields                | 6.50               | 41.6                | 34.4               | 23.8               | 8.85              | 31.8               | 21.4              | 13.5               |
| Dynam3D (Ours)      | ✓   | 3D Patch-Instance-Zone Tokens | 6.22               | 48.9                | 40.1               | 28.5               | 8.12              | 38.4               | 24.7              | 18.8               |

Images from the pre-exploration setting are collected at the navigable viewpoints annotated in the Matterport3D [22] dataset, which are then used to construct the Patch-Instance-Zone representations of the entire scene. For the lifelong memory setting, we group the evaluation episodes by scene, with navigation samples from the same scene evaluated consecutively within a group. For each scene, previously stored 3D representations can be leveraged in subsequent episodes to simulate gradual familiarization of the agent with the environment during task execution.

Table 3 shows improvements in Success Rate (SR) on R2R-CE and REVERIE-CE under pre-exploration and lifelong memory settings. However, experimental details regarding agent familiarity with the test scenes and overlap with training data are limited, raising concerns about potential overfitting. Further controls are required to verify that gains reflect genuine lifelong memory effects and not environment memorization or data leakage. Without stricter evaluation protocols, the reported substantial performance increases relative to NaVid [5] may be inflated.

Table 3: Evaluation of VLN for pre-exploration and lifelong memory. Pre-exploration allows agents to scan and encode environmental representations before evaluation, while lifelong memory enables agents to retain the environmental representations of previous episodes for subsequent episodes.

| Methods        | Pre-exploration | Lifelong Memory | R2R-CE Val NE↓ | R2R-CE Val OSR↑ | R2R-CE Val SR↑ | R2R-CE Val SPL↑ | REVERIE-CE Val NE↓ | REVERIE-CE Val OSR↑ | REVERIE-CE Val SR↑ | REVERIE-CE Val SPL↑ |
| -------------- | --------------- | --------------- | -------------- | --------------- | -------------- | --------------- | ------------------ | ------------------- | ------------------ | ------------------- |
| NaVid [5]      | ×               | ×               | 5.47           | 49.1            | 37.4           | 35.9            | 6.74               | 36.3                | 26.6               | 20.8               |
| g3D-LF [15]    | ×               | ×               | 5.70           | 59.5            | 47.2           | 34.6            | 6.50               | 41.6                | 34.4               | 23.8               |
| g3D-LF [15]    | ✓               | ✓               | 5.46           | 62.5            | 51.8           | 39.9            | 6.44               | 43.3                | 37.1               | 25.9               |
| Dynam3D (Ours) | ×               | ×               | 5.34           | 62.1            | 52.9           | 45.7            | 6.22               | 48.9                | 40.1               | 28.5               |
| Dynam3D (Ours) | ✓               | ×               | 5.04           | 66.2            | 57.1           | 52.7            | 6.09               | 56.8                | 48.1               | 37.3               |
| Dynam3D (Ours) | ×               | ✓               | 5.21           | 64.4            | 55.6           | 48.1            | 6.31               | 52.8                | 45.0               | 32.7               |
| Dynam3D (Ours) | ✓               | ✓               | 5.11           | 67.2            | 58.4           | 50.4            | 6.02               | 56.4                | 49.5               | 38.1               |

4.3 Experiments on Real-World and Dynamic Environments

As shown in Tables 4, 5 and Figure 4, we evaluate our Dynam3D on real-world static and dynamic environments using the Hello Robot Stretch 3. Each setting includes 20 test cases, and navigation is considered successful if the robot stops within 1 meter of the target. In the static environment (Table 4), Dynam3D achieves higher success rates than baselines, reaching 70% after pre-exploration. In the dynamic setting (Figure 4 and Table 5), the target is moved manually once the robot approaches, simulating environmental changes. However, ethical approval, participant consent, and safety protocols for these manipulations are not discussed, limiting transparency on real-world deployment risks. Details on experimental controls and safety measures are provided in the supplementary materials.

Table 4: Real-world navigation experiments in static environments.

| Methods           | NE↓ | OSR↑ | SR↑ |
| ----------------- | --- | ---- | --- |
| NaVid             | 2.2 | 45   | 35  |
| g3D-LF            | 3.1 | 40   | 30  |
| Dynam3D           | 1.4 | 65   | 55  |
| + Pre-exploration | 0.8 | 75   | 70  |

Table 5: Real-world navigation experiments in dynamic environments.

| Methods           | NE↓ | OSR↑ | SR↑ |
| ----------------- | --- | ---- | --- |
| NaVid             | 3.6 | 45   | 20  |
| g3D-LF            | 4.6 | 35   | 10  |
| Dynam3D           | 1.9 | 60   | 45  |
| + Pre-exploration | 1.4 | 75   | 45  |

4.4 Analysis for Dynamic Representation and Lifelong Memory

In our Dynam3D, the agent discards outdated representations within its current field of view and adds or updates new ones to ensure real-time understanding of dynamic environments with each new observation. However, this does not conflict with its advantage of maintaining lifelong memory.

Figure 4: A demonstration of navigation in a dynamic real-world environment.

Features corresponding to regions outside the robot’s current view are preserved and not updated, serving as long-term memory. While this approach retains previously seen scene information, no explicit mechanisms prevent memory saturation or handle outdated long-term features, potentially limiting scalability. This design allows some adaptability in dynamic environments but may introduce trade-offs between update fidelity and memory usage, warranting further quantitative analysis.

## 4.5 Computational Cost and Real-Time Analysis

We evaluate computational cost on the R2R-CE dataset using a single NVIDIA RTX 4090 GPU. During training, each navigation step takes 455 ms (~0.46 seconds) on average: 83 ms for 3D representation updates, 315 ms for the large language model, and 57 ms for other operations. During inference, the average step time increases to 649 ms (~0.65 seconds) with 83 ms for 3D representation updates, 540 ms for large language model inference, and 26 ms for the remaining components. Most navigation episodes can be completed within 20 to 40 navigation steps, and our navigation system supports real-time 3D representation updates and navigation action prediction for efficient training and inference.

## 4.6 Ablation Study

Table 6: Ablation study of Dynam3D on R2R-CE and REVERIE-CE Val Unseen benchmarks. We analyze the contribution of different components: input view type (currently observed Single View patches vs. Rendered Panoramic patches), high-level 3D representations (Instance and Zone), and the Subspace Align. loss used for supervising instance representations.

| Single View | Rendered Pano | Instance | Subspace Align. | Zone | R2R-CE Val Unseen<br/>NE↓ | R2R-CE Val Unseen<br/>OSR↑ | R2R-CE Val Unseen<br/>SR↑ | R2R-CE Val Unseen<br/>SPL↑ | REVERIE-CE Val Unseen<br/>NE↓ | REVERIE-CE Val Unseen<br/>OSR↑ | REVERIE-CE Val Unseen<br/>SR↑ | REVERIE-CE Val Unseen<br/>SPL↑ |
| ----------- | ------------- | -------- | ---------------- | ---- | ------------------------- | -------------------------- | ------------------------- | -------------------------- | ----------------------------- | ------------------------------ | ----------------------------- | ------------------------------ |
| ×           | ×             | ✓        | ✓                | ✓    | 7.59                      | 43.9                       | 32.7                      | 21.1                       | 8.76                          | 27.4                           | 19.3                          | 11.4                           |
| ✓           | ×             | ✓        | ✓                | ✓    | 6.14                      | 51.7                       | 43.1                      | 33.9                       | 7.85                          | 35.9                           | 28.4                          | 21.8                           |
| ×           | ✓             | ×        | ×                | ×    | 5.63                      | 51.1                       | 45.7                      | 40.2                       | 7.89                          | 34.8                           | 25.7                          | 17.8                           |
| ×           | ✓             | ×        | ×                | ✓    | 5.77                      | 54.7                       | 47.9                      | 43.3                       | 6.58                          | 38.5                           | 26.8                          | 21.0                           |
| ×           | ✓             | ✓        | ✓                | ×    | 5.26                      | 61.8                       | 52.4                      | 45.7                       | 6.37                          | 46.2                           | 39.3                          | 26.2                           |
| ×           | ✓             | ✓        | ×                | ✓    | 5.44                      | 58.8                       | 50.7                      | 43.2                       | 6.31                          | 45.1                           | 38.4                          | 25.8                           |
| ×           | ✓             | ✓        | ✓                | ✓    | 5.34                      | 62.1                       | 52.9                      | 45.7                       | 6.22                          | 48.9                           | 40.1                          | 28.5                           |

Contribution of 3D Representations. From the full model (last row), removing the rendered panoramic patch tokens (row 1), instance tokens (row 4), zone tokens (row 5), or both instance and zone tokens (row 3) leads to performance drops of varying degrees. The results show that the impact on performance follows the order: Patch > Instance > Zone tokens.

Specifically, when both 3D instance and zone representations are removed and only 3D patch tokens from the feature fields [15] are used (row 3), there is a substantial performance drop, particularly on REVERIE-CE, where SR decreases by nearly 15%. This highlights the critical role of the hierarchical instance–zone representation in supporting effective navigation and large-scale exploration, as local patch-level features alone provide limited spatial coverage.


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


Importance of Subspace Alignment. Without Subspace Alignment supervision (row 6), the navigation performance significantly decreases. This highlights the limitations of naive CLIP feature distillation for 3D instance supervision. Subspace Contrastive Learning effectively mitigates the instance feature bias from different views.

Value of Rendered Panoramic Views. As analyzed in VLN-3DFF [28], rendered panoramic patch tokens substantially mitigate the limited field of view caused by monocular cameras on most robots. Comparing row 2 and last row in Table 6, replacing the rendered panoramic tokens with CLIP patch features extracted from the current monocular view results in about a 10% drop in navigation success rates on both R2R-CE and REVERIE-CE.

Table 7: Robustness study on the R2R-CE Val Unseen benchmark with simulated SLAM noise and depth noise.

| SLAM Noise | Depth Noise | NE↓  | OSR↑ | SR↑  | SPL↑ |
| ---------- | ----------- | ---- | ---- | ---- | ---- |
| ×          | ×           | 5.34 | 62.1 | 52.9 | 45.7 |
| ✓          | ×           | 5.46 | 61.3 | 52.1 | 45.0 |
| ×          | ✓           | 5.39 | 61.9 | 52.3 | 45.2 |
| ✓          | ✓           | 5.47 | 60.7 | 50.9 | 44.1 |

Robustness to Noise. To further validate the model’s navigation performance in noisy environments, we introduce different types of noise into the simulator and evaluate Dynam3D’s performance. We simulate the potential noise from a SLAM system in localization and pose estimation, as well as the measurement errors inherent in depth cameras.

To simulate SLAM noise, we introduce localization noise uniformly sampled from −5 cm to +5 cm and orientation noise from −3° to +3° at each step. Depth camera noise is added with a magnitude proportional to distance.

In addition to these, we perform ablation by degrading the quality of 3D instance and zone embeddings through feature dropout and spatial jitter to evaluate sensitivity. Table 7 shows that while navigation success rate (SR) decreases with noise addition, it degrades significantly more when 3D representation quality is compromised. This indicates the navigation performance heavily depends on the fidelity of the hierarchical 3D representations, exposing potential failure modes. A significant reason is that our Dynam3D leverages 2D foundation models (e.g., CLIP [11], FastSAM [12]) to extract visual semantic features, which are unaffected by SLAM and depth noise. Even with minor spatial position deviations due to noise, the 3D-aware representations obtained by our model through the hierarchical encoding of Patch-Instance-Zone remain sufficient for perceiving spatial geometry and scene layout at the object and zone levels.


## Conclusion


5 Conclusion

We present Dynam3D, a dynamic hierarchical 3D representation framework designed for monocular vision-and-language navigation. By aligning patch-instance-zone features with language semantics and enabling incremental scene updates, Dynam3D aims to improve spatial understanding, memory retention, and adaptability in dynamic environments. While our experimental results indicate competitive performance across VLN benchmarks and initial real-world tests, further work is needed to fully assess generalization capabilities. Overall, Dynam3D contributes to ongoing efforts in incorporating structured 3D representations in embodied navigation.

Limitations. Our Dynam3D predicts navigation actions without explicitly outputting the coordinate of target instance, limiting its applicability to some tasks such as mobile manipulation. Moreover, it lacks capabilities for question answering, dialogue, and task updates, showing potential directions for better navigation agents.


## Appendix


# Acknowledgement

This research work is supported by the Agency for Science, Technology and Research (A*STAR) under its MTC Programmatic Funds (Grant No. M23L7b0021), and the Tier 2 grant MOE-T2EP20124-0015 from the Singapore Ministry of Education.


## References


# References

[1] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton van den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3674–3683, 2018.

[2] Yuankai Qi, Qi Wu, Peter Anderson, Xin Wang, William Yang Wang, Chunhua Shen, and Anton van den Hengel. REVERIE: Remote embodied visual referring expression in real indoor environments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9982–9991, 2020.

[3] Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Vision-and-language navigation in continuous environments. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVIII, pages 104–120. Springer, 2020.

[4] Zihan Wang, Yaohui Zhu, Gim Hee Lee, and Yachun Fan. NavRAG: Generating user demand instructions for embodied navigation through retrieval-augmented LLM. arXiv preprint arXiv:2502.11142, 2025.

[5] Jiazhao Zhang, Kunyu Wang, Rongtao Xu, Gengze Zhou, Yicong Hong, Xiaomeng Fang, Qi Wu, Zhizheng Zhang, and He Wang. NaviD: Video-based VLM plans the next step for vision-and-language navigation. In Proceedings of Robotics: Science and Systems (RSS), 2024.

[6] Jiazhao Zhang, Kunyu Wang, Shaoan Wang, Minghan Li, Haoran Liu, Songlin Wei, Zhongyuan Wang, Zhizheng Zhang, and He Wang. Uni-NaviD: A video-based vision-language-action model for unifying embodied navigation tasks. arXiv preprint arXiv:2412.06224, 2024.

[7] An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Xueyan Zou, Jan Kautz, Erdem Biyik, Hongxu Yin, Sifei Liu, and Xiaolong Wang. NavILA: Legged robot vision-language-action model for navigation. In RSS, 2025.

[8] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-LLaVA: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.

[9] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. VILA: On pre-training for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26689–26699, 2024.

[10] Wei-Lin Chiang, Zhuohan Li, Ziqing Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, et al. Vicuna: An open-source chatbot impressing GPT-4 with 90% ChatGPT quality. See https://vicuna.lmsys.org (accessed 14 April 2023), 2(3):6, 2023.

[11] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021.

[12] Xu Zhao, Wenchao Ding, Yongqi An, Yinglong Du, Tao Yu, Min Li, Ming Tang, and Jinqiao Wang. Fast Segment Anything. arXiv preprint arXiv:2306.12156, 2023.

[13] Xiuwei Xu, Huangxing Chen, Linqing Zhao, Ziwei Wang, Jie Zhou, and Jiwen Lu. Embodied-SAM: Online segment any 3D thing in real time. In The Thirteenth International Conference on Learning Representations.

[14] Seungjun Lee, Yuyang Zhao, and Gim Hee Lee. Segment Any 3D Object with Language. In The Thirteenth International Conference on Learning Representations (ICLR), 2025.

[15] Zihan Wang and Gim Hee Lee. G3D-LF: Generalizable 3D-Language Feature Fields for Embodied Tasks. arXiv preprint arXiv:2411.17030, 2024.

[16] Yicong Hong, Qi Wu, Yuankai Qi, Cristian Rodriguez-Opazo, and Stephen Gould. VLN BERT: A Recurrent Vision-and-Language BERT for Navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1643–1653, 2021.

[17] Shizhe Chen, Pierre-Louis Guhur, Cordelia Schmid, and Ivan Laptev. History Aware Multimodal Transformer for Vision-and-Language Navigation. Advances in Neural Information Processing Systems, 34:5834–5847, 2021.

[18] Shizhe Chen, Pierre-Louis Guhur, Makarand Tapaswi, Cordelia Schmid, and Ivan Laptev. Think Global, Act Local: Dual-Scale Graph Transformer for Vision-and-Language Navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16537–16547, 2022.

[19] Liuyi Wang, Zongtao He, Ronghao Dang, Mengjiao Shen, Chengju Liu, and Qijun Chen. Vision-and-Language Navigation via Causal Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13139–13150, 2024.

[20] Yanyuan Qiao, Yuankai Qi, Yicong Hong, Zheng Yu, Peng Wang, and Qi Wu. HOP+: History-Enhanced and Order-Aware Pre-Training for Vision-and-Language Navigation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(7):8524–8537, 2023.

[21] Rui Liu, Xiaohan Wang, Wenguan Wang, and Yi Yang. Bird’s-Eye-View Scene Graph for Vision-Language Navigation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10968–10980, 2023.

[22] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3D: Learning from RGB-D Data in Indoor Environments. In International Conference on 3D Vision (3DV), 2017.

[23] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, et al. Habitat: A Platform for Embodied AI Research. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9339–9347, 2019.

[24] Yicong Hong, Zun Wang, Qi Wu, and Stephen Gould. Bridging the Gap Between Learning in Discrete and Continuous Environments for Vision-and-Language Navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022.

[25] Zihan Wang, Xiangyang Li, Jiahao Yang, Yeqi Liu, and Shuqiang Jiang. GridMM: Grid Memory Map for Vision-and-Language Navigation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15625–15636, 2023.

[26] Dong An, Hanqing Wang, Wenguan Wang, Zun Wang, Yan Huang, Keji He, and Liang Wang. ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[27] Dong An, Yuankai Qi, Yangguang Li, Yan Huang, Liang Wang, Tieniu Tan, and Jing Shao. BEV-BERT: Multimodal Map Pre-Training for Language-Guided Navigation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2737–2748, 2023.

[28] Zihan Wang, Xiangyang Li, Jiahao Yang, Yeqi Liu, and Shuqiang Jiang. Sim-to-Real Transfer via 3D Feature Fields for Vision-and-Language Navigation. In 8th Annual Conference on Robot Learning (CoRL), 2024.

[29] Yanyuan Qiao, Wenqi Lyu, Hui Wang, Zixu Wang, Zerui Li, Yuan Zhang, Mingkui Tan, and Qi Wu. Open-NAV: Exploring Zero-Shot Vision-and-Language Navigation in Continuous Environments with Open-Source LLMs. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2025.

[30] Sixian Zhang, Xinhang Song, Xinyao Yu, Yubing Bai, Xinlong Guo, Weijie Li, and Shuqiang Jiang. HOZ++: Versatile hierarchical object-to-zone graph for object navigation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[31] Duo Zheng, Shijia Huang, Lin Zhao, Yiwu Zhong, and Liwei Wang. Towards learning a generalist model for embodied navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13624–13634, 2024.

[32] Gengze Zhou, Yicong Hong, Zun Wang, Xin Eric Wang, and Qi Wu. NavGPT-2: Unleashing navigational reasoning capability for large vision-language models. arXiv preprint arXiv:2407.12366, 2024.

[33] Jiaqi Chen, Bingqian Lin, Ran Xu, Zhenhua Chai, Xiaodan Liang, and Kwan-Yee K. Wong. MapGPT: Map-guided prompting with adaptive path planning for vision-and-language navigation. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 2024.

[34] Sixian Zhang, Xinyao Yu, Xinhang Song, Xiaohan Wang, and Shuqiang Jiang. Imagine before go: Self-supervised generative map for object goal navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16414–16425, 2024.

[35] Jiaqi Chen, Bingqian Lin, Xinmin Liu, Lin Ma, Xiaodan Liang, and Kwan-Yee K. Wong. Affordance-oriented planning using foundation models for continuous vision-language navigation. In Proceedings of the AAAI Conference on Artificial Intelligence, 2025.

[36] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in Neural Information Processing Systems, 36:34892–34916, 2023.

[37] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024.

[38] Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu, Hao Fei, Hongyuan Zhu, Jiayuan Fan, and Tao Chen. LL3DA: Visual interactive instruction tuning for omni-3D understanding, reasoning, and planning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26428–26438, 2024.

[39] Haifeng Huang, Yilun Chen, Zehan Wang, Rongjie Huang, Runsen Xu, Tai Wang, Luping Liu, Xize Cheng, Yang Zhao, Jiangmiao Pang, et al. Chat-Scene: Bridging 3D scene and large language models with object identifiers. arXiv preprint arXiv:2312.08168, 2023.

[40] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3D world. In Proceedings of the International Conference on Machine Learning (ICML), 2024.

[41] Rao Fu, Jingyu Liu, Xilun Chen, Yixin Nie, and Wenhan Xiong. Scene-LLM: Extending language models for 3D visual understanding and reasoning. arXiv preprint arXiv:2403.11401, 2024.

[42] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3D-LLM: Injecting the 3D world into large language models. Advances in Neural Information Processing Systems, 36:20482–20494, 2023.

[43] Duo Zheng, Shijia Huang, and Liwei Wang. Video-3D LLM: Learning position-aware video representation for 3D scene understanding. arXiv preprint arXiv:2412.00493, 2024.

[44] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. LLaVA-3D: A simple yet effective pathway to empowering LMMs with 3D-awareness. arXiv preprint arXiv:2409.18125, 2024.

[45] Zihan Wang, Xiangyang Li, Jiahao Yang, Yeqi Liu, Junjie Hu, Ming Jiang, and Shuqiang Jiang. Look-ahead exploration with neural radiance representation for continuous vision-language navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13753–13762, 2024.

[46] Rizhao Qiu, Yafei Hu, Ge Yang, Yuchen Song, Yang Fu, Jianglong Ye, Jiteng Mu, Ruihan Yang, Nikolay Atanasov, Sebastian Scherer, et al. Learning generalizable feature fields for mobile manipulation. arXiv preprint arXiv:2403.07563, 2024.
[47] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-annotated 3D reconstructions of indoor scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5828–5839, 2017.
[48] Karmesh Yadav, Ram Ramrakhya, Santhosh Kumar Ramakrishnan, Theo Gervet, John Turner, Aaron Gokaslan, Noah Maestre, Angel X. Chang, Dhruv Batra, Manolis Savva, et al. Habitat-Matterport 3D Semantics Dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4927–4936, 2023.
[49] Johanna Wald, Armen Avetisyan, Nassir Navab, Federico Tombari, and Matthias Nießner. RIO: 3D object instance re-localization in changing indoor environments. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7658–7667, 2019.
[50] Baoxiong Jia, Yixin Chen, Huangyue Yu, Yan Wang, Xuesong Niu, Tengyu Liu, Qing Li, and Siyuan Huang. SceneVerse: Scaling 3D vision-language learning for grounded scene understanding. In European Conference on Computer Vision (ECCV), 2024.
[51] Rizhao Qiu, Ge Yang, Weijia Zeng, and Xiaolong Wang. Feature splatting: Language-driven physics-based scene synthesis and editing. arXiv preprint arXiv:2404.01223, 2024.
[52] XTuner Contributors. XTuner: A toolkit for efficiently fine-tuning LLMs. https://github.com/InternLM/xtuner, 2023.
[53] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 Technical Report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024.
[54] Georgios Georgakis, Karl Schmeckpeper, Karan Wanchoo, Soham Dan, Eleni Miltsakaki, Dan Roth, and Kostas Daniilidis. Cross-modal map learning for vision and language navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15460–15470, 2022.
[55] Peihao Chen, Dongyu Ji, Kunyang Lin, Runhao Zeng, Thomas Li, Mingkui Tan, and Chuang Gan. Weakly-supervised multi-granularity map learning for vision-and-language navigation. Advances in Neural Information Processing Systems, 35:38149–38161, 2022.
[56] Yuxing Long, Wenzhe Cai, Hongcheng Wang, Guanqi Zhan, and Hao Dong. InstructNav: Zero-shot system for generic instruction navigation in unexplored environment. In 8th Annual Conference on Robot Learning, 2024.
[57] Jiaqi Chen, Bingqian Lin, Xinmin Liu, Lin Ma, Xiaodan Liang, and Kwan-Yee K. Wong. Affordances-oriented planning using foundation models for continuous vision-language navigation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 23568–23576, 2025.
[58] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Yuri Feigin, Peter Fu, Thomas Gebauer, Daniel Kurz, Tal Dimry, Brandon Joffe, Arik Schwartz, et al. ARKitScenes: A diverse real-world dataset for 3D indoor scene understanding using mobile RGB-D data. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1).
[59] Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. Structured3D: A large photo-realistic dataset for structured 3D modeling. In Proceedings of the European Conference on Computer Vision (ECCV), 2020.
[60] Zun Wang, Jialu Li, Yicong Hong, Yi Wang, Qi Wu, Mohit Bansal, Stephen Gould, Hao Tan, and Yu Qiao. Scaling data generation in vision-and-language navigation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12009–12020, 2023.

[61] Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. In *Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics*, pages 627–635. JMLR Workshop and Conference Proceedings, 2011.

[62] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive Learning Rates with Sublinear Memory Cost. In *International Conference on Machine Learning*, pages 4596–4604. PMLR, 2018.

[63] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training Deep Nets with Sublinear Memory Cost. *arXiv preprint arXiv:1604.06174*, 2016.

[64] Peiqi Liu, Zhanqiu Guo, Mohit Warke, Soumith Chintala, Chris Paxton, Nur Muhammad Mahi Shafiullah, and Lerrel Pinto. DynaMem: Online Dynamic Spatio-Semantic Memory for Open-World Mobile Manipulation. In *CoRL 2024 Workshop on Mastering Robot Manipulation in a World of Abundant Data*.


## Appendix


A Supplementary Material

## A.1 Datasets and Experimental Details

3D-Language Datasets and Training Details. To train the Dynam3D representation model, we follow SceneVerse [50] and g3D-LF [15] in collecting over 5K scenes with 2M language annotations from ScanNet [47], HM3D [48], Matterport3D [22], 3RScan [49], ARKitScenes [58], and Structured3D [59]. All datasets were used in compliance with relevant ethics guidelines and data privacy requirements where applicable, including consent for data collection and usage for research purposes. For each episode, we randomly sample sufficient posed RGB-D images from raw videos or the Habitat simulator [23] to construct and update the hierarchical patch-instance-zone representations. The updated representations after each observed frame are supervised using the losses defined in Section 3.1. We pre-train our Dynam3D representation model on the aforementioned dataset for 100K episodes (approximately 8 days) using four RTX 6000 Ada GPUs. The training is performed with a batch size of 4 and a learning rate of 1e-4.

Navigation Datasets and Training Details. To train our 3D-VLM with sufficient navigation data, we transfer datasets generated by ScaleVLN [60] and NavRAG [4] from discrete environments to the continuous Habitat simulator [23]. After removing samples with impassable paths, we obtain 4M+ instruction-trajectory pairs in continuous settings. For a comprehensive and fair evaluation, we evaluate our model on R2R-CE [3], REVERIE-CE and NavRAG-CE by transferring REVERIE [2] and NavRAG [4] datasets to continuous environments. To balance data quality and scale, we randomly sample model-generated data (ScaleVLN, NavRAG; 4M+) and human-annotated data (R2R-CE, REVERIE-CE; 20K+) at a 1:1 ratio during 3D-VLM training. Navigation training proceeds in two stages: 1) Imitation learning. The agent strictly follows ground-truth paths to enhance instruction following and multimodal alignment; 2) Exploration and correction. Following ETPNav [26], we adopt a waypoint predictor [24] to generate multiple candidate waypoints. We utilize the DAgger strategy [61, 18] to enhance error correction by deliberately introducing probabilistic deviations that mislead the agent towards incorrect waypoints. The agent is then guided back to the correct path, thereby strengthening its ability to recover from navigation errors. We pre-train the 3D-VLM model on the navigation datasets for 100K episodes (50K for stage one, 50K for stage two, approximately 9 days) using two RTX 6000 Ada GPUs. The training is performed with a batch size of 4 and a learning rate of 1e-6. During training, all parameters of the 3.8B LLaVA-Phi-3-mini [52, 53] are optimized, except the generalizable feature field model [15] and the pre-trained Dynam3D representation model. To mitigate memory consumption and enable efficient training of large models, we employ the Adafactor optimizer [62] in conjunction with Gradient Checkpointing [63].

Details of Real-world Navigation. We employ the Hello Robot Stretch 3 for real-world navigation experiments, leveraging its real-time localization and pose estimation capabilities. An Intel RealSense D435i RGB-D camera is mounted on the robot’s head to facilitate 3D scene representation construction and incremental updates. Our real-world experimental framework is adapted from DynaMem [64], with extensions for obstacle avoidance and movement. The model is deployed on a workstation equipped with an NVIDIA RTX 4090 GPU and 64GB of RAM, and communicates with the robot over a local area network established via a WiFi access point. The experimental environment consists of a home-style setting constructed for robot evaluation, encompassing a living room, kitchen, meeting room, and office. To ensure a fair comparison under the unseen setting, none of the objects or rooms within the environment are included in the training data.