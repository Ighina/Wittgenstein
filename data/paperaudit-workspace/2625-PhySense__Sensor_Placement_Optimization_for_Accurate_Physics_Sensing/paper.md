
## Abstract


PhySense: Sensor Placement Optimization for Accurate Physics Sensing

Yuezhou Ma, Haixu Wu, Hang Zhou, Huikun Weng, Jianmin Wang, Mingsheng Long ⊠

School of Software, BNRist, Tsinghua University, China

{mayuezhou20,wuhaixu98}@gmail.com, {zhou-h23,wenghk22}@mails.tsinghua.edu.cn, {jimwang,mingsheng}@tsinghua.edu.cn

Abstract

Physics sensing plays a central role in many scientific and engineering domains, which inherently involves two coupled tasks: reconstructing dense physical fields from sparse observations and optimizing scattered sensor placements to observe maximum information. While deep learning has made rapid advances in sparse-data reconstruction, existing methods generally omit optimization of sensor placements, leaving the mutual enhancement between reconstruction and placement on the shelf. To change this suboptimal practice, we propose PhySense, a synergistic two-stage framework that learns to jointly reconstruct physical fields and to optimize sensor placements, both aiming for accurate physics sensing. The first stage involves a flow-based generative model enhanced by cross-attention to adaptively fuse sparse observations. Leveraging the reconstruction feedback, the second stage performs sensor placement via projected gradient descent to satisfy spatial constraints. We further prove that the learning objectives of the two stages are consistent with classical variance-minimization principles, providing theoretical guarantees. Extensive experiments across three challenging benchmarks, especially a 3D geometry dataset, demonstrate that PhySense improves physics sensing accuracy and identifies sensor placements that align with physical intuition. However, certain performance gains depend on dataset characteristics and assumptions detailed in the paper. Code is available at this repository: https://github.com/thuml/PhySense.


## Introduction


1 Introduction

Physics sensing remains a foundational task across several domains [45, 23, 47], including fluid dynamics [32, 15, 50], meteorology [39, 5], and industrial applications [13, 3]. The task aims to reconstruct the spatiotemporal information of a physical system from limited sparse observations gathered by pre-placed sensors. In real-world scenarios, the number of sensors is usually limited due to spatial constraints, power consumption, and environmental restrictions. As a result, allocating the limited sensors to the most informative positions is crucial for high reconstruction accuracy. Poorly placed sensors lead to spatial blind spots and information loss, resulting in insufficient observations, while well-placed sensors enable the recovery of fine-scale physical patterns. Therefore, advancing accurate physics sensing requires not only improving the capabilities of reconstruction models, but also developing principled strategies to determine optimal sensor placements.

Recent advances in deep learning have demonstrated remarkable capabilities in sparse-data reconstruction, owing to its ability to approximate highly nonlinear mappings [16, 44, 37] between scattered observations and dense physical fields. Current reconstruction methods fall into two categories: deterministic and generative. Classic methods such as VoronoiCNN [14] utilize Voronoi tessellations in CNN frameworks for reconstruction. Some recent work (e.g., [41]) attempts implicit neural representation-based sparse reconstruction but remains preliminary and with limited generalization. The landscape of sparse-data physical field reconstruction continues to evolve. Moreover, due to the inherently ill-posed nature [22] of the reconstruction task, where sparse observations provide insufficient

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

a Random Sensor Placement

Sensor position | Reconstruction | Error map

b Optimized Sensor Placement

Sensor position | Reconstruction | Error map

Figure 1: Performance comparison under the same reconstruction model but different sensor placements. (a) Random placement yields poor results due to inadequate spatial coverage. (b) Our optimized placement achieves accurate reconstruction by discovering informative regions, including side mirrors.

To recover dense fields, generative models [17, 42, 30, 29] have been increasingly explored. For example, DiffusionPDE [18] and S³GM [28] model physical field distributions via diffusion models, and integrate sensor observations during sampling through training-free guidance [7, 49]. Despite their reconstruction accuracy, these methods share a fundamental limitation: in their practical experiments, sensor placements are either randomly distributed or arbitrarily fixed rather than being systematically optimized. Consequently, the synergistic potential of co-optimizing reconstruction quality and placement strategy remains largely unexplored.

In practice, sensor placement determines the structure of the observation space and constrains what the model can perceive, while the feedback from the model’s reconstruction should guide where sensing is most beneficial. Notably, as illustrated in Figure 1, even under an identical reconstruction model, performance varies dramatically with different sensor placements. Random sensor placement fails to capture important spatial regions, resulting in degraded performance, whereas placements strategically optimized through reconstruction feedback yield significantly enhanced reconstruction accuracy by targeting informative regions, such as the side mirrors of a car. These results suggest the existence of a positive feedback loop between reconstruction quality and sensor placement optimization.

Building upon these insights, this paper presents PhySense, a synergistic two-stage framework that combines physical field reconstruction with sensor placement optimization. In the first stage, we train a base reconstruction model using flow matching [29], capable of generating dense fields from all feasible sensor placements. In the second stage, leveraging the reconstruction feedback, sensors are optimized using *projected gradient descent* to satisfy spatial geometries. Furthermore, we prove that the learning objectives of the two stages are consistent with classical variance-minimization principles, providing theoretical guarantees. We evaluate our method across three challenging benchmarks, including turbulent flow simulations, reanalysis of global sea temperature, and industrial simulations of aerodynamic surface pressure over a 3D car on an irregular geometry. In all cases, PhySense consistently achieves state-of-the-art performance and discovers informative sensor placements previously unconsidered. Overall, our contributions can be summarized as follows:

• We introduce PhySense, a synergistic two-stage framework for accurate physics sensing, which integrates a flow-based generative reconstruction model with a sensor placement optimization strategy through *projected gradient descent* to respect spatial constraints.

• We prove the learning objectives of the reconstruction model and sensor placement optimization are consistent with classical variance-minimization targets, providing theoretical guarantees.

• PhySense achieves consistent state-of-the-art reconstruction accuracy with 49% relative gain across three challenging benchmarks and discovers informative sensor placements.


## Preliminaries


## 2 Preliminaries

### 2.1 Deep Reconstruction Models

Recent deep learning methods have demonstrated promising performance in reconstructing dense physical fields from sparse measurements. Existing reconstruction models can be broadly categorized into deterministic and generative approaches. Representative deterministic methods are VoronoiCNN [14] and Senseiver [41]. VoronoiCNN constructs a Voronoi tessellation from scattered sensor observations, and learns to map the resulting structured representation to the target physical field. Senseiver encodes arbitrarily sized scattered inputs into a latent space using cross-attention and adopts an implicit neural representation as its decoder. However, since sparse reconstruction is an


![Figure 7](image_7.jpg)
<!-- Caption: Preliminaries, index 7 -->


inherently ill-posed problem, these deterministic methods lack the ability to characterize uncertainty in the reconstructed fields. Deep generative models have also been applied to the sparse reconstruction task. For example, S³GM [28] and DiffusionPDE [18] both tackle the reconstruction task by first learning a generative model of the underlying dynamics and then guiding the sampling process using sparse sensor observations. Crucially, these methods only utilize sparse observations during the sampling process—not during training—thus requiring thousands of sampling steps to align with sparse inputs, which compromises the efficiency of deep models. Despite promising reconstruction accuracy, all their experiments are conducted under random or fixed sensor placements, which leads to suboptimal reconstruction accuracy due to potentially non-informative sensor placements.

2.2 Sensor Placement Optimization

Optimal criterion Optimal sensor placement aims to maximize information extraction from physical systems by strategically positioning sensors. This problem is rooted in optimal experimental design (OED) theory [33, 36], which provides basic criteria for evaluating sensor placements. Three most widely-used criteria are: (1) A-optimality, which minimizes the average variance of all data estimates; (2) D-optimality, which maximizes the overall information content; and (3) E-optimality, which focuses on the worst-case estimation error. More details about these criteria can be found in [32]. In this work, we primarily focus on A-optimality. While these criteria originated in classical OED, they have been adapted to sensor placement optimization by incorporating spatial constraints [26].

Optimization algorithm Sensor placement optimization is a fundamentally hard problem due to its inherent high-dimensional nature. For finite option problems, an exhaustive search over all feasible placements may be tractable, but for more common high-dimensional problems, principled approximations become necessary. The theories of compressed sensing [9] show that signals admitting sparse representations in a known basis, such as the Fourier basis, can be accurately reconstructed with a small number of sensor observations. However, in real-world applications, the assumption of a suitable and known basis may not hold. Moreover, recent advances in machine learning have enabled data-driven discovery of low-dimensional structures such as low-rank subspaces [11, 46, 27]. This shift from fixed to data-driven representation spaces has spurred the development of placement optimization methods that leverage empirical priors or learned latent bases for efficient compression and reconstruction [32, 40]. For example, the SSPOR algorithm [32] first learns a data-driven basis, then applies QR factorization to select basis-specific sensor locations that maximize reconstruction performance. While recent work has begun to explore statistical-based sensor placement, deep learning-based approaches remain under explored. Besides, the further challenge lies in how to effectively co-optimize the reconstruction and the placement for accurate physics sensing.

2.3 Flow-based Models

Flow matching [29, 30, 31, 1] is a framework for solving transport mapping problem: given two distributions $$\mathbf{X}_0 \sim \pi_0$$ and $$\mathbf{X}_1 \sim \pi_1$$, the goal is to find a continuous mapping $$T$$, such that $$T(\mathbf{X}_0) \sim \pi_1$$. The method's efficiency and capacity have made it particularly valuable in vision tasks [12]. Unlike diffusion models that require computationally intensive ODE [42] or SDE [17, 43, 24] solving, flow matching introduces an ODE model that transfers $$\pi_0$$ to $$\pi_1$$ optimally via the straight line, theoretically the shortest path between two points, $$d\mathbf{X}_t = \mathbf{v}_\theta(\mathbf{X}_t, t) dt, t \in [0, 1]$$, where $$\mathbf{v}_\theta(\mathbf{X}_t, t)$$ is a learnable velocity field. To estimate v, flow matching solves a simple least squares regression problem that fits $$\mathbf{v}_\theta$$ to $$(\mathbf{X}_1 - \mathbf{X}_0)$$. The resulting dynamics generate samples via efficient push-forward along the theoretically shortest path, $$\mathbf{X}_{t+dt} = \mathbf{X}_t + \mathbf{v}_\theta(\mathbf{X}_t, t) dt$$, where $$\mathbf{X}_0 \sim \pi_0$$. In this paper, we consider to integrate this advanced technique for a more efficient reconstruction model.


## Method


3 Method

To enable accurate sensing of physical fields, we co-optimize reconstruction and placement through a synergistic two-stage framework. Our approach begins by training a flow-based model capable of reconstructing dense fields from scattered measurements across all feasible placements. Leveraging the reconstruction feedback, PhySense seamlessly integrates reconstruction and placement processes, forming a closed loop (Fig. 2(a)), and formulates placement as a spatially constrained optimization problem, efficiently solved via projected gradient descent. Moreover, the optimization objective theoretically provides a bidirectional control with respect to the variance-minimization target.

Figure 2: (a) PhySense works as a closed-loop physics sensing paradigm that iteratively co-optimizes sensor placements and reconstruction quality under the reconstruction feedback. (b) Reconstruction stage: A flow-based model with cross-attention mechanisms trained to process arbitrary sensor placements ("*←--*" indicates information flow for subsequent generation and placement optimization).

Problem setup. Let D ⊂ ℝ^d be a bounded domain, where there exist C physical fields of interest. We consider M = {p1, …, pm} sensors distributed across D, where each sensor measures the target quantities at its position. Our physics sensing task is to accurately reconstruct the entire physical field X1 based on sparse sensor observations X1(p*) under the optimal sensor placement p*.

3.1 Flow-based Reconstructor

Physics sensing, as an inverse problem to reconstruct dense physical fields from sparse observations, naturally requires generative approaches capable of handling the inherent ill-posedness. In our framework, we formulate the reconstruction task as an optimal transport flow problem between the target physical field distribution π1 and a standard Gaussian source distribution π0, with sensor placements and sparse observations serving as conditioning variables.

Notably, the reconstruction model is expected to provide instructive feedback for any current sensor placement in the subsequent placement optimization stage. Therefore, we implement random sensor placement sampling during the flow model training, which helps the model extract maximal useful information from any feasible sensor placement. Therefore, our training flow loss is formalized as
L = ∫_0^1 E_{X0, X1, p}[||(X1 − X0) − vθ(Xt, t, p, X1(p))||^2] dt, Xt = tX1 + (1 − t)X0. (1)
where vθ represents our learned velocity model, and p denotes a sensor placement randomly sampled from a uniform distribution on D. This model design must address two practical challenges: (1) handling diverse physical domains spanning regular grids to irregular meshes, and (2) effectively incorporating sparse observations for consistent reconstruction. For the first challenge, we leverage the state-of-the-art DiT [35, 10] for regular grids and Transolver [48], the most advanced approach for general geometry modeling, for irregular geometries, ensuring optimal parameterization of the velocity field across various domains. The second challenge is resolved through a cross-attention mechanism that dynamically associates scattered sensor measurements with their relevant spatial influence regions during training. This not only provides some physical insights into sensor–field interactions but also enhances generalization across varying number and placements of sensors.

Theoretically, we prove that our reconstructor can learn the conditional expectation of physical fields.

Theorem 3.1 (Flow-based reconstructor is an unbiased estimator of physical fields). Let the training flow loss Eq. (1) be minimized over a class of velocity fields v. The optimal learned velocity,
v*(x, t, p, X1(p)) = E[X1 − X0 | Xt, t, p, X1(p)]
equals the conditional mean of all feasible directions between the target data X1 and the initial noise X0. Define the reconstructed data as the integral along the optimal flow, X̄1 := X0 + ∫_0^1 v*(Xt, t, p, X1(p)) dt, then X̄1 is an unbiased estimator of the target physical fields given sensor placement p and corresponding observation X1(p), i.e., E[X̄1 | p, X1(p)] = E[X1 | p, X1(p)].

3.2 Sensor Placement Optimization

Based on the well-trained reconstructor, we formulate placement optimization as a constrained nonlinear optimization problem. We employ projected gradient descent to search for locally optimal sensor placements that reduce the flow loss while respecting geometric constraints. Although theoretical guarantees for global near A-optimality remain limited, this practical approach yields improved placements in experiments.


![Figure 16](image_16.jpg)
<!-- Caption: Method, index 16 -->



![Figure 17](image_17.jpg)
<!-- Caption: Method, index 17 -->


Projected gradient descent
Optimizing sensor placement presents unique challenges compared to standard model parameter optimization, since solutions must adhere to complex spatial constraints imposed by the physical domain $\mathcal{D}$. While gradient-based updates can suggest improved placements, naive application often violates geometric feasibility—particularly in irregular geometries. Therefore, we employ projected gradient descent (Fig. 3), which iteratively (1) computes accuracy-improving gradients and (2) projects the updated placements back to the feasible space via nearest-point mapping. This process can be formalized as
$$\mathbf{p}^{(k+1)}=\mathrm{Proj}_{\mathcal{D}}\!\left(\mathbf{p}^{(k)}-\eta\,\nabla_{\mathbf{p}}\mathcal{L}_{\text{flow}}(\mathbf{p}^{(k)})\right), \quad (2)$$
where $\mathbf{p}^{(k)}$ denotes the sensor placement at the $k$-th iteration, and $\mathrm{Proj}_{\mathcal{D}}(\cdot)$ denotes the projection operator onto the feasible domain $\mathcal{D}$. The strategy maintains strict spatial constraints while systematically exploring superior placements, effectively bridging unconstrained optimization with physical feasibility requirements.

Figure 3: Placement optimization stage: Based on a well-trained reconstructor, sensors are optimized by geometry-constrained projected gradient descent to minimize the flow loss with theoretical guarantees.

Optimization objective
The selection of optimization objective significantly impacts both computational cost and reconstruction accuracy. Although minimizing the reconstruction error $\|\bar{\mathbf{X}}_1-\mathbf{X}_1\|^2$ seems conceptually straightforward, its requirement for backpropagation among multiple flow-process steps results in prohibitive computational and memory overhead. In contrast, the flow loss defined in Eq. (1) maintains computational tractability through single flow-process-step gradient calculation and, due to the optimal transport path between $\mathbf{X}_0$ and $\mathbf{X}_1$, theoretically guarantees near A-optimal sensor placements, a dual advantage that makes it our preferred choice.

The following theoretical analysis elucidates what our placement optimization objective fundamentally captures and how it relates to the well-established A-optimality framework.

Assumption 3.2 (Conditional Gaussian Assumption - Simplification). For analytical tractability, we approximate the conditional distribution of the physical field given sensor placement $\mathbf{p}$ and observations $\mathbf{X}_1(\mathbf{p})$ as $\mathbf{X}_1\,|\,(\mathbf{p},\mathbf{X}_1(\mathbf{p})) \approx \mathcal{N}(\mu_{\mathbf{p}}, \Sigma_{\mathbf{p}})$, where $\mu_{\mathbf{p}}$ and $\Sigma_{\mathbf{p}}$ depend on $\mathbf{p}$. This is a modeling convenience; in practice, the true distribution may be non-Gaussian and multimodal. The independence between $\mathbf{X}_0$ and $\mathbf{X}_1$ is retained as a simplifying assumption.

Definition 3.3 (Classical A-optimal placement). We define the conditional covariance matrix of $\mathbf{X}_1$ given placement $\mathbf{p}$ and corresponding observation $\mathbf{X}_1(\mathbf{p})$ as
$\mathrm{Var}(\mathbf{p})=\mathbb{E}\!\left[(\mathbf{X}_1-\mathbb{E}\mathbf{X}_1)(\mathbf{X}_1-\mathbb{E}\mathbf{X}_1)^\top\,\big|\,\mathbf{p},\mathbf{X}_1(\mathbf{p})\right]=\Sigma_{\mathbf{p}}$.
A sensor placement $\mathbf{p}$ is called A–optimal if it minimizes the trace of $\Sigma_{\mathbf{p}}$,
$$\mathbf{p}^{A\text{-opt}}:=\arg\min_{\mathbf{p}\in\mathcal{D}^m}\mathcal{L}_A(\mathbf{p}),\quad\text{with }\mathcal{L}_A(\mathbf{p}):=\mathrm{tr}(\Sigma_{\mathbf{p}})\equiv\mathbb{E}\|\mathbf{X}_1-\mathbb{E}\mathbf{X}_1\|^2,$$
where $\mathcal{D}^m$ denotes all feasible placements for $m$ sensors. By definition, an A–optimal placement attains the minimal total conditional variance of the reconstructed field among feasible placements.

Definition 3.4 (Flow-loss-minimized sensor placement). Assume the velocity model $\mathbf{v}$ attains its optimum $\mathbf{v}^*$. Another optimal placement is defined if it minimizes the residual flow loss
$\mathbf{p}^*=\arg\min_{\mathbf{p}\in\mathcal{D}^m}\mathcal{L}_{\text{flow}}(\mathbf{p})$, where $\mathcal{L}_{\text{flow}}:=\mathbb{E}_{\mathbf{X}_t,t}\!\left[\left\|(\mathbf{X}_1-\mathbf{X}_0)-\mathbf{v}^*\right\|^2\right]$, which measures the conditional variance when the optimal transport flow path is used instead of the true data transition.

Theorem 3.5 (Approximate relationship between flow-loss and A-optimal placements). Under Assumption 3.2 and idealized model conditions, the objectives may be related approximately by inequalities of the form:
$$c_1 \mathcal{L}_{\text{flow}}^2(\mathbf{p}) - \epsilon \leq \mathcal{L}_A(\mathbf{p}) \leq c_2 \mathcal{L}_{\text{flow}}^2(\mathbf{p}) + \epsilon,$$
where $c_1, c_2$ are positive constants and $\epsilon$ captures deviations arising from model imperfections, estimation noise, and non-Gaussianity. This relation only holds under assumptions and should be treated as a heuristic rather than strict guarantee.

Proof sketch. The proof’s core lies in the non-trivial simplification of $\mathcal{L}_{\text{flow}}$ through careful analysis of the structure of the conditional covariance matrix. Owing to the favorable properties of the optimal transport path, the involved complex integrals can be calculated in a closed form, offering further insight into the structure of the flow matching. More details can be found in Appendix A.2. □


![Figure 20](image_20.jpg)
<!-- Caption: Method, index 20 -->


Remark 3.6. Consequently, the two objectives admit a two-sided polynomial bound of degree 2, which implies that minimizing $\mathcal{L}_{\text{flow}}$ inherently constrains $\mathcal{L}_A$ to lie within a bounded neighborhood of its optimum, and vice versa. This provides a theoretical guarantee for achieving near A-optimal sensor placement under the efficiently implementable flow-loss-minimization objective.

Overall design In summary, PhySense seamlessly integrates two synergistic stages for accurate physics sensing. (1) Reconstruction: a flow-based model learns to reconstruct dense physical fields from arbitrary scattered inputs through optimal transport flow dynamics. During training, both the number and placements of sensors are randomized to ensure generalizability. (2) Placement optimization: under fixed-number sensors, placements are optimized on the residual flow loss via projected gradient descent, constrained to feasible geometries (e.g., 3D surfaces). Theoretically, we prove that the reconstruction model learns the conditional expectation of target physical fields, and the flow-loss minimization objective polynomially controls the classical A-optimal criteria.


## Experiments


## 4 Experiments

We perform comprehensive experiments to evaluate PhySense, including turbulent flow simulation, reanalysis of global sea temperature with land constraints and aerodynamic pressure on a 3D car.

Benchmarks As presented in Table 1, our experiments encompass both regular grids and unstructured meshes in the 2D and 3D domains. The benchmarks include: (1) Turbulent Flow: numerical simulations of channel flow turbulence following the setup in Senseiver [41]; (2) Sea Temperature: 27-year daily sea surface temperature at 1° resolution with intricate land constraints from the GLORYS12 reanalysis dataset [21]; and (3) Car Aerodynamics: high-fidelity OpenFOAM simulations [20] of surface pressure over a complex car geometry from ShapeNet [6], where we reconstruct the irregular pressure fields from scattered sensor observations.

Table 1: Summary of benchmarks in PhySense. #Mesh records the size of discretized mesh points.

| Benchmarks       | Geometry          | #Dim | #Mesh |
| ---------------- | ----------------- | ---- | ----- |
| Turbulent Flow   | Regular Grid      | 2D   | 6144  |
| Sea Temperature  | Land Constraints  | 2D   | 64800 |
| Car Aerodynamics | Unstructured Mesh | 3D   | 95428 |

Baselines We evaluate PhySense via two approaches: (1) comparing the reconstruction model under randomly sampled sensor placements, and (2) comparing sensor placements with frozen PhySense’s reconstruction model. The reconstruction benchmarks include five representative baselines: classical statistical-based method SSPOR [32], deterministic models VoronoiCNN [14] and Senseiver [41], and generative approaches S³GM [28] and DiffusionPDE [18]. For placement evaluation, we compare our optimized placement against random sampling and SSPOR’s optimized placement [32].

Implementations All methods of each benchmark maintain identical training configurations including epochs, batch sizes, ADAM optimizer [25], and maximum learning rate, while preserving other details as specified in their original papers. For the reconstruction model, we employ DiT [35] for turbulent flow and sea temperature benchmarks, while using Transolver [48] as a Transformer backbone to fit irregular mesh for car aerodynamics benchmark. The placement optimization stage is performed iteratively with hyperparameters tuned per benchmark; epoch counts range between 3 and 10 depending on learning rate schedules and convergence criteria. Exact iteration counts are detailed in Appendix C.3 to ensure reproducibility. Both two stages employ relative L2 as the metrics for the reconstructed fields. Additional implementation details are provided in Appendix C.

### 4.1 Turbulent Flow

Setups The benchmark focuses on reconstructing turbulent flow within a channel using a 2-D slice discretized on a 128 × 48 regular grid. This dataset is widely used, with classical models such as Senseiver [41] and VoronoiCNN [14] serving as benchmarks. We replicate Senseiver’s setup; specifically, 25–300 sensors are randomly selected to train the reconstructor, with sensor positions uniformly sampled across the entire domain. A setting with 30 sensors represents low coverage, while 200 sensors corresponds to high coverage. The model is trained using all available simulation data and evaluated on its ability to reconstruct the corresponding fields under varying sensor placements. Besides, due to the computational cost of DiffusionPDE and S³GM—each reconstruction involving thousands of model evaluations—we restrict their evaluation to 500 randomly sampled cases.

Results As shown in Table 2, PhySense achieves state-of-the-art performance among all sensor counts, surpassing the second-best method by a substantial margin. Despite starting from our strong

Table 2: Performance comparison on the turbulent flow and sea surface temperature benchmarks. Relative L2 error is reported. PhySense-opt denotes the performance under optimized sensor placement, while all other deep models, including PhySense, use randomly placed sensors. Besides, SSPOR is a classical method that performs both placement optimization and reconstruction. # indicates the sensor number. Promotion refers to the relative error reduction of PhySense-opt w.r.t. the best baseline.

| Models             | Placement Strategy | Turbulent Flow #200 | Turbulent Flow #100 | Turbulent Flow #50 | Turbulent Flow #30 | Sea Temperature #100 | Sea Temperature #50 | Sea Temperature #25 | Sea Temperature #15 |
| ------------------ | ------------------ | -------------------- | -------------------- | ------------------- | ------------------- | --------------------- | -------------------- | -------------------- | -------------------- |
| SSPOR [32]         | SSPOR              | 0.3018               | 0.4429               | 0.5921              | 0.6637              | 0.0756                | 0.0752               | 0.0789               | 0.0719               |
| VoronoiCNN [14]    | Random             | 0.3588               | 0.4870               | 0.6919              | 0.7992              | 0.1496                | 0.1537               | 0.2433               | 0.2817               |
| Sensceiver [41]    |                    | 0.1842               | 0.2316               | 0.3740              | 0.5746              | 0.0715                | 0.0732               | 0.0769               | 0.0784               |
| S³GM [28]          |                    | 0.1856               | 0.2016               | 0.2681              | 0.6421              | 0.1115                | 0.1137               | 0.1581               | 0.1795               |
| DiffusionPDE [18]  |                    | 0.1993               | 0.3312               | 0.6604              | 0.9163              | 0.0664                | 0.0775               | 0.0859               | 0.1139               |
| PhySense           | Random             | 0.1233               | 0.1527               | 0.2586              | 0.5176              | 0.0439                | 0.0452               | 0.0477               | 0.0520               |
| PhySense-opt       | Optimized          | 0.1106               | 0.1257               | 0.1558              | 0.2157              | 0.0426                | 0.0430               | 0.0437               | 0.0439               |
| Relative Promotion |                    | 39.96%               | 38.56%               | 41.89%              | 62.46%              | 35.84%                | 41.26%               | 43.17%               | 38.94%               |

Figure 4: (a) Reconstruction loss versus the number of sensors. Random sampling, classical SSPOR, and our optimized method are compared using the same PhySense reconstruction base model. (b) Visualization of reconstruction results, error maps, and sensor placements (denoted by ⋆). Our optimized placement clearly outperforms the other two strategies on the turbulent flow benchmark.

Building upon the reconstruction model, the reconstruction accuracy is further significantly improved through sensor placement optimization. The improvement is particularly notable under low coverage with 30 sensors relative to the high coverage setup with 200 sensors. Remarkably, even with fewer sensors, the optimized 30-sensor placement exceeds the performance of the unoptimized 50-sensor counterpart. As shown in Fig. 4(a), our learned placement strategy consistently yields the lowest reconstruction error compared to the other two strategies, especially in low coverage settings, demonstrating its superior effectiveness. Fig. 4(b) visualizes selected sensor placements and error distributions. While these results suggest improved coverage compared to random placement, further quantitative analysis would be required to conclusively determine the relation to information gain and sensor redundancy.

## 4.2 Sea Temperature

Setups. This benchmark is derived from the GLORYS12 reanalysis dataset [21], focusing on the sea surface temperature variable. The data are downsampled to a 1° × 1° resolution, resulting in global ocean fields of size 360 × 180. During training, between 10 and 100 sensors are randomly selected, with all sensors uniformly sampled from valid ocean regions, excluding land areas. The reconstruction task aims to recover dense temperature fields from these sparse observations. Our training set comprises 9,843 daily samples from 1993 to 2019, with data from 2020–2021 reserved for testing. This setup allows us to evaluate two critical aspects of model capability: (1) adaptability to varying sensor placements, and (2) generalization to temporal distribution shifts. Moreover, the presence of complex land–sea boundaries introduces significant challenges for both reconstruction and sensor placement optimization. We address this by providing a land mask as an additional input.

Results. PhySense-opt consistently achieves state-of-the-art performance across all sensor numbers (average improvement 40.0%), as shown in Table 2. Besides, SSPOR [32] performs reasonably well on this benchmark, due to the relatively stable temporal variations of sea temperature over the 30-year period.


![Figure 26](image_26.jpg)
<!-- Caption: Experiments, index 26 -->


Figure 5: Different placement strategies comparison and case study on the sea temperature benchmark.

This also explains its poor performance on the turbulent flow, highlighting the limitations of traditional methods in capturing complex, non-stationary physical fields.

The optimized sensor placement improves performance by an additional 12.0% over the unoptimized counterpart in the low coverage setting (25 and 15 sensors). As shown in Fig. 5(a), the placement selected by SSPOR performs comparably to random selection, underscoring the importance of jointly co-optimizing reconstruction modeling and sensor placement optimization. Remarkably, just 15 optimally placed sensors—covering only 0.023% of the spatial domain—are sufficient to achieve reconstruction accuracy comparable with the best settings on this benchmark, which highlights the feasibility of global ocean temperature field reconstruction with limited sensing resources.

Moreover, our placement strategy successfully avoids land regions and performs effective optimization under intricate land-sea boundaries (Fig. 5(b)). Such counterintuitive optimized placement is difficult to obtain through human-designed heuristics. The error map within the red box clearly indicates that the optimized placement captures more information, which in turn improves the reconstruction accuracy.

4.3 Car Aerodynamics

Setups This benchmark is the first to address physics sensing tasks on 3D irregular meshes. We select a fine 3D car model with 95,428 mesh points and simulate its pressure field using OpenFOAM (see Appendix C.1 for simulation details). The simulations span driving velocities from 20 to 40 m/s and yaw angles from -10 to 10 degrees. We simulate 100 cases in total, splitting them into 75 for training and 25 for testing. The task involves reconstructing the surface pressure field from 10 to 200 randomly sampled sensors placed exclusively on the vehicle surface, since placing sensors in the surrounding but off-surface regions presents practical challenges. This setup imposes particularly challenging conditions for placement optimization due to the complexity of the 3D surface and its geometric constraints.

Table 3: Main results on the car aerodynamics benchmark. Due to resource constraints, only the best deterministic baseline and generative baseline are selected. To support irregular meshes, the backbone of DiffusionPDE [18] is replaced with Transolver [48].

Models | Car Aerodynamics #200 | Car Aerodynamics #100 | Car Aerodynamics #50 | Car Aerodynamics #30 | Car Aerodynamics #15
Senseiver [41] | 0.1009 | 0.1012 | 0.1018 | 0.1022 | 0.1053
DiffusionPDE [18] | 0.0967 | 0.0966 | 0.0987 | 0.1055 | 0.2095
PhySense | 0.0375 | 0.0382 | 0.0395 | 0.0416 | 0.0465
PhySense-opt | 0.0369 | 0.0370 | 0.0370 | 0.0372 | 0.0386
Promotion | 61.84% | 61.70% | 62.51% | 63.60% | 63.34%

Results While Table 3 highlights only select sensor counts, subsequent analyses (Appendix D) reveal that performance degrades notably at very low sensor numbers (below 10), and variance across runs increases. Inclusion of full sensor count ranges is necessary to fully assess model behavior. Moreover, our placement strategy proves effective, especially in low-coverage scenarios. With only 15 optimized sensors, we achieve a 17.0% improvement (from 0.0465 to 0.0386), and the reconstruction accuracy matches that of using 100 randomly placed sensors. Moreover, Fig. 1 illustrates that, since the wind approaches from the front of the driving vehicle, regions such as the front end and side mirrors experience significant pressure gradients, likely caused by flow separation or viscous effects. Our optimized sensor placement successfully captures these high-variation regions, validating the effectiveness of the projected gradient descent optimization. Furthermore, our optimized sensor placement can be deployed in real-world settings, such as wind tunnel experiments, to collect actual pressure data. This enables correction of discrepancies in simulation results and even of underlying physical systems, reducing the requirements for repeating costly real experiments and narrowing the sim-to-real gap.


![Figure 29](image_29.jpg)
<!-- Caption: Experiments, index 29 -->


Figure 6: (a) The cross-attention map visualizations from the last layer of PhySense, where sensor positions are indicated by red arrows. (b) Initial placement sensitivity analysis shows that varying initial sensor placements converge to different but equally effective placements, achieving similar reconstruction accuracy. (c) Efficiency comparison on turbulent flow: these diffusion-based methods require 1,000 generation steps (100× slower than PhySense), but achieve even worse accuracy.

## 4.4 Model Analysis

Ablations also include systematic sensitivity analyses to key training hyperparameters such as learning rate, optimizer type, and projected gradient descent iteration counts. These studies reveal moderate sensitivity in very low sensor counts, but overall stable convergence behavior.

Table 4: Ablation on sparse information incorporation via an interpolated field (Rep. Inter.).

| Ablations     | Turbulent Flow<br/>#200 | Turbulent Flow<br/>#100 | Turbulent Flow<br/>#50 | Turbulent Flow<br/>#30 |
| ------------- | ----------------------- | ----------------------- | ---------------------- | ---------------------- |
| Rep. Inter.   | 0.1571                  | 0.2006                  | 0.3584                 | 0.7509                 |
| Ours          | **0.1233**              | **0.1527**              | **0.2586**             | **0.5176**             |

Cross attention visualization To demonstrate how PhySense dynamically associates sensor observations with their relevant spatial influence regions, we visualize the cross-attention maps from the last cross attention block. As shown in Fig. 6(a), PhySense captures physically meaningful regions of influence for each sensor on the complex mesh, indicating the underlying physical structure for its high-fidelity reconstruction. Moreover, a highlighted pair in the second row shows two spatially adjacent sensors exhibit diametrically opposed attention patterns due to local geometric variations.

Efficiency analysis The efficiency comparison on turbulent flow benchmark in Fig. 6(c) shows that our method achieves significantly lower reconstruction errors across all step counts. Notably, while DiffusionPDE and S³GM require over 50 steps to achieve an error below 0.5, our method consistently maintains a relative L2 error below 0.2 even with as few as 2 steps. This efficiency advantage stems from fundamental differences in sparse information incorporation strategies. DiffusionPDE and S³GM indirectly modify the data generation gradient based on sparse observations which require numerous iterative refinements to achieve proper guidance, while our approach directly conditions the model adaptively on sparse inputs during training. This architectural difference enables immediate and effective utilization of observational information from the very first inference step. As a result, we achieve better performance with only 1% model evaluations required by these baseline methods.

Initial placement sensitivity During the sensor placement stage, different placement initializations lead to varying final sensor distributions. Meanwhile, as shown in Fig. 6(b), after placement optimization, the reconstruction performance remains consistently high-quality and comparable across these variants, indicating that our placement optimization process is not sensitive to different initializations.

Progressive sensor placement optimization A key advantage of our flow-based reconstructor is its inherent support for progressive sensor placement optimization. Unlike non-generative models, which operate on static inputs (e.g., pure noise or zeros), our model processes a noise-data mixture X_t throughout the flow. This design inherently embeds a spectrum of reconstruction difficulties, corresponding to different noise levels, thereby amortizing the optimization process across the entire flow. Consequently, it provides more informative, tractable feedback for optimizing sensor placement.

Table 5: Ablation on progressive sensor placement optimization via a pure noise input.

| Ablations            | Turbulent Flow<br/>#200 | Turbulent Flow<br/>#100 | Turbulent Flow<br/>#50 | Turbulent Flow<br/>#30 |
| -------------------- | ----------------------- | ----------------------- | ---------------------- | ---------------------- |
| noise-input          | 0.1118                  | 0.1293                  | 0.1630                 | 0.2286                 |
| mix-input (original) | **0.1106**              | **0.1257**              | **0.1558**             | **0.2157**             |
| Promotion            | 1.07%                   | 2.78%                   | 4.42%                  | 5.65%                  |


![Figure 32](image_32.jpg)
<!-- Caption: Experiments, index 32 -->



![Figure 33](image_33.jpg)
<!-- Caption: Experiments, index 33 -->



![Figure 34](image_34.jpg)
<!-- Caption: Experiments, index 34 -->


Table 6: Performance comparison with four additional placement baselines on the turbulent flow benchmark, measured by relative L2 error. Results obtained with limited sensors (e.g., 30) are highlighted in blue, and the top-performing methods with comparable results are shown in bold.

| Placement Strategies            | Senseiver #200 | Senseiver #100 | Senseiver #50 | Senseiver #30 | PhySense #200 | PhySense #100 | PhySense #50 | PhySense #30 |
| ------------------------------- | -------------- | -------------- | ------------- | ------------- | ------------- | ------------- | ------------ | ------------ |
| Random Sample                   | 0.1842         | 0.2316         | 0.3740        | 0.5746        | 0.1233        | 0.1527        | 0.2586       | 0.5176       |
| Grid Sample                     | 0.1615         | 0.2273         | 0.3845        | 0.5869        | 0.1235        | 0.1521        | 0.2436       | 0.4930       |
| Min-Max Sample                  | 0.1549         | 0.2091         | 0.3795        | 0.5747        | 0.1196        | 0.1533        | 0.3157       | 0.5022       |
| Our PGD on reconstruction model | **0.1388**     | **0.1750**     | **0.3025**    | **0.4570**    | **0.1106**    | **0.1257**    | **0.1558**   | **0.2157**   |
| Optimized Sensors from PhySense | **0.1533**     | **0.1833**     | **0.3035**    | **0.4601**    | **0.1106**    | **0.1257**    | **0.1558**   | **0.2157**   |

Sensor placement. An ablation study shown in Table 5 confirms that replacing the noise–data mix input with pure noise input in the feedback loop leads to notable performance degradation, which is more pronounced with fewer sensors, underscoring the unique benefit of this flow-process feedback.

Placement generalization. We compare our placement strategy with four more placement baselines across Senseiver and PhySense under different sensor numbers on the turbulent flow dataset.

(i) Grid: We first place sensors on a uniform grid using the largest square number within the budget, and then randomly allocate the remainder. This strategy guarantees spatial coverage but remains agnostic to the physical dynamics. Practically, its performance is similar to random placement, aligning with classical sampling theory [4] that questions the efficacy of grid sampling for complex systems.

(ii) Min-Max: A heuristic method that promotes spread-out sensor placements by iteratively placing the next sensor at the point that maximizes the minimum distance to existing sensors. However, this approach does not yield significant improvements over random placement in our experiments.

(iii) Our Projected Gradient Descent: An upper-bound method using any reconstruction model (Senseiver or PhySense) and performing our projected gradient descent to optimize sensor placements.

(iv) Optimized Sensors from PhySense: To evaluate the generality of the learned placements by PhySense, we feed the placements into the Senseiver. The performance closely matches that of Our PGD and significantly outperforms all other baselines, particularly when the number of sensors is limited (e.g., 30), a challenging scenario to place sensors. This suggests that our learned placements effectively capture physically informative structures that generalize beyond specific reconstructors.

Optimized sensor distribution visualization. To explore the distribution of optimized sensors, we visualize their relationship with information distribution (measured by the model’s spatial gradient magnitude) and variance distribution (point-wise variance of the physical field) on the turbulent flow benchmark. Interestingly, most of the optimized sensors lie within high-variance regions. More importantly, each sensor is located near a high-information point, and their spatial distributions exhibit similar patterns. This suggests that the optimizer is not merely clustering in high-variance areas but is strategically identifying spatially separated, high-sensitivity locations that capture the field’s global dynamics. This aligns with the sensing principle that maximal information gain requires sampling diverse, non-redundant points rather than densely covering localized variance hotspots.

Figure 7: Visualization of information and variance distribution on the turbulent flow benchmark.


## Conclusion


5 Conclusion

To co-optimize physics-field reconstruction and sensor placement for accurate physics sensing, this paper presents PhySense as a synergistic two-stage framework. The first stage develops a base model through flow matching that reconstructs dense physical fields from all feasible sensor placements. The second stage strategically optimizes sensor positions through projected gradient descent to respect geometry constraints. Furthermore, our theoretical analysis reveals a deep connection between the framework’s objectives and the classical A-optimal principle, i.e., variance minimization. Extensive experiments indicate strong performance across benchmarks, notably on a complex 3D irregular dataset, and identify sensor placements of interest. Further work is necessary to generalize findings and address limitations discussed.


![Figure 40](image_40.jpg)
<!-- Caption: Conclusion, index 40 -->



![Figure 41](image_41.jpg)
<!-- Caption: Conclusion, index 41 -->



## Appendix


Acknowledgments and Disclosure of Funding

This work was supported by the National Natural Science Foundation of China (62021002 and U2342217), the BNRist Innovation Fund (BNR2024RC01010), and the National Engineering Research Center for Big Data Software. We acknowledge that large-scale physics sensing models require careful consideration of computational cost, environmental impact, and ethical deployment contexts, which are not fully addressed in this study.


## References


References

[1] Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In ICLR, 2023.
[2] Gal Berkooz, Philip Holmes, and John L. Lumley. The proper orthogonal decomposition in the analysis of turbulent flows. Annual Review of Fluid Mechanics, 1993.
[3] Steven L. Brunton, J. Nathan Kutz, Krithika Manohar, Aleksandr Y. Aravkin, Kristi Morgansen, Jennifer Klemisch, Nicholas Goebel, James Buttrick, Jeffrey Poskin, Adriana W. Blom-Schieber, et al. Data-driven aerospace engineering: reframing the industry with machine learning. AIAA Journal, 2021.
[4] Emmanuel J. Candes and Terence Tao. Near-optimal signal recovery from random projections: Universal encoding strategies? IEEE Transactions on Information Theory, 2006.
[5] Alberto Carrassi, Marc Bocquet, Laurent Bertino, and Geir Evensen. Data assimilation in the geosciences: An overview of methods, issues, and perspectives. Wiley Interdisciplinary Reviews: Climate Change, 2018.
[6] Angel X. Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. ShapeNet: An information-rich 3D model repository. arXiv preprint arXiv:1512.03012, 2015.
[7] Hyungjin Chung, Jeongsol Kim, Michael T. McCann, Marc L. Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In ICLR, 2023.
[8] Brian M. de Silva, Krithika Manohar, Emily Clark, Bingni W. Brunton, J. Nathan Kutz, and Steven L. Brunton. PySensors: A Python package for sparse sensor placement. Journal of Open Source Software, 2021.
[9] David L. Donoho. Compressed sensing. IEEE Transactions on Information Theory, 2006.
[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.
[11] Zlatko Drmac and Serkan Gugercin. A new selection operator for the discrete empirical interpolation method—improved a priori error bound and extensions. SIAM Journal on Scientific Computing, 2016.
[12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.
[13] Luigi Fortuna, Salvatore Graziani, Alessandro Rizzo, Maria G. Xibilia, et al. Soft sensors for monitoring and control of industrial processes. Springer, 2007.
[14] Kai Fukami, Romit Maulik, Nesar Ramachandra, Koji Fukagata, and Kunihiko Taira. Global field reconstruction from sparse sensors with Voronoi tessellation-assisted deep learning. Nature Machine Intelligence, 2021.
[15] Yaqing Gu and Dean S. Oliver. An iterative ensemble Kalman filter for multiphase fluid flow data assimilation. SPE Journal, 2007.
[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.
[17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.
[18] Jiahe Huang, Guandao Yang, Zichen Wang, and Jeong Joon Park. DiffusionPDE: Generative PDE-solving under partial observation. In NeurIPS, 2024.
[19] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In ICML, 2021.

[20] Hrvoje Jasak. OpenFOAM: Open Source CFD in Research and Industry. *International Journal of Naval Architecture and Ocean Engineering*, 2009.

[21] Jean-Michel Lellouche, Eric Greiner, Romain Bourdallé-Badie, Gilles Garric, Angélique Melet, Marie Drévillon, Clément Bricaud, Mathieu Hamon, Olivier Le Galloudec, Charly Régnier, et al. The Copernicus Global 1/12° Oceanic and Sea Ice GLORYS12 Reanalysis. *Frontiers in Earth Science*, 2021.

[22] Sergey I. Kabanikhin. Inverse and Ill-Posed Problems: Theory and Applications. de Gruyter, 2011.

[23] George Em Karniadakis, Ioannis G. Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. *Nature Reviews Physics*, 2021.

[24] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the Design Space of Diffusion-Based Generative Models. In *NeurIPS*, 2022.

[25] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. In *ICLR*, 2015.

[26] Andreas Krause, Ajit Singh, and Carlos Guestrin. Near-Optimal Sensor Placements in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies. *JMLR*, 2008.

[27] J. Nathan Kutz, Steven L. Brunton, Bingni W. Brunton, and Joshua L. Proctor. *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems*. SIAM, 2016.

[28] Zeyu Li, Wang Han, Yue Zhang, Qingfei Fu, Jingxuan Li, Lizi Qin, Ruoyu Dong, Hao Sun, Yue Deng, and Lijun Yang. Learning spatiotemporal dynamics with a pretrained generative model. *Nature Machine Intelligence*, 2024.

[29] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow Matching for Generative Modeling. In *ICLR*, 2023.

[30] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. In *ICLR*, 2023.

[31] Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, and Saining Xie. SiT: Exploring Flow and Diffusion-Based Generative Models with Scalable Interpolant Transformers. In *ECCV*. Springer, 2024.

[32] Krithika Manohar, Bingni W. Brunton, J. Nathan Kutz, and Steven L. Brunton. Data-Driven Sparse Sensor Placement for Reconstruction: Demonstrating the Benefits of Exploiting Known Patterns. *IEEE Control Systems Magazine*, 2018.

[33] Panos Y. Papalambros and Douglass J. Wilde. *Principles of Optimal Design: Modeling and Computation*. Cambridge University Press, 2000.

[34] Adam Paszke, S. Gross, Francisco Massa, A. Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Z. Lin, N. Gimelshein, L. Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In *NeurIPS*, 2019.

[35] William Peebles and Saining Xie. Scalable Diffusion Models with Transformers. In *ICCV*, 2023.

[36] Friedrich Pukelsheim. *Optimal Design of Experiments*. SIAM, 2006.

[37] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving Language Understanding by Generative Pre-Training. 2018.

[38] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. In *MICCAI*, 2015.

[39] Bertrand Rouet-Leduc, Claudia Hulbert, and Paul A. Johnson. Continuous chatter of the Cascadia subduction zone revealed by machine learning. *Nature Geoscience*, 2019.

[40] Samuel H. Rudy, Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz. Data-driven discovery of partial differential equations. *Science Advances*, 2017.

[41] Javier E. Santos, Zachary R. Fox, Arvind Mohan, Daniel O’Malley, Hari Viswanathan, and Nicholas Lubbers. Development of the Senseiver for efficient field reconstruction from sparse observations. *Nature Machine Intelligence*, 2023.

[42] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models. In *ICLR*, 2021.
[43] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-Based Generative Modeling Through Stochastic Differential Equations. In *ICLR*, 2021.
[44] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. In *NeurIPS*, 2017.
[45] Hanchen Wang, Tianfan Fu, Yuanqi Du, Wenhao Gao, Kexin Huang, Ziming Liu, Payal Chandak, Shengchao Liu, Peter van Katwyk, Andreea Deac, et al. Scientific discovery in the age of artificial intelligence. *Nature*, 2023.
[46] Karen Willcox. Unsteady flow sensing and estimation via the gappy proper orthogonal decomposition. *Computers & Fluids*, 2006.
[47] Haixu Wu, Huakun Luo, Yuezhou Ma, Jianmin Wang, and Mingsheng Long. ROPINN: Region-optimized Physics-Informed Neural Networks. *NeurIPS*, 2024.
[48] Haixu Wu, Huakun Luo, Haowen Wang, Jianmin Wang, and Mingsheng Long. Transolver: A Fast Transformer Solver for PDEs on General Geometries. In *ICML*, 2024.
[49] Haotian Ye, Haowei Lin, Jiaqi Han, Minkai Xu, Sheng Liu, Yitao Liang, Jianzhu Ma, James Y. Zou, and Stefano Ermon. TFG: Unified Training-Free Guidance for Diffusion Models. In *NeurIPS*, 2024.
[50] Hang Zhou, Yuezhou Ma, Haixu Wu, Haowen Wang, and Mingsheng Long. UniSolver: PDE-conditional Transformers Towards Universal Neural PDE Solvers. In *ICML*, 2025.


## Appendix


A Proofs of Theorems in the Method Section

A.1 Proof of Theorem 3.1

Proof. By the pointwise optimality of the squared loss in the training flow loss (1),
$$\mathbf{v}^*(\mathbf{x}, t, \mathbf{p}, \mathbf{X}_1(\mathbf{p})) = \mathbb{E}\big[\mathbf{X}_1 - \mathbf{X}_0 \,\big|\, \mathbf{X}_t = \mathbf{x},\, t,\, \mathbf{p},\, \mathbf{X}_1(\mathbf{p})\big].$$

Taking the conditional expectation of \(\bar{\mathbf{X}}_1\) gives
$$\mathbb{E}\big[\bar{\mathbf{X}}_1 \,\big|\, \mathbf{p}, \mathbf{X}_1(\mathbf{p})\big]
= \mathbb{E}\left[\mathbf{X}_0 + \int_0^1 \mathbf{v}^*(\mathbf{X}_t, t, \mathbf{p}, \mathbf{X}_1(\mathbf{p}))\, dt \,\Big|\, \mathbf{p}, \mathbf{X}_1(\mathbf{p})\right]$$
$$= \underbrace{\mathbb{E}\big[\mathbf{X}_0 \,\big|\, \mathbf{p}, \mathbf{X}_1(\mathbf{p})\big]}_{(A)} + \underbrace{\mathbb{E}\left[\int_0^1 \mathbf{v}^*(\mathbf{X}_t, t, \mathbf{p}, \mathbf{X}_1(\mathbf{p}))\, dt \,\Big|\, \mathbf{p}, \mathbf{X}_1(\mathbf{p})\right]}_{(B)}.$$

Step 1: Move the expectation inside the integral (Fubini’s theorem):
$$(B) = \int_0^1 \mathbb{E}\big[\mathbf{v}^*(\mathbf{X}_t, t, \mathbf{p}, \mathbf{X}_1(\mathbf{p})) \,\big|\, \mathbf{p}, \mathbf{X}_1(\mathbf{p})\big]\, dt.$$

Step 2: Substitute the definition of \(\mathbf{v}^*\) and apply the law of total expectation:
\[
\mathbb{E}\big[\mathbf{v}^*(\mathbf{X}_t, t, \mathbf{p}, \mathbf{X}_1(\mathbf{p})) \,\big|\, \mathbf{p}, \mathbf{X}_1(\mathbf{p})\big]
= \mathbb{E}\Big[\mathbb{E}\big[\mathbf{X}_1 - \mathbf{X}_0 \,\big|\, \mathbf{X}_t, t, \mathbf{p}, \mathbf{X}_1(\mathbf{p})\big] \,\Big|\, \mathbf{p}, \mathbf{X}_1(\mathbf{p})\Big]
= \mathbb{E}\big[\mathbf{X}_1 - \mathbf{X}_0 \,\big|\, \mathbf{p}, \mathbf{X}_1(\mathbf{p})\big].
\]
The right-hand side no longer depends on \(t\), hence
\[
(B) = \int_0^1 \mathbb{E}\big[\mathbf{X}_1 - \mathbf{X}_0 \,\big|\, \mathbf{p}, \mathbf{X}_1(\mathbf{p})\big]\, dt
= \mathbb{E}\big[\mathbf{X}_1 - \mathbf{X}_0 \,\big|\, \mathbf{p}, \mathbf{X}_1(\mathbf{p})\big].
\]

Step 3: Combine \((A)\) and \((B)\):
\[
(A) + (B) = \mathbb{E}\big[\mathbf{X}_0 \,\big|\, \mathbf{p}, \mathbf{X}_1(\mathbf{p})\big] + \mathbb{E}\big[\mathbf{X}_1 - \mathbf{X}_0 \,\big|\, \mathbf{p}, \mathbf{X}_1(\mathbf{p})\big]
= \mathbb{E}\big[\mathbf{X}_1 \,\big|\, \mathbf{p}, \mathbf{X}_1(\mathbf{p})\big].
\]
Therefore, the flow-based reconstructor is an unbiased estimator of physical fields. □

A.2 Proof of Theorem 3.5

Part I. First, we will prove \(\mathcal{L}_A(\mathbf{p}) = \mathbb{E}\big[\|\mathbf{X}_1 - \mathbb{E}[\mathbf{X}_1]\|^2\big] = \sum_{i=1}^d \lambda_i(\mathbf{p})\), where \(\{\lambda_i(\mathbf{p})\}_{i=1}^d\) are the eigenvalues of \(\Sigma_{\mathbf{p}}\).

Because \(\mathbf{X}_1 \mid (\mathbf{p}, \mathbf{X}_1(\mathbf{p})) \sim \mathcal{N}(0, \Sigma_{\mathbf{p}})\), we have
\[
\mathcal{L}_A(\mathbf{p}) = \mathbb{E}\big[\|\mathbf{X}_1 - \mathbb{E}[\mathbf{X}_1]\|^2\big]
= \operatorname{tr}\big(\operatorname{Var}(\mathbf{X}_1)\big) = \operatorname{tr}(\Sigma_{\mathbf{p}}).
\]
Let the orthogonal decomposition of the placement-dependent covariance matrix be
\[
\Sigma_{\mathbf{p}} = U \Lambda U^\top, \quad \Lambda = \operatorname{diag}\big(\lambda_1(\mathbf{p}), \ldots, \lambda_d(\mathbf{p})\big),
\]
where \(U\) is an orthogonal matrix, i.e., \(U U^\top = I\). All eigenvalues \(\lambda_i(\mathbf{p})\) are non-negative because the covariance matrix \(\Sigma_{\mathbf{p}}\) is positive semidefinite. Since the trace is invariant under orthogonal similarity transformations, we obtain
\[
\mathcal{L}_A(\mathbf{p}) = \operatorname{tr}(\Sigma_{\mathbf{p}}) = \operatorname{tr}(\Lambda) = \sum_{i=1}^d \lambda_i(\mathbf{p}). \tag{A.0}
\]

Part 2. Second, we will show,
\[
\mathcal{L}_{\text{flow}}(\mathbf{p})=\int_{0}^{1}\mathbb{E}\Big[\big\|(\mathbf{X}_1-\mathbf{X}_0)-\mathbf{v}^*(\mathbf{X}_t,t,\mathbf{p},\mathbf{X}_1(\mathbf{p}))\big\|^2\Big]dt
\]
\[
\stackrel{\text{def}}{=}\int_{0}^{1}\operatorname{tr}\big(\operatorname{Var}(\mathbf{X}_1-\mathbf{X}_0\mid\mathbf{X}_t,t,\mathbf{p},\mathbf{X}_1(\mathbf{p}))\big)\,dt
\]
\[
=\sum_{i=1}^{d}\frac{\pi}{2}\sqrt{\lambda_i(\mathbf{p})}.
\]

Step 1: Calculate the conditional variance \(\operatorname{Var}(\mathbf{X}_1-\mathbf{X}_0\mid\mathbf{X}_t,t,\mathbf{p},\mathbf{X}_1(\mathbf{p}))\)

We denote that,
\[
\mathbf{Z}=\begin{pmatrix}\mathbf{X}_0\\ \mathbf{X}_1\end{pmatrix},\quad
\Sigma_{\mathbf{Z}}=\operatorname{diag}(I,\Sigma_{\mathbf{p}}),\quad
\mathbf{Y}=\mathbf{X}_1-\mathbf{X}_0,\quad
\mathbf{X}_t=(1-t)\mathbf{X}_0+t\mathbf{X}_1\quad(0\le t\le 1).
\]
Since \(\mathbf{X}_0\) and \(\mathbf{X}_1\) are independent, \(\mathbf{Z}\) conditioned on \((\mathbf{p},\mathbf{X}_1(\mathbf{p}))\) remains Gaussian,
\[
\mathbf{Z}\mid(\mathbf{p},\mathbf{X}_1(\mathbf{p}))\sim\mathcal{N}(0,\Sigma_{\mathbf{Z}}).
\]
Writing \(\mathbf{Y}\) and \(\mathbf{X}_t\) as linear maps of \(\mathbf{Z}\),
\[
\mathbf{Y}=V\mathbf{Z},\quad V=(-I,I),\qquad
\mathbf{X}_t=C\mathbf{Z},\quad C=((1-t)I,tI),
\]
thus the pair \((\mathbf{Y},\mathbf{X}_t)\) is jointly Gaussian. The standard conditional–covariance formula yields
\[
\operatorname{Var}(\mathbf{Y}\mid\mathbf{X}_t,t,\mathbf{p},\mathbf{X}_1(\mathbf{p}))=V\Sigma_{\mathbf{Z}}V^{\!T}-V\Sigma_{\mathbf{Z}}C^{\!T}(C\Sigma_{\mathbf{Z}}C^{\!T})^{-1}C\Sigma_{\mathbf{Z}}V^{\!T}\qquad\text{(A.1)}
\]
with
\[
V\Sigma_{\mathbf{Z}}V^{\!T}=I+\Sigma_{\mathbf{p}},\qquad
C\Sigma_{\mathbf{Z}}C^{\!T}=(1-t)^2I+t^2\Sigma_{\mathbf{p}},\qquad
V\Sigma_{\mathbf{Z}}C^{\!T}=-(1-t)I+t\Sigma_{\mathbf{p}}.
\]
Substituting into (A.1) gives
\[
\operatorname{Var}(\mathbf{Y}\mid\mathbf{X}_t,t,\mathbf{p},\mathbf{X}_1(\mathbf{p}))=I+\Sigma_{\mathbf{p}}-(t\Sigma_{\mathbf{p}}-(1-t)I)\big((1-t)^2I+t^2\Sigma_{\mathbf{p}}\big)^{-1}(t\Sigma_{\mathbf{p}}-(1-t)I)\qquad\text{(A.2)}
\]

Step 2: Diagonalisation and scalar reduction

Recall from (A.2) the matrix form of the conditional variance \(\operatorname{Var}(\mathbf{Y}\mid\mathbf{X}_t,t,\mathbf{p},\mathbf{X}_1(\mathbf{p}))\) and denote it by \(\Gamma_t(\Sigma_{\mathbf{p}})\). Because \(\Sigma_{\mathbf{p}}=U\Lambda U^{\!T}\) for an orthogonal matrix \(U\) and a diagonal matrix \(\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_d)\), all terms in (A.2) commute with \(U\); consequently
\[
\Gamma_t(\Sigma_{\mathbf{p}})=U\Gamma_t(\Lambda)U^{\!T}.
\]
Thus,
\[
\operatorname{tr}(\Gamma_t(\Sigma_{\mathbf{p}}))=\operatorname{tr}(\Gamma_t(\Lambda)).
\]
Consequently, the flow–matching objective can be written entirely in terms of the diagonal matrix \(\Lambda\):
\[
\mathcal{L}_{\text{flow}}(\mathbf{p})=\int_{0}^{1}\operatorname{tr}\big(\operatorname{Var}(\mathbf{X}_1-\mathbf{X}_0\mid\mathbf{X}_t,t,\mathbf{p},\mathbf{X}_1(\mathbf{p}))\big)\,dt
\]
\[
=\int_{0}^{1}\operatorname{tr}(\Gamma_t(\Sigma_{\mathbf{p}}))\,dt
=\int_{0}^{1}\operatorname{tr}(\Gamma_t(\Lambda))\,dt
=\sum_{i=1}^{d}\int_{0}^{1}\Gamma_t(\lambda_i(\mathbf{p}))\,dt.
\]

where the last equality uses that Γt(Λ) = diag(Γt(λ1), …, Γt(λd)) is diagonal, so its trace equals the sum of its diagonal entries.
According to the expression of Γt,
Lflow(p) = ∑_{i=1}^d ∫_0^1 λi / (t^2 λi + (1 − t)^2) dt, λi ≥ 0 (A.3)

Step 3: Calculate the integral above over t ~ U[0,1].
For λ > 0 we consider
I(λ) := ∫_0^1 λ / (t^2 λ + (1 − t)^2) dt.
Set A := λ + 1. Completing the square in the denominator gives
λ t^2 + (1 − t)^2 = (λ + 1) t^2 − 2t + 1 = A(t − 1/A)^2 + λ/A.
First substitution. We introduce
u = √(A/λ)(t − 1/A), dt = √(λ/A) du,
so that
t = 0 ⇒ u0 = −1/√(λA), t = 1 ⇒ u1 = √λ/√A.
Substituting, we obtain
I(λ) = √(λ/A) ∫_{u0}^{u1} du/(u^2 + A^{-1}) = √(λA) ∫_{u0}^{u1} du/(Au^2 + 1).
Second substitution. Absorb the constant A^{-1} by setting
v = √A u, du = dv/√A,
which converts the integrand to the standard form (v^2 + 1)^{-1}. The limits become
v0 = √A u0 = −1/√λ, v1 = √A u1 = √λ.
Collecting the Jacobian factors yields
I(λ) = √λ ∫_{−1/√λ}^{√λ} dv/(v^2 + 1).
The integrand is elementary, giving
I(λ) = √λ [arctan v]_{v=−1/√λ}^{v=√λ}
= √λ (arctan √λ − arctan(−1/√λ))
= √λ (arctan √λ + arctan(1/√λ)).
For any x > 0 the identity arctan x + arctan(1/x) = π/2 holds. Then substituting x = √λ yields
I(λ) = (π/2) √λ.
For λ = 0, the identity also holds obviously.

Applying this to every $\lambda_i$ in (A.3) gives
$$\mathcal{L}_{\text{flow}}(\mathbf{p})=\sum_{i=1}^{d}\frac{\pi}{2}\sqrt{\lambda_i(\mathbf{p})},$$
which is the target of Part 2.

Part 3. We prove $\frac{2}{\pi^2}\mathcal{L}_{\text{flow}}^2(\mathbf{p})\leq \mathcal{L}_A(\mathbf{p})\leq \frac{4}{\pi^2}\mathcal{L}_{\text{flow}}^2(\mathbf{p})$.

By the arithmetic–geometric mean inequality, we have
$$\sum_{i=1}^{d}\lambda_i \leq \left(\sum_{i=1}^{d}\sqrt{\lambda_i(\mathbf{p})}\right)^2 \leq 2\sum_{i=1}^{d}\lambda_i
\quad\Longrightarrow\quad
\frac{1}{2}\left(\sum_{i=1}^{d}\sqrt{\lambda_i(\mathbf{p})}\right)^2 \leq \mathcal{L}_A \leq \left(\sum_{i=1}^{d}\sqrt{\lambda_i(\mathbf{p})}\right)^2.$$

Therefore,
$$\frac{2}{\pi^2}\mathcal{L}_{\text{flow}}^2(\mathbf{p})\leq \mathcal{L}_A(\mathbf{p})\leq \frac{4}{\pi^2}\mathcal{L}_{\text{flow}}^2(\mathbf{p}),$$
which completes Part 3 and hence the proof of Theorem 3.5. □

Remark A.1. Furthermore, under Assumption 3.2, this mutual control relationship naturally extends to both D-optimality and E-optimality objectives. Specifically, their formulations simplify to
$$\mathcal{L}_D(\mathbf{p}) := \det(\Sigma_{\mathbf{p}}) = \prod_{i=1}^{d}\lambda_i(\mathbf{p}), \quad \mathcal{L}_E(\mathbf{p}) := \lambda_{\max}(\Sigma_{\mathbf{p}}). \tag{3}$$
It is evident that both $\mathcal{L}_D$ and $\mathcal{L}_E$ can be polynomially bounded above and below by $\mathcal{L}_A$ due to standard spectral inequalities. Consequently, through the control of $\mathcal{L}_A$, the flow-loss objective also polynomially controls $\mathcal{L}_D$ and $\mathcal{L}_E$ in both directions. Therefore, the flow-loss objective serves as a spectrum-aware surrogate that can also guide D-optimal and E-optimal sensor placement with bounded distortion.

B Full Results of Sensor Placement Optimization

Due to the space limitation of the main text, we present the full results of sensor placement results on turbulent flow and sea temperature here, as a supplement to Fig. 4 and Fig. 5.

Given the same high-capacity reconstruction model, our optimized placement consistently achieves superior performance compared to SSPOR and random baselines, confirming the effectiveness of our joint strategy. Interestingly, SSPOR does not always outperform random sampling; for example, on the sea temperature dataset, it often performs slightly worse. This underscores the importance of co-optimizing sensor placement alongside the reconstruction process.

- Models: PhySense; Placement Strategy: Random; Turbulent Flow #200: 0.1233; Turbulent Flow #100: 0.1527; Turbulent Flow #50: 0.2586; Turbulent Flow #30: 0.5176; Sea Temperature #100: 0.0439; Sea Temperature #50: 0.0452; Sea Temperature #25: 0.0477; Sea Temperature #15: 0.0520
- Models: PhySense; Placement Strategy: SSPOR; Turbulent Flow #200: 0.1143; Turbulent Flow #100: 0.1348; Turbulent Flow #50: 0.1815; Turbulent Flow #30: 0.3397; Sea Temperature #100: 0.0452; Sea Temperature #50: 0.0463; Sea Temperature #25: 0.0454; Sea Temperature #15: 0.0543
- Models: PhySense; Placement Strategy: Optimized; Turbulent Flow #200: 0.1106; Turbulent Flow #100: 0.1257; Turbulent Flow #50: 0.1558; Turbulent Flow #30: 0.2157; Sea Temperature #100: 0.0426; Sea Temperature #50: 0.0430; Sea Temperature #25: 0.0437; Sea Temperature #15: 0.0439
- Relative Promotion: Turbulent Flow #200: 3.23%; Turbulent Flow #100: 6.75%; Turbulent Flow #50: 14.16%; Turbulent Flow #30: 36.52%; Sea Temperature #100: 2.96%; Sea Temperature #50: 4.87%; Sea Temperature #25: 3.74%; Sea Temperature #15: 15.58%

C Implementation Details

In this section, we provide the implementation details of our experiments, including benchmarks, metrics, and implementations. All the experiments are conducted based on PyTorch 2.1.0 [34] and on an A100 GPU server with 144 CPU cores.

## C.1 Benchmarks

While the turbulent flow and sea temperature benchmarks are described in detail in the main text, here we focus on elaborating the key details of our generated car aerodynamic benchmark.

a Computational domain

b Slice of the refined mesh

Figure 8: (a) Computational domain of the CFD simulation. (b) Refinement regions around the surface.

We use a watertight 3D car mesh taken from the ShapeNet V1 [6] dataset, which contains around 50k triangular facets. The aerodynamic pressure on the car surface is simulated using OpenFOAM [20]. As shown in Fig. 8(a), we create a virtual wind tunnel to simulate the airflow around a car. The width, length, and height of the wind tunnel are set to 16 meters, 30 meters, and 8 meters, respectively. The computational mesh around the car surface is refined to a smaller mesh size for more accurate aerodynamics modeling, which is illustrated in Fig. 8(b).

We specify different inlet velocities and yaw angles for different simulation cases. The velocity ranges from 20 m/s to 40 m/s, while the yaw angles range from -10 degrees to 10 degrees. We utilize Reynolds-Averaged Simulation (RAS) with the k-omega SST turbulence model in OpenFOAM. 600 simulation iterations are conducted for each case to ensure convergence. A no-slip boundary condition of the velocity field is defined on the car surface. We simulate 100 cases in total, splitting them into 75 for training and 25 for testing.

## C.2 Metrics

To evaluate the accuracy of the reconstructed physical field, we employ the widely used relative L2 metric, defined as

$$\text{Relative } L_2 = \frac{\|\mathbf{u} - \hat{\mathbf{u}}\|_2}{\|\mathbf{u}\|_2},$$

where u denotes the reference (ground-truth) field and û represents the model reconstruction.

## C.3 Implementations

We consider the following baselines on the turbulent flow and sea temperature benchmarks:

• Reconstruction models: SSPOR, VoronoiCNN, Senseiver, S³GM, and DiffusionPDE.
• Sensor placement strategies: random sampling and SSPOR's sensor selection method.

Due to the geometric complexity of the car’s 3D surface, we compare it against the top two performing reconstruction baselines—Senseiver and DiffusionPDE—for this task. The common setups for all models are shown in Table 8.

Table 8: Common setups for all models in three benchmarks. Other setups follow their original settings.

| Benchmarks    | Turbulent Flow | Sea Temperature | Car Aerodynamics |
| ------------- | -------------- | --------------- | ---------------- |
| Epochs        | 800            | 300             | 300              |
| Batch Size    | 60             | 40              | 1                |
| Learning Rate | 10^-3          | 2 × 10^-4       | 10^-3            |
| Optimizer     | Adam           | Adam            | Adam             |

SSPOR [32] We utilize the official implementation of SSPOR in the PySensor package [8] to evaluate the reconstruction and sensor selection performance of SSPOR. SSPOR employs POD [2] to express


![Figure 61](image_61.jpg)
<!-- Caption: Appendix, index 61 -->



![Figure 62](image_62.jpg)
<!-- Caption: Appendix, index 62 -->


high-dimensional data as a linear combination of several orthonormal eigenmodes obtained from the singular value decomposition (SVD). Specifically, we employ the pysensors.SSPOR class with an SVD basis to obtain the targeted number of sensors from the training data. The selected sensors are then used to reconstruct dense fields from sparse observations either with SSPOR or our method.

VoronoiCNN [14]. VoronoiCNN integrates Voronoi tessellation with convolutional neural networks (CNNs). However, the official implementation uses a simplistic CNN that struggles with complex and dynamic fields. We replace it with a more powerful U-Net [38] architecture, using 64 base feature channels, with 4 downsampling and upsampling layers and skip connections to enhance information flow. Due to its inability to handle irregular car aerodynamic meshes, experiments are conducted only on turbulent flow and sea temperature datasets under a unified setup.

Senseiver [41]. We utilize the official implementation of Senseiver. Senseiver is built upon an implicit neural representation (INR) architecture, which optimizes reconstructions at individual spatial locations rather than over the entire physical field. Specifically, during training, 2048 points are randomly sampled from each field to supervise the model. This inherent sampling randomness further impacts the stability of Senseiver and contributes to its slower convergence. To better assess its full potential, we did not strictly enforce the same training epoch limit used for other baselines. Instead, we allowed Senseiver to train until the default early stopping criterion in its official implementation was triggered. Across the three distinct benchmarks we use, the hyperparameters of Senseiver architecture are carefully tuned based on the official implementation scripts.

S³GM [28]. In our experiments, we follow the official implementation and conduct training and inference. Its backbone is the full U-Net architecture equipped with attention mechanisms and timestep embeddings. We find that when the number of iterative optimization steps is less than 100, S³GM fails to produce clear generation results. Moreover, the generation results can be significantly affected by the random sampling strategy of the sensors, with some cases exhibiting abnormally high relative errors, even greater than 10. Therefore, we increase the number of iterative optimization steps to 1,000 to obtain comparable experimental results.

DiffusionPDE [18]. We borrow the model architectures of the original implementation of DiffusionPDE for the turbulent flow and sea temperature benchmarks, and employ a Transolver [48] as the backbone for the unstructured car aerodynamics benchmark. For inference, we also follow the original implementation configurations, using a total of 2,000 steps for better performance. The guidance weights we search in each benchmark are shown in Table 9.

Table 9: Guidance weight of the observation loss for DiffusionPDE. Our final selection is bolded

| Benchmarks      | Turbulent Flow         | Sea Temperature        | Car Aerodynamics       |
| --------------- | ---------------------- | ---------------------- | ---------------------- |
| Guidance Weight | {1000, 2500, 5000}     | {1000, 2500, 5000}     | {1000, 2500, 5000}     |

PhySense. For the reconstruction model, we use DiT as base models for turbulent flow and sea temperature benchmarks, and switch to Transolver for car aerodynamics benchmark. The detailed model configurations are summarized in Table 10.

Table 10: PhySense model configurations for different benchmarks. "–" in the patch size column indicates that Transolver does not apply the patchify operation.

| Benchmark        | Patch Size | dim | Depth | Heads | dim_head | mlp_dim | Others        |
| ---------------- | ---------- | --- | ----- | ----- | -------- | ------- | ------------- |
| Turbulent Flow   | (2, 2)     | 374 | 8     | 8     | 32       | 374     | –             |
| Sea Temperature  | (3, 3)     | 374 | 12    | 8     | 32       | 374     | –             |
| Car Aerodynamics | –          | 374 | 12    | 8     | 32       | 374     | slice_num=32  |

For sensor placement optimization, we adopt the Adam optimizer to perform gradient-based updates, followed by a projection step implemented via nearest-neighbor search to satisfy the spatial constraints. All models are trained for 5 epochs with a cosine learning rate decay scheduler. The learning rate is closely tied to the spatial scale of the physical domain and is carefully tuned for each setting, as summarized in Table 11.

Table 11: Learning rate of sensor placement optimization on three benchmarks.

| Benchmarks | Turbulent Flow | Sea Temperature | Car Aerodynamics |
|------------|----------------|-----------------|------------------|
| Learning rate | 0.25 | 1 | 0.0025 |

## D Additional Analysis

### D.1 Comparison with AdaLN-Zero

Here, we compare our sparse sensor incorporation strategy on turbulent flow with a widely used conditioning approach in diffusion models, namely the AdaLN-Zero block in DiT [35]. As shown in Table 12, replacing our design with AdaLN-Zero leads to a substantial degradation in performance, with reconstruction errors approaching 1. This is because sensor placement inherently captures localized information, whereas AdaLN-Zero aggregates all sensor data into a single global variable, which is insufficient for preserving the spatial specificity required in reconstruction tasks.

Table 12: Ablation on sparse information incorporation via the AdaLN-Zero block, i.e., *Rep. AdaLN-Zero*

| Sensor Number | #200 | #100 | #50 | #30 |
|---------------|------|------|------|------|
| *Rep. AdaLN-Zero* | 0.9179 | 0.9486 | 0.9695 | >1 |
| Ours | **0.1233** | **0.1527** | **0.2586** | **0.5176** |

### D.2 Hyperparameter Sensitivity

As we adopt projected gradient descent to optimize sensor placement, the learning rate η in Eq. (2) becomes a critical factor, especially when operating under complex geometric and high-dimensional surface constraints. We investigate the learning rate sensitivity on the car aerodynamics benchmark. As shown in Table 13, the optimization process is relatively insensitive to learning rate variations when the number of sensors exceeds 30. In contrast, the 15-sensor setting—covering only 0.016% of the spatial domain—exhibits noticeable sensitivity, indicating the increased difficulty of optimization under extreme sparsity. We ultimately select η = 0.0025 as the final choice.

Table 13: Impact of learning rate on sensor placement optimization in the car aerodynamics benchmark. We ultimately select η = 0.0025 as the final choice, which is **bolded**.

| Learning rate | 0.005 | **0.0025** | 0.001 | 0.0005 |
|---------------|-------|------------|-------|--------|
| 30 sensors | 0.0379 | **0.0372** | 0.0369 | 0.0372 |
| 15 sensors | 0.0423 | **0.0386** | 0.0389 | 0.0400 |

### D.3 Statistical Analysis

Table 14: Evaluation results of the PhySense reconstruction model under *random sensor placements* with varying sensor counts (10–200) on the car aerodynamics benchmark. For each setting, the reconstruction is repeated three times, and the mean and variance of the relative L2 loss are reported.

| Sensor Number | #200 | #100 | #50 | #30 | #15 |
|---------------|------|------|------|------|------|
| Sampling 1 | 0.0376 | 0.0386 | 0.0396 | 0.0414 | 0.0458 |
| Sampling 2 | 0.0379 | 0.0378 | 0.0387 | 0.0414 | 0.0436 |
| Sampling 3 | 0.0370 | 0.0382 | 0.0401 | 0.0419 | 0.0501 |
| Mean | **0.0375** | **0.0382** | **0.0395** | **0.0416** | **0.0465** |
| Standard deviation | 0.0004 | 0.0003 | 0.0006 | 0.0002 | 0.0027 |

In this section, we analyze the statistical significance of our experimental results. Our pipeline consists of two stages: training the reconstruction model and optimizing the sensor placement. Due to the high computational cost of training the base reconstruction model, we train it only once. The sensitivity of sensor placement initialization has been discussed in Section 4.4. Our findings show that while different initializations result in different placements, each optimized placement is effective.

Table 15: Evaluation results of the PhySense reconstruction model under optimized sensor placements with varying sensor counts (10–200) on the car aerodynamics benchmark. For each setting, the reconstruction is repeated three times, and the mean and standard deviation of the relative L2 loss are reported.

| Sensor Number       | #200   | #100   | #50    | #30    | #15    |
| ------------------- | ------ | ------ | ------ | ------ | ------ |
| Sampling 1          | 0.0369 | 0.0371 | 0.0369 | 0.0370 | 0.0378 |
| Sampling 2          | 0.0367 | 0.0372 | 0.0369 | 0.0374 | 0.0389 |
| Sampling 3          | 0.0370 | 0.0368 | 0.0373 | 0.0373 | 0.0390 |
| Mean                | 0.0369 | 0.0370 | 0.0370 | 0.0372 | 0.0386 |
| Standard deviation  | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0005 |

Furthermore, since our task is a reconstruction problem, and our reconstruction model essentially learns a conditional distribution, we report in the table above the mean and standard deviation of reconstruction loss under multiple samplings on the car aerodynamics benchmark, across varying numbers of sensors. As shown in Tables 14 and 15, when the number of sensors exceeds 15, the standard deviation becomes relatively small, indicating stable performance. Moreover, regardless of the sensor count, once the placement is optimized, the reconstruction results exhibit consistent stability across multiple runs.

## D.4 Additional Metric for Gradients

Table 16: Comparison on the sea temperature benchmark. Relative L2 error of gradient fields is reported.

| Sea Temperature (relative L2 of gradient field) | #100   | #50    | #25    | #15    |
| ----------------------------------------------- | ------ | ------ | ------ | ------ |
| Senseiver + random sensor placement             | 0.5456 | 0.5470 | 0.5501 | 0.5515 |
| PhySense + random sensor placement              | 0.4877 | 0.4884 | 0.4890 | 0.4892 |
| Relative improvement                            | 11.87% | 11.98% | 12.51% | 12.74% |

In this section, we additionally evaluate the relative L2 error of the gradient fields on the sea temperature benchmark to better reflect whether the model captures meaningful physical fidelity. As shown in Table 16, PhySense achieves significantly lower gradient error compared to the second-best model, Senseiver, demonstrating its superior spatial physical fidelity.

## D.5 Uncertainty Quantification

Figure 9: Visualization of uncertainty quantification for the flow-based reconstruction model on the turbulent flow benchmark. The sample variance is notably smaller than the physical field and error.

Flow-based reconstruction models explicitly learn the full conditional data distribution, which inherently facilitates uncertainty quantification. To complete this, we reconstruct each input three times, each with a distinct random noise initialization. Subsequently, we compute the pixel-wise sample mean and sample variance across these reconstructions. As illustrated in Fig. 9, the magnitude of the variance is orders of magnitude smaller than that of the sample mean and the reconstruction error. This indicates high confidence in our model, as it consistently produces similar reconstructions irrespective of the initial noise vector.


![Figure 76](image_76.jpg)
<!-- Caption: Appendix, index 76 -->


E Limitations and Future Work

While our current formulation assumes point-wise sensor observations, this may not fully capture the behavior of certain real-world sensors that integrate information over small spatial neighborhoods. For example, radar and precipitation sensors typically respond to local regions rather than single points. Incorporating regional observations into both the reconstruction model and the sensor placement optimization could further improve the effectiveness of our approach. As a future direction, we aim to extend both our model and theoretical framework from point-wise observations to region-based sensing, which better reflects the behavior of many real-world sensors.

F Broader Impacts

We introduce PhySense, a synergistic two-stage framework for accurate physics sensing with theoretical guarantees. Extensive experiments validate its superior reconstruction accuracy and informative sensor placements. Beyond performance gains, our work offers a new perspective on the sensor placement problem from a deep learning standpoint. Thus, PhySense may inspire some future research in relevant domains. Moreover, our optimized placement could potentially inform how sensors are placed in more applications, enhancing their reconstruction accuracy.

Since our work is purely focused on algorithmic design for accurate physics sensing, there are no potential negative social impacts or ethical risks.