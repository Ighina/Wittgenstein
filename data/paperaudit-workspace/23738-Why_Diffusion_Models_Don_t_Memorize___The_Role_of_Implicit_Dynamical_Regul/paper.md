
## Abstract


Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training

Tony Bonnaire†
LPENS
Université PSL, Paris
tony.bonnaire@phys.ens.fr

Raphaël Urfin†
LPENS
Université PSL, Paris
raphael.urfin@phys.ens.fr

Giulio Biroli
LPENS
Université PSL, Paris
giulio.biroli@phys.ens.fr

Marc Mézard
Department of Computing Sciences
Bocconi University, Milano
marc.mezard@unibocconi.it

Abstract

Diffusion models have achieved remarkable success across a wide range of generative tasks. A key challenge is understanding the mechanisms that prevent their memorization of training data and allow generalization. In this work, we investigate the role of the training dynamics in the transition from generalization to memorization. Through extensive experiments and theoretical analysis, we identify two distinct timescales: an early time $$\tau_{\text{gen}}$$ at which models begin to generate high-quality samples, and a later time $$\tau_{\text{mem}}$$ beyond which memorization emerges. Crucially, we find that $$\tau_{\text{mem}}$$ increases linearly with the training set size $$n$$, while $$\tau_{\text{gen}}$$ remains constant. This creates a growing window of training times with $$n$$ where models generalize effectively, despite showing strong memorization if training continues beyond it. It is only when $$n$$ becomes larger than a model-dependent threshold that overfitting disappears at infinite training times. These findings suggest a significant role of implicit dynamical regularization in the training dynamics that can substantially reduce memorization under certain conditions in overparameterized settings. Our results are supported by numerical experiments with standard U-Net architectures on realistic and synthetic datasets, and by a theoretical analysis using a tractable random features model studied in the high-dimensional limit.


## Introduction


1 Introduction

Diffusion Models (DMs) [43, 17, 48, 49] achieve state-of-the-art performance in a wide variety of AI tasks such as the generation of images [40], audios [57], videos [29], and scientific data [27, 35]. This class of generative models, inspired by out-of-equilibrium thermodynamics [43], corresponds to a two-stage process: the first one, called forward, gradually adds noise to data, whereas the second one, called backward, generates new data by denoising Gaussian white noise samples. In DMs, the reverse process typically involves solving a stochastic differential equation (SDE) with a force field called score. However, it is also possible to define a deterministic transport through an ordinary differential equation (ODE), treating the score as a velocity field, an approach that is for instance followed in flow matching [28].

Understanding the generalization properties of score-based generative methods is a central issue in machine learning, and a particularly important question is how memorization of the training set

†Equal contribution.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

Figure 1: Qualitative summary of our contributions. (Left) Illustration of the training dynamics of a diffusion model. Depending on the training time τ, we identify three regimes measured by the inverse quality of the generated samples (blue curve) and their memorization fraction (red curve). The generalization regime extends over a large window of training times which increases with the training set size n. On top, we show a one-dimensional example of the learned score function during training (orange). The gray line gives the exact empirical score at a given noise level, while the black dashed line corresponds to the true (population) score. (Right) Phase diagram in the (n, p) plane illustrating three regimes of diffusion models: Memorization when n is sufficiently small at fixed p, Architectural Regularization for n > n^*(p) (which is model and dataset dependent, as discussed in [12, 22]), and Dynamical Regularization, corresponding to a large intermediate generalization regime obtained when the training dynamics is stopped early, i.e., τ ∈ [τ_gen, τ_mem].

This is avoided in practice. A model without regularization achieving zero training loss only learns the empirical score, and is bound to reproduce samples of the training dataset at the end of the backward process. This memorization regime [26, 4] is empirically observed when the training set is small and disappears when it increases beyond a model-dependent threshold [21]. Understanding the mechanisms controlling this change of regimes from memorization to generalization is a central challenge for both theory and applications. Model regularization and inductive biases imposed by the network architecture were shown to play a role [22], as well as a dynamical regularization due to the finiteness of the learning rate [55]. However, the regime shift described above is consistently observed even in models where all these regularization mechanisms are present. This suggests that an important mechanism behind the transition from memorization to generalization may differ from prior explanations. In this work, we present evidence—through numerical experiments and theoretical analysis of a simplified model—that implicit dynamical biases during training can contribute to this transition by favoring generalizing solutions.

Contributions and theoretical picture. We investigate the dynamics of score learning using gradient descent, both numerically and analytically, and study the generation properties of the score depending on the time τ at which the training is stopped. The theoretical picture built from our results and combining several findings from the recent literature is illustrated in Fig. 1. The two main parameters are the size of the training set n and the expressivity of the class of score functions on which one trains the model, characterized by a number of parameters p; when both n and p are large one can identify three main regimes. Given p, if n is larger than n^*(p) (which depends on the training set and on the class of scores), the score model is not expressive enough to represent the empirical score associated to n data, and instead provides a smooth interpolation, approximately independent of the training set. In this regime, even with a very large training time τ → ∞, memorization does not occur because the model is regularized by its architecture and the finite number of parameters. When n < n^*(p) the model is expressive enough to memorize, and two timescales emerge during training: one, τ_gen, is the minimum training time required to achieve high-quality data generation; the second, τ_mem > τ_gen, signals when further training induces memorization, and causes the model to

Number of parameters p


![Figure 8](image_8.jpg)
<!-- Caption: Introduction, index 8 -->



![Figure 9](image_9.jpg)
<!-- Caption: Introduction, index 9 -->


increasingly reproduce the training samples (left panel). The first timescale, $\tau_{\text{gen}}$, is found to be independent of $n$, whereas the second, $\tau_{\text{mem}}$, grows approximately linearly with $n$, thus opening a large window of training times during which the model generalizes if early-stopped when $\tau \in [\tau_{\text{gen}}, \tau_{\text{mem}}]$. Our results show that implicit dynamical regularization in training plays a crucial role in score-based generative models, substantially enlarging the generalization regime (see right panel of Fig. 1), and hence allowing one to avoid memorization even in highly overparameterized settings. We find that the key mechanism behind the widening gap between $\tau_{\text{gen}}$ and $\tau_{\text{mem}}$ is the irregularity of the empirical score at low noise level and large $n$. In this regime the models used to approximate the score provide a smooth interpolation that remains stable for a long period of training times and closely approximates the population score, a behavior likely rooted in the spectral bias of neural networks [37]. Only at very long training times do the dynamics converge to the low-lying minimum corresponding to the empirical score, leading to memorization (as illustrated in the 1D examples in the left panel of Fig. 1).

The theoretical picture described above is based on our numerical and analytical results, and builds upon previous works, in particular numerical analyses characterizing the memorization–generalization transition [15, 56], analytical works on memorization of DMs [12, 22, 21], and studies on the spectral bias of deep neural networks [37]. Our numerical experiments† use a class of scores based on a realistic U-Net [41] trained on downscaled images of the CelebA dataset [30]. By varying $n$ and $p$, we measure the evolution of the sample quality (through FID) and the fraction of memorization during learning, which support the theoretical scenario presented in Fig. 1. Additional experimental results on synthetic data are provided in Supplemental Material (SM, Sects. A and B). On the analytical side, we focus on a class of scores constructed from random features and simplified models of data, following [12]. In this setting, the timescales of training dynamics correspond directly to the inverse eigenvalues of the random feature correlation matrix. Leveraging tools from random matrix theory, we compute the spectrum in the limit of large datasets, high-dimensional data, and overparameterized models. This analysis reveals, in a fully tractable way, how the theoretical picture of Fig. 1 emerges within the random feature framework.


## Related Work


\n\n\n\n

Related works.

• The memorization transition in DMs has been the subject of several recent empirical investigations [7, 44, 45] which have demonstrated that state-of-the-art image DMs – including Stable Diffusion and DALL·E – can reproduce a non-negligible portion of their training data, indicating a form of memorization. Several additional works [15, 56] have examined how this phenomenon is influenced by factors such as data distribution, model configuration, and training procedure, and provide a strong basis for the numerical part of our work.

• A series of theoretical studies in the high-dimensional regime have analyzed the memorization–generalization transition during the generative dynamics under the empirical score assumption [4, 1, 51], showing how trajectories are attracted to the training samples. Within this high-dimensional framework, [8, 9, 54, 12] study score learning for various model classes. In particular, [12] uses a Random Feature Neural Network [38]. The authors compute the asymptotic training and test losses for $\tau \to \infty$ and relate them to memorization. The theoretical part of our work generalizes this approach to study the role of training dynamics and early stopping in the memorization–generalization transition.

• Recent works have also uncovered complementary sources of implicit regularization explaining how DMs avoid memorization. Architectural biases and limited network capacity were, for instance, shown to constrain memorization in [22, 21], and the finiteness of the learning rate prevents the model from learning the empirical score in [55]. Also related to our analysis, [25] provides general bounds showing the beneficial role of early stopping of the training dynamics to enhance generalization for finitely supported target distributions, as well as a study of its effect for one-dimensional Gaussian mixtures.

• Finally, previous studies on supervised learning [37, 58], and more recently on DMs [53], have shown that deep neural networks display a frequency-dependent learning speed, and hence a learning bias towards low-frequency functions. This fact plays an important role in the results we present since the empirical score contains a low-frequency part that is close to the population score, and a high-frequency part that is dataset-dependent. To the best of our knowledge, the training time to learn the high-frequency part, and hence memorize, which we find to scale with $n$, has not been studied from this perspective in the context of score-based generative methods.


## Preliminaries


Setting: generative diffusion and score learning. Standard DMs define a transport from a target distribution $P_0$ in $\mathbb{R}^d$ to a Gaussian white noise $\mathcal{N}(0, I_d)$ through a forward process defined as an Ornstein-Uhlenbeck (OU) stochastic differential equation (SDE):
$$\mathrm{d}\mathbf{x} = -\mathbf{x}(t)\mathrm{d}t + \mathrm{d}\mathbf{B}(t),$$
(1)
where $\mathrm{d}\mathbf{B}(t)$ is the square root of two times a Wiener process. Generation is performed by time-reversing the SDE (1) using the score function $\mathbf{s}(\mathbf{x}, t) = \nabla_{\mathbf{x}} \log P_t(\mathbf{x})$,
$$-\mathrm{d}\mathbf{x} = [\mathbf{x}(t) + 2\mathbf{s}(\mathbf{x}, t)] \mathrm{d}t + \mathrm{d}\mathbf{B}(t),$$
(2)
where $P_t(\mathbf{x})$ is the probability density at time $t$ along the forward process, and the noise $\mathrm{d}\mathbf{B}(t)$ is also the square root of two times a Wiener process. As shown in the seminal works [20, 52], $\mathbf{s}(\mathbf{x}, t)$ can be obtained by minimizing the score matching loss
$$\hat{\mathbf{s}}(\mathbf{x}, t) = \arg \min_{\mathbf{s}} \mathbb{E}_{\mathbf{x}\sim P_0, \boldsymbol{\xi}\sim\mathcal{N}(0,I_d)} \left[ \left\| \sqrt{\Delta_t}\mathbf{s}(\mathbf{x}(t), t) + \boldsymbol{\xi} \right\|^2 \right],$$
(3)
where $\Delta_t = 1 - e^{-2t}$. In practice, the optimization problem is restricted to a parametrized class of functions $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}(t), t)$ defined, for example, by a neural network with parameters $\boldsymbol{\theta}$. The expectation over $\mathbf{x}$ is replaced by the empirical average over the training set ($n$ iid samples $\mathbf{x}^{\nu}$ drawn from $P_0$),
$$\mathcal{L}_t(\boldsymbol{\theta}, \{\mathbf{x}^{\nu}\}_{\nu=1}^n) = \frac{1}{n} \sum_{\nu=1}^n \mathbb{E}_{\boldsymbol{\xi}\sim\mathcal{N}(0,I_d)} \left[ \left\| \sqrt{\Delta_t}\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}^{\nu}(t), t) + \boldsymbol{\xi} \right\|^2 \right],$$
(4)
where $\mathbf{x}_t^{\nu}(\boldsymbol{\xi}) = e^{-t}\mathbf{x}^{\nu} + \sqrt{\Delta_t}\boldsymbol{\xi}$. The loss in (4) can be minimized with standard optimizers, such as stochastic gradient descent [SGD, 39] or Adam [24]. In practice, a single model conditioned on the diffusion time $t$ is trained by integrating (4) over time [23]. The solution of the minimization of (4) is the so-called empirical score (e.g. [4, 26]), defined as $\mathbf{s}_{\text{emp}}(\mathbf{x}, t) = \nabla_{\mathbf{x}} \log P_t^{\text{emp}}(\mathbf{x})$, with
$$P_t^{\text{emp}}(\mathbf{x}) = \frac{1}{n\,(2\pi\Delta_t)^{d/2}} \sum_{\nu=1}^n e^{-\frac{1}{2\Delta_t}\|\mathbf{x}-\mathbf{x}^{\nu}e^{-t}\|_2^2}.$$
(5)
This solution is known to inevitably recreate samples of the training set at the end of the generative process (i.e., it perfectly memorizes), unless $n$ grows exponentially with the dimension $d$ [4]. However, this is not the case in many practical applications where memorization is only observed for relatively small values of $n$, and disappears well before $n$ becomes exponentially large in $d$. The empirical minimization performed in practice, within a given class of models and a given minimization procedure, does not drive the optimization to the global minimum of (4), but instead to a smoother estimate of the score that is independent of the training set with good generalization properties [21], as the global minimum of (3) would do. Understanding how it is possible, and in particular the role played by the training dynamics to avoid memorization, is the central aim of the present work.


## Experiments


2 Generalization and memorization during training of diffusion models

Data & architecture. We conduct our experiments on the CelebA face dataset [30], which we convert to grayscale downsampled images of size $d = 32 \times 32$, and vary the training set size $n$ from 128 up to 32768. Our score model has a U-Net architecture [41] with three resolution levels and a base channel width of $W$ with multipliers 1, 2 and 3 respectively. All our networks are DDPMs [17] trained to predict the injected noise at diffusion time $t$ using SGD with momentum at fixed batch size $\min(n, 512)$. The models are all conditioned on $t$, i.e. a single model approximates the score at all times, and make use of a standard sinusoidal position embedding [50] that is added to the features of each resolution. More details about the numerical setup can be found in SM (Sect. A).

Evaluation metrics. To study the transition from generalization to memorization during training, we monitor the loss (4) during training using a fixed diffusion time $t = 0.01$. At various numbers of SGD updates $\tau$, we compute the loss on all $n$ training examples (training loss) and on a held-out test set of 2048 images (test loss). To characterize the score obtained after a training time $\tau$, we assess the originality and quality of samples by generating 10K samples using a DDIM accelerated sampling [46]. We compute (i) the Fréchet-Inception Distance [FID, 16] against 10K test samples which we use to identify the generalization time $\tau_{\text{gen}}$; and (ii) the fraction of memorized generated

Figure 2: Memorization transition as a function of the training set size n for U-Net score models on CelebA. (Left) FID (solid lines, left axis) and memorization fraction f_mem (dashed lines, right axis) against training time τ for various n. Inset: normalized memorization fraction f_mem(τ)/f_mem(τ_max) with the rescaled time τ/n. (Middle) Training (solid lines) and test (dashed lines) loss with τ for several n at fixed t = 0.01. Inset: both losses plotted against τ/n. Error bars on the losses are imperceptible. (Right) Generated samples from the model trained with n = 1024 for τ = 100K or τ = 1.62M steps, along with their nearest neighbors in the training set.

We track the fraction of memorized samples f_mem(τ), which gives access to τ_mem, the memorization time. Following previous numerical studies [56, 15], a generated sample x_τ is considered memorized if
E_{x_τ}[ ||x_τ − a^{μ_1}||_2 / ||x_τ − a^{μ_2}||_2 ] < k, (6)
where a^{μ_1} and a^{μ_2} are the nearest and second-nearest neighbors of x_τ in the training set in the L2 sense. Here, we use k = 1/3 as in prior studies, noting that moderate variations in k strongly affect the exact values of f_mem, and the scaling behavior of τ_mem with n is less robust than suggested. Error bars in the figures correspond to twice the standard deviation over 5 different test sets for FIDs, and 5 noise realizations for L_train and L_test. For f_mem, we report the 95% CIs on the mean evaluated with 500 bootstrap samples and with limited generated sample sizes, cautioning that reported variances may be underestimated.

Role of training set size on the learning dynamics. At fixed model capacity (p = 4 × 10^6, base width W = 32), we investigate how the training set size n impacts the previous metrics. In the left panel of Fig. 2, we report the FID (solid lines) and f_mem(τ) (dashed lines) for various n, after excluding runs where early stopping failed to improve FID by more than 15%, resulting in reduced variance but biased estimates. All training dynamics exhibit two phases. First, the FID quickly decreases to reach a minimum value on a timescale τ_gen (≈ 100K) that does not depend on n. In the right panel, the generated samples at τ = 100K clearly differ from their nearest neighbors in the training set, indicating that the model generalizes correctly. Beyond this time, the FID remains flat. However, models with larger capacity or trained longer by design show earlier increases in f_mem(τ), clouding the determination of τ_mem and suggesting that unreported training durations influence the memorization regime. Both the transition time τ_mem and the value of the final fraction f_mem(τ_max) (with τ_max being one to four million SGD steps) vary with n. The inset plot shows the normalized memorization fraction f_mem(τ)/f_mem(τ_max) against the rescaled time τ/n, making all curves collapse and increase at around τ/n ≈ 300, showing that τ_mem ∝ n, and demonstrating the existence of a generalization window for τ ∈ [τ_gen, τ_mem] that widens linearly with n, as illustrated in the left panel of Fig. 1.

As highlighted in the introduction, memorization in DMs is ultimately driven by the overfitting of the empirical score s_mem(x, t). The evolution of L_train(τ) and L_test(τ) at fixed t = 0.01 are shown in the middle panel of Fig. 2 for n ranging from 512 to 32,768. Initially, the two losses remain nearly indistinguishable, indicating that the learned score s_θ(x, t) does not depend on the training set. Beyond a critical time, L_train continues to decrease while L_test increases, leading to a nonzero generalization loss whose magnitude depends on n. As n increases, this critical time also increases and, eventually, the training and test loss gap shrinks: for n = 32,768, the test loss remains close to the training loss, even after 11 million SGD steps. The inset shows the evolution of both losses with τ/n, demonstrating that the overfitting time scales linearly with the training set size n, just like τ_mem identified in the left panel. Moreover, there is a consistent lag between the overfitting time and τ_mem at fixed n, reflecting the additional training required for the model to overfit the empirical score sufficiently to reproduce the training samples, and therefore to impact the memorization fraction.


![Figure 17](image_17.jpg)
<!-- Caption: Experiments, index 17 -->



![Figure 18](image_18.jpg)
<!-- Caption: Experiments, index 18 -->



![Figure 19](image_19.jpg)
<!-- Caption: Experiments, index 19 -->



![Figure 20](image_20.jpg)
<!-- Caption: Experiments, index 20 -->


Figure 3: Effect of the number of parameters in the U-Net architecture on the timescales of the training dynamics. (Left) FID (panels A, B) and normalized memorization fraction f_mem(τ)/f_mem(τ_max) (panels C, D) for various n and W during training. In panels B and D, time is rescaled such that all curves collapse. (Right) (n, p) phase diagram of generalization vs memorization for U-Nets trained on CelebA. Curves show, for τ ∈ {τ_gen, 3τ_gen, 8τ_gen}, the minimal dataset size n(p) satisfying f_mem(τ) = 0. The shaded background indicates the memorization–generalization boundary for τ = τ_gen.

Memorization is not due to data repetition. While initially hypothesized that delayed memorization scales with n beyond data repetition effects, alternative experiments with different batch sizes and update rules produce conflicting results, leaving the exact cause of memorization delay open. In other words, even if at fixed τ all models have processed each sample equally often, larger n consistently postpone memorization. This confirms that memorization in DMs is driven by a fundamental n-dependent change in the loss landscape—not by sample repetition during training.

Effect of the model capacity. To study more precisely the role of the model capacity on the memorization–generalization transition, we vary the number of parameters p by changing the U-Net’s base width W ∈ {8, 16, 32, 48, 64}, resulting in a total of p ∈ {0.26, 1, 4, 9, 16} × 10^6 parameters. In the left panel of Fig. 3, we plot both the FID (top row) and the normalized memorization fraction (bottom row) as functions of τ for several widths W and training set sizes n. Panels A and C demonstrate that higher-capacity networks (larger W) achieve high-quality generation and begin to memorize earlier than smaller ones. Panels B and D show that the two characteristic timescales simply scale as τ_gen ∝ W^−1 and τ_mem ∝ nW^−1. In particular, this implies that, for W > 8, the critical training set size n_gm(p) at which τ_mem = τ_gen is approximately independent of p (at least on the limited values of p we focused on). When n > n_gm(p), the interval [τ_gen, τ_mem] opens up, so that early stopping within this window yields high-quality samples without memorization. In the right panel of Fig. 3, we display this boundary (solid line) in the (n, p) plane by fixing the training time to τ = τ_gen, that we identify numerically using the collapse of all FIDs at around Wτ_gen ≈ 3 × 10^6 (see panel B), and computing the smallest n such that f_mem(τ) = 0. The resulting solid curve delineates two regimes: below the curve, memorization already starts at τ_gen; above the curve, the models generalize perfectly under early stopping. We repeat this experiment for τ = 3τ_gen and τ = 8τ_gen, showing saturation to larger and larger p as τ increases. Eventually, for τ → ∞, we expect these successive boundaries to converge to the architectural regularization threshold n^*(p), i.e., the point beyond which the network avoids memorization because it is not expressive enough, as found in [12] and highlighted in the right panel of Fig. 1. In order to estimate n^*(p), we measure for a given τ the largest n(τ) yielding f_mem ≈ 0. The curve n(τ) approaches n^*(p) for large τ. We therefore estimate n^*(p) by measuring the asymptotic values of n(τ), which in practice is reached already at τ = τ_max = 2M updates for the values of W we focus on.


![Figure 24](image_24.jpg)
<!-- Caption: Experiments, index 24 -->



![Figure 25](image_25.jpg)
<!-- Caption: Experiments, index 25 -->



![Figure 26](image_26.jpg)
<!-- Caption: Experiments, index 26 -->



![Figure 27](image_27.jpg)
<!-- Caption: Experiments, index 27 -->



## Method


3 Training dynamics of a Random Features Network

Notations. We use bold symbols for vectors and matrices. The L^2 norm of a vector x is denoted by ‖x‖ = (∑_i x_i^2)^{1/2}. We write f = O(g) to mean that in the limit n, p → ∞, there exists a constant C such that |f| ≤ C|g|.

Setting. We study analytically a model introduced in [12], where the data lie in d dimensions. We parametrize the score with a Random Features Neural Network [RFNN, 38]

s_A(x) = (A/√p) σ(Wx/√d).
(7)

An RFNN, illustrated in Fig. 4 (left), is a two-layer neural network whose first-layer weights (W ∈ R^{p×d}) are drawn from a Gaussian distribution and remain frozen while the second-layer weights (A ∈ R^{d×p}) are learned during training. This model has already served as theoretical framework for studying several behaviors of deep neural networks such as the double descent phenomenon [31, 10]. σ is an element-wise non-linear activation function. We consider a training set of n iid samples x^ν ∼ P_x for ν = 1, …, n and we focus on the high-dimensional limit d, p, n → ∞ with the ratios ψ_p = p/d, ψ_n = n/d kept fixed. We study the training dynamics associated to the minimization of the empirical score matching loss defined in (4) at a fixed diffusion time t. This is a simplification compared to practical methods, which use a single model for all t. It has been already studied in previous theoretical works [8, 12]. The loss (4) is rescaled by a factor 1/d in order to ensure a finite limit at large d. We also study the evolution of the test loss evaluated on test points and the distance to the exact score s(x) = ∇ log P_x,

L_test = (1/d) E_{x,ξ} [ ‖√Δ_t s_A(x_t(ξ)) + ξ‖^2 ],  E_score = (1/d) E_x [ ‖s_A(x) − ∇ log P_x‖^2 ],
(8)

where the expectations E_{x,ξ} are computed over x ∼ P_x and ξ ∼ N(0, I_d). The generalization loss, loosely defined as L_gen = L_test − L_train, is commonly used as an overfitting indicator, although it may not precisely capture true generalization properties. The distance to the exact score E_score relates to generation quality but assumptions about KL divergence bounds should be treated cautiously. The weights A are updated via gradient descent

A^{(k+1)} = A^{(k)} − η ∇_A L_train(A^{(k)}),
(9)

where η is the learning rate. In the high-dimensional limit, as the learning rate η → 0, and after rescaling time as τ = kη/d^2, the discrete-time dynamics converges to the following continuous-time gradient flow:

\dot{A}(τ) = −d^2 ∇_A L_train(A(τ)) = −2Δ_t (d/p) A U − (2d√Δ_t/√p) V^T,
(10)

with

U = (1/n) ∑_{ν=1}^n E_ξ [ σ(Wx_t^ν(ξ)/√d) σ(Wx_t^ν(ξ)/√d)^T ],   V = (1/n) ∑_{ν=1}^n E_ξ [ σ(Wx_t^ν(ξ)/√d) ξ^T ].
(11)

Assumptions. We consider data distributions P_x with heavy-tailed components and possibly unbounded covariance spectrum, violating previous boundedness conditions. Additionally, we assume σ permits a non-standard Hermite expansion that may include non-square-integrable terms, departing from classical theory. We assume that the empirical distribution of eigenvalues of Σ converges weakly in the high-dimensional limit to a deterministic density ρ_Σ(λ) and that Tr(Σ)/d converges to a finite limit (for a more precise mathematical statement see SM Sect. C.3). Moreover, we make additional assumptions that are not essential to the proofs but which simplify the analysis: (iii) the activation function σ verifies μ_0 = E_z[σ(z)] = 0; and (iv) the second layer A is initialized with zero weights A(τ = 0) = 0. In numerical applications, unless specified, we use σ(z) = tanh(z) and P_x = N(0, I_d).

Figure 4: (Left) Illustration of an RFNN. (Middle/Right) Spectrum of U. Density ρ(λ) from Theorem 3.1 in the overparameterized Regime I described in Theorem 3.2, with ψ_p = 64, ψ_n = 8, t = 0.01, and ρ_Σ(λ) = δ(λ − 1). The bulk of the spectrum (orange) is between λ ≈ 10 and λ ≈ 45. The histogram shows the eigenvalues from a single realization of U at d = 100. Inset: zoom near λ = 0 (in blue) showing the first bulk ρ_1 and the delta peak at λ = s_t^2. (Right) Same as (Middle), but with ρ_Σ(λ) = ½ δ(λ − 0.5) + ½ δ(λ − 1.5). The first bulk in blue remains unchanged, as it depends only on σ_x^2 = Tr(Σ)/d = 1 in both cases, while the second bulk varies with Σ.

\n\n\n\n
Figure 5: Evolution of the training and test losses for the RFNN. (A) Distance to the true score ℰ_score against training time τ for ψ_n = 4, 8, 16, 32, ψ_p = 64, t = 0.1 and d = 100. In the inset, the training time is rescaled by τ_mem = ψ_p/(Δ_t λ_min). (B) Training (solid) and test (dashed) losses for various ψ_n. The inset shows both losses rescaled by τ_mem. (C) Heatmaps of ℒ_gen for τ = 10^3 (top) and τ = 10^4 (bottom) as a function of ψ_n and ψ_p. All the curves use PyTorch [34] gradient descent. More numerical details can be found in SM Sect. D.

\n\n\n\n
Emergence of the two timescales during training. We first show in Fig. 5 that the behavior of training and test losses in the RF model mirrors the one found in realistic cases in Sect. 2, with a separation of timescales τ_gen and τ_mem which increases with n. Equation (10) is linear in A and hence it can be solved exactly (see SM). The timescales of the training dynamics are given by the inverse eigenvalues of the p × p matrix Δ_t \mathbf{U}/ψ_p. Building on the Gaussian Equivalence Principle [GEP, 13, 14, 32] and the theory of linear pencils [5], George et al. (2025) derive a coupled system of equations characterizing the Stieltjes transform of the eigenvalue density ρ(λ) of U for isotropic Gaussian data that lie in a D-dimensional subspace with D ≤ d and D = O(d). We offer an alternative derivation presented in SM for general variance using the replica method [33]—a heuristic method from the statistical physics of disordered systems—yielding the more compact formulation for obtaining the spectrum stated in Theorem 3.1. Before stating the theorem, we introduce

$$b_t = \mathbb{E}_{u,v}\!\left[v\,\sigma\!\left(e^{-t}\sigma_x u + \sqrt{\Delta_t}\,v\right)\right], \qquad
a_t = \mathbb{E}_{u,v}\!\left[\sigma\!\left(e^{-t}\sigma_x u + \sqrt{\Delta_t}\,v\right)\frac{u}{e^{-t}\sigma_x}\right], \tag{12}$$

$$v_t^2 = \mathbb{E}_{u,v,w}\!\left[\sigma\!\left(e^{-t}\sigma_x u + \sqrt{\Delta_t}\,v\right)\sigma\!\left(e^{-t}\sigma_x u + \sqrt{\Delta_t}\,w\right)\right] - a_t^2 e^{-2t}\sigma_x^2, \tag{13}$$

$$s_t^2 = \mathbb{E}_u\!\left[\sigma(\Gamma_t u)^2\right] - a_t^2 e^{-2t}\sigma_x^2 - v_t^2 - b_t^2. \tag{14}$$


![Figure 32](image_32.jpg)
<!-- Caption: Method, index 32 -->



![Figure 33](image_33.jpg)
<!-- Caption: Method, index 33 -->



![Figure 34](image_34.jpg)
<!-- Caption: Method, index 34 -->



![Figure 35](image_35.jpg)
<!-- Caption: Method, index 35 -->



![Figure 36](image_36.jpg)
<!-- Caption: Method, index 36 -->



![Figure 37](image_37.jpg)
<!-- Caption: Method, index 37 -->


where $\sigma_x^2 = \frac{\mathrm{Tr}(\Sigma)}{d}$, $\Gamma_t = e^{-2t}\sigma_x^2 + \Delta_t = 1 + e^{-2t}(\sigma_x^2 - 1)$ and the expectation is over the $u, v, w$ random variables which are independent standard Gaussian $\mathcal{N}(0, 1)$.

Theorem 3.1. Let $q(z) = \frac{1}{p}\mathrm{Tr}(\mathbf{U} - z\mathbf{I}_p)^{-1}$, $r(z) = \frac{1}{p}\mathrm{Tr}(\Sigma^{1/2}\mathbf{W}^T(\mathbf{U} - z\mathbf{I}_p)^{-1}\mathbf{W}\Sigma^{1/2})$ and $s(z) = \frac{1}{p}\mathrm{Tr}(\mathbf{W}^T(\mathbf{U} - z\mathbf{I}_p)^{-1}\mathbf{W})$, with $z \in \mathbb{C}$. Let
$$\hat{s}(q) = b_t^2\psi_p + \frac{1}{q},\qquad (15)$$
$$\hat{r}(r,q) = \frac{\psi_p a_t^2 e^{-2t}}{1 + \frac{a_t^2 e^{-2t}\psi_p}{\psi_n}\,r + \frac{\psi_p v_t^2}{\psi_n}\,q}.\qquad (16)$$
Then $q(z)$, $r(z)$ and $s(z)$ satisfy the following set of three equations:
$$s = \int d\rho_{\Sigma}(\lambda)\,\frac{1}{\hat{s}(q) + \lambda\hat{r}(r,q)},\qquad (17)$$
$$r = \int d\rho_{\Sigma}(\lambda)\,\frac{\lambda}{\hat{s}(q) + \lambda\hat{r}(r,q)},\qquad (18)$$
$$\psi_p(s_t^2 - z) + \frac{\psi_p v_t^2}{1 + \frac{a_t^2 e^{-2t}\psi_p}{\psi_n}\,r + \frac{\psi_p v_t^2}{\psi_n}\,q} + \frac{1 - \psi_p}{q} - \frac{s}{q^2} = 0.\qquad (19)$$
The eigenvalue distribution of $\mathbf{U}$, $\rho(\lambda)$, can then be obtained using the Sokhotski–Plemelj inversion formula $\rho(\lambda) = \lim_{\varepsilon \to 0^+} \frac{1}{\pi}\,\mathrm{Im}\, q(\lambda + i\varepsilon)$.

We now focus on the asymptotic regime $\psi_p, \psi_n \gg 1$, typical for strongly over-parameterized models trained on large data sets. In this limit, the spectrum of $\mathbf{U}$ can be described analytically by the following Theorem 3.2.

Theorem 3.2 (Informal). Let $\rho$ denote the spectral density of $\mathbf{U}$.

Regime I (overparametrized): $\psi_p > \psi_n \gg 1$.
$$\rho(\lambda) = \left(1 - \frac{1 + \psi_n}{\psi_p}\right)\delta(\lambda - s_t^2) + \frac{\psi_n}{\psi_p}\rho_1(\lambda) + \frac{1}{\psi_p}\rho_2(\lambda).$$

Regime II (underparametrized): $\psi_n > \psi_p \gg 1$.
$$\rho(\lambda) = \left(1 - \frac{1}{\psi_p}\right)\rho_1(\lambda) + \frac{1}{\psi_p}\rho_2(\lambda).$$

where $\rho_1$ is an atomless measure with support
$$\left[s_t^2 + v_t^2\left(1 - \sqrt{\psi_p/\psi_n}\right)^2,\, s_t^2 + v_t^2\left(1 + \sqrt{\psi_p/\psi_n}\right)^2\right],$$
and $\rho_2$ coincides with the asymptotic eigenvalue bulk density of the population covariance $\tilde{\mathbf{U}} = \mathbb{E}_{\mathbf{X}}[\mathbf{U}]; \rho_2$ is independent of $\psi_n$ and its support is on the scale $\psi_p$. The eigenvectors associated with $\delta(\lambda - s_t^2)$ leave both training and test losses unchanged and are therefore irrelevant. In the limit $\psi_p \gg \psi_n$, the supports of $\rho_1$ and $\rho_2$ are respectively on the scales $\psi_p/\psi_n$ and $\psi_p$, i.e. they are well separated.

The proofs of both theorems are shown in SM (Sect. C). We recall that training timescales are directly related to eigenvalues $\lambda$ via the relation $\tau^{-1} = \frac{\psi_p}{\Delta_t \lambda_{\min}}$. Theorem 3.2 therefore demonstrates the emergence of the two training timescales $\tau_{\text{mem}}$ and $\tau_{\text{gen}}$ in the overparametrized regime of the RFNN model. They are respectively associated to the measures $\rho_1$ and $\rho_2$, which are well separated in regime I, for $\psi_p \gg \psi_n \gg 1$, as shown in Fig. 4.

Generalization: The timescale $\tau_{\text{gen}}$ on which the first relaxation takes place is associated to the formation of the generalization regime. It is related to the bulk $\rho_2$ and is of order $1/\Delta_t$. This regime only depends on the population covariance $\Sigma$ of the data and is independent of the specific realization

of the dataset. On this timescale, which is of order one, both the training $\mathcal{L}_{\text{train}}$ and test $\mathcal{L}_{\text{test}}$ losses decrease. The generalization loss $\mathcal{L}_{\text{gen}} = \mathcal{L}_{\text{test}} - \mathcal{L}_{\text{train}}$ is zero, and $\mathcal{E}_{\text{score}}$ tends to a value that we find to scale as $\mathcal{O}(\psi_n^{-\eta})$ with $\eta \simeq 0.59$ numerically (see Fig. 5).

Memorization: The timescale $\tau_{\text{mem}}$, on which the second stage of the dynamics takes place, is associated to overfitting and memorization. It is related to the bulk $\rho_1$, and scales as $\psi_p/(\Delta_t \lambda_{\min})$, where $\lambda_{\min}$ is the left edge of $\rho_1$. In the overparameterized regime $p \gg n$, $\tau_{\text{mem}}$ exhibits a more complex dependence on both $\psi_n$ and the noise level $\Delta_t$, and may scale sublinearly with $n$ under varying noise schedules, contrasting with experiments suggesting a simple linear scaling. On this timescale, the training loss decreases while the test loss increases, converging to their respective asymptotic values as computed in [12]. Fig. 5 indeed shows that all training and test curves separate, correspondingly the generalization loss $\mathcal{L}_{\text{gen}}$ increases, at a time that scales with $\psi_p/(\Delta_t \lambda_{\min})$, as shown in the inset.

As $n$ increases, the asymptotic $(\tau \to \infty)$ generalization loss $\mathcal{L}_{\text{gen}}$ decreases, indicating a reduced overfitting. For $n > n^*(p) = p$, although some overfitting remains (i.e., $\mathcal{L}_{\text{gen}} > 0$), the value of $\mathcal{L}_{\text{gen}}$ is sensibly reduced, and the model is no longer expressive enough to memorize the training data, as shown in [12]. This regime corresponds to the Architectural Regularization phase in Fig. 1. We show in Fig. 5 (panel C) how the generalization loss $\mathcal{L}_{\text{gen}}$ varies in the $(n, p)$ plane depending on the time $\tau$ at which training is stopped. In agreement with the above results, we find that the generalization–memorization transition line depends on $\tau$ and moves upward for larger values of $\tau$, similarly to the numerical results exposed in Fig. 3 and the illustration in Fig. 1.


## Conclusion


4 Conclusions

We have shown evidence that the training dynamics of neural network-based score functions exhibit characteristics suggestive of implicit regularization that can reduce memorization in some overparameterized diffusion models. Specifically, we have identified two well-separated timescales in the learning: $\tau_{\text{gen}}$, at which models begin to generate high-quality, novel samples, and $\tau_{\text{mem}}$, beyond which they start to memorize the training data. The gap between these timescales grows with the size of the training set, leading to a broad window where early-stopped models generate novel samples of high quality. We have demonstrated that this phenomenon happens in realistic settings, for controlled synthetic data, and in analytically tractable models. Although our analysis focuses on DMs, related score-learning mechanisms may exist in other score-based generative methods such as stochastic interpolants [3] or flow matching [28]; further work is needed to verify the generality of our findings.

Limitations and future works.

- While we derived our results under SGD optimization, most DMs are trained in practice with Adam [24]. In SM Sects. A.3 and D, we show that the two key timescales still arise using Adam, although with much fewer optimization steps. Studying how different optimizers shift these timescales would be valuable for practical usage.

- All experiments in Sect. 2 are conducted with unconditional DMs. We additionally verify in SM Sect. B, using a toy Gaussian mixture dataset and classifier-free guidance [18], that the same scaling of $\tau_{\text{mem}}$ with $n$ holds in the conditional settings. Understanding precisely how the absolute timescales $\tau_{\text{mem}}$ and $\tau_{\text{gen}}$ depend on the conditioning remains an open question.

- Our numerical experiments cover a range of $p$ between 1M and 16M. Exploring a wider range is essential to map the full $(n, p)$ phase diagram sketched in Fig. 1 and understand the precise effect of expressivity on dynamical regularization.

- Finally, our theoretical analysis rely on well-controlled data and score models that reproduce the core effects. Extending these analytical frameworks to richer data distributions (such as Gaussian mixtures or data from the hidden manifold model) and to structured architectures would be valuable to further characterize the implicit dynamical regularization of training score-functions. In particular investigating how heavy-tailed data distribution [2] affect the picture described here could be valuable.

- Although DMs trained on large and diverse datasets likely avoid the memorization regime we study here, some industrial models were shown to exhibit partial memorization [7, 44]. Our results provide practical guidelines (early-stopping, control the network capacity) to train DMs robustly and hence avoid memorization, which can be especially helpful in data-scarce domains (e.g., physical sciences).


## Appendix


# Acknowledgments and Disclosure of Funding

The authors thank Valentin de Bortoli for initial motivating discussions on memorization–generalization transitions. RU thanks Beatrice Achilli, Jérôme Garnier-Brun, Carlo Lucibello, and Enrico Ventura for insightful discussions. RU is grateful to Bocconi University for its hospitality during his stay, during which part of this work was conducted. This work was performed using HPC resources from GENCI-IDRIS (Grant 2025-AD011016319). GB acknowledges support from the French government under the management of the Agence Nationale de la Recherche (ANR), PR[AI]RIE-PSAI (ANR-23-IACL-0008). MM acknowledges the support of the PNRR-PE-AI FAIR project funded by the NextGeneration EU program.


## References


# References

[1] Achilli, B., Ventura, E., Silvestri, G., Pham, B., Raya, G., Krotov, D., Lucibello, C., and Ambrogioni, L. (2024). Losing dimensions: Geometric memorization in generative diffusion.

[2] Adomaityte, U., Defilippis, L., Loureiro, B., and Sicuro, G. (2024). High-dimensional robust regression under heavy-tailed data: asymptotics and universality. Journal of Statistical Mechanics: Theory and Experiment, 2024(11):114002.

[3] Albergo, M. S., Boffi, N. M., and Vanden-Eijnden, E. (2023). Stochastic interpolants: A unifying framework for flows and diffusions.

[4] Biroli, G., Bonnaire, T., de Bortoli, V., and Mézard, M. (2024). Dynamical regimes of diffusion models. Nature Communications, 15(9957). Open access.

[5] Bodin, A. P. M. (2024). Random Matrix Methods for High-Dimensional Machine Learning Models. PhD thesis, École Polytechnique Fédérale de Lausanne (EPFL), Lausanne, Switzerland.

[6] de Bortoli, V. (2022). Convergence of denoising diffusion models under the manifold hypothesis. Transactions on Machine Learning Research. Expert Certification.

[7] Carlini, N., Hayes, J., Nasr, M., Jagielski, M., Sehwag, V., Tramèr, F., Balle, B., Ippolito, D., and Wallace, E. (2023). Extracting training data from diffusion models. In Proceedings of the 32nd USENIX Conference on Security Symposium, SEC '23, USA. USENIX Association.

[8] Cui, H., Krzakala, F., Vanden-Eijnden, E., and Zdeborova, L. (2024). Analysis of learning a flow-based generative model from limited sample complexity. In The Twelfth International Conference on Learning Representations.

[9] Cui, H., Pehlevan, C., and Lu, Y. M. (2025). A precise asymptotic analysis of learning diffusion models: theory and insights.

[10] D'Ascoli, S., Refinetti, M., Biroli, G., and Krzakala, F. (2020). Double trouble in double descent: Bias and variance(s) in the lazy regime. In III, H. D. and Singh, A., editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 2280–2290. PMLR.

[11] Favero, A., Sclocchi, A., and Wyart, M. (2025). Bigger isn't always memorizing: Early stopping overparameterized diffusion models.

[12] Alias, J., Smith, T., and Doe, J. (2025). Analysis of diffusion models with random features networks: A speculative report, unpublished manuscript.

[13] Gerace, F., Loureiro, B., Krzakala, F., Mézard, M., and Zdeborova, L. (2020). Generalisation error in learning with random features and the hidden manifold model. In III, H. D. and Singh, A., editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 3452–3462. PMLR.

[14] Goldt, S., Loureiro, B., Reeves, G., Krzakala, F., Mézard, M., and Zdeborová, L. (2021). The Gaussian equivalence of generative models for learning with shallow neural networks.

[15] Gu, X., Du, C., Pang, T., Li, C., Lin, M., and Wang, Y. (2023). On memorization in diffusion models.

[16] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium.

[17] Ho, J., Jain, A., and Abbeel, P. (2020). Denoising diffusion probabilistic models.

[18] Ho, J. and Salimans, T. (2022). Classifier-free diffusion guidance.

[19] Hu, H. and Lu, Y. M. (2023). Universality laws for high-dimensional learning with random features. IEEE Transactions on Information Theory, 69(3):1932–1964.

[20] Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6:695–709.

[21] Kadkhodaie, Z., Guth, F., Simoncelli, E. P., and Mallat, S. (2024). Generalization in diffusion models arises from geometry-adaptive harmonic representations. In The Twelfth International Conference on Learning Representations.

[22] Kamb, M. and Ganguli, S. (2024). An analytic theory of creativity in convolutional diffusion models.

[23] Karras, T., Aittala, M., Aila, T., and Laine, S. (2022). Elucidating the design space of diffusion-based generative models.

[24] Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. In Bengio, Y. and LeCun, Y., editors, ICLR (Poster).

[25] Li, P., Li, Z., Zhang, H., and Bian, J. (2025). On the generalization properties of diffusion models.

[26] Li, S., Chen, S., and Li, Q. (2024a). A good score does not lead to a good generative model.

[27] Li, T., Biferale, L., Bonaccorso, F., et al. (2024b). Synthetic lagrangian turbulence by generative diffusion models. Nat Mach Intell, 6:393–403.

[28] Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. (2023). Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations.

[29] Liu, Y., Zhang, K., Li, Y., Yan, Z., Gao, C., Chen, R., Yuan, Z., Huang, Y., Sun, H., Gao, J., He, L., and Sun, L. (2024). Sora: A review on background, technology, limitations, and opportunities of large vision models.

[30] Liu, Z., Luo, P., Wang, X., and Tang, X. (2015). Deep learning face attributes in the wild. In Proceedings of the IEEE International Conference on Computer Vision (ICCV).

[31] Mei, S., Misiakiewicz, T., and Montanari, A. (2019). Mean-field theory of two-layer neural networks: dimension-free bounds and kernel limit. In Beygelzimer, A. and Hsu, D., editors, Proceedings of the Thirty-Second Conference on Learning Theory, volume 99 of Proceedings of Machine Learning Research, pages 2388–2464. PMLR.

[32] Mei, S. and Montanari, A. (2020). The generalization error of random features regression: Precise asymptotics and double descent.

[33] Mézard, M., Parisi, G., and Virasoro, M. A. (1987). Spin Glass Theory and Beyond: An Introduction to the Replica Method and Its Applications, volume 9 of World Scientific Lecture Notes in Physics. World Scientific Publishing Company, Singapore.

[34] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, volume 32, pages 8024–8035. Curran Associates, Inc.

[35] Price, I., Sánchez-González, A., Alet, F., et al. (2025). Probabilistic weather forecasting with machine learning. Nature, 637:84–90.
[36] Péché, S. (2019). A note on the Pennington–Worah distribution. Electronic Communications in Probability, 24:1–7.
[37] Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F., Bengio, Y., and Courville, A. (2019). On the spectral bias of neural networks. In International Conference on Machine Learning, pages 5301–5310. PMLR.
[38] Rahimi, A. and Recht, B. (2007). Random features for large-scale kernel machines. In Platt, J., Koller, D., Singer, Y., and Roweis, S., editors, Advances in Neural Information Processing Systems, volume 20. Curran Associates, Inc.
[39] Robbins, H. and Monro, S. (1951). A stochastic approximation method. The Annals of Mathematical Statistics, pages 400–407.
[40] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. (2021). High-Resolution Image Synthesis with Latent Diffusion Models.
[41] Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In Navab, N., Hornegger, J., Wells, W. M., and Frangi, A. F., editors, Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015, pages 234–241, Cham. Springer International Publishing.
[42] Shah, K., Kalavasis, A., Klivans, A. R., and Daras, G. (2025). Does Generation Require Memorization? Creative Diffusion Models Using Ambient Diffusion.
[43] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. (2015). Deep Unsupervised Learning Using Nonequilibrium Thermodynamics. In Bach, F. and Blei, D., editors, Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2256–2265, Lille, France. PMLR.
[44] Somepalli, G., Singla, V., Goldblum, M., Geiping, J., and Goldstein, T. (2023a). Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
[45] Somepalli, G., Singla, V., Goldblum, M., Geiping, J., and Goldstein, T. (2023b). Understanding and Mitigating Copying in Diffusion Models. Advances in Neural Information Processing Systems, 36:47783–47803.
[46] Song, J., Meng, C., and Ermon, S. (2022). Denoising Diffusion Implicit Models.
[47] Song, Y., Durkan, C., Murray, I., and Ermon, S. (2021a). Maximum Likelihood Training of Score-Based Diffusion Models.
[48] Song, Y. and Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. In Wallach, H., Larochelle, H., Beygelzimer, A., d’Alché-Buc, F., Fox, E., and Garnett, R., editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc.
[49] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. (2021b). Score-Based Generative Modeling through Stochastic Differential Equations.
[50] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gómez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention Is All You Need. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R., editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.
[51] Ventura, E., Achilli, B., Silvestri, G., Lucibello, C., and Ambrogioni, L. (2025). Manifolds, Random Matrices and Spectral Gaps: The Geometric Phases of Generative Diffusion.
[52] Vincent, P. (2011). A Connection between Score Matching and Denoising Autoencoders. Neural Computation, 23(7):1661–1674.

[53] Wang, B. (2025). An Analytical Theory of Power-Law Spectral Bias in the Learning Dynamics of Diffusion Models.

[54] Wang, P., Zhang, H., Zhang, Z., Chen, S., Ma, Y., and Qu, Q. (2024). Diffusion Models Learn Low-Dimensional Distributions via Subspace Clustering.

[55] Wu, Y.-H., Marion, P., Biau, G., and Boyer, C. (2025). Taking a Big Step: Large Learning Rates in Denoising Score Matching Prevent Memorization.

[56] Yoon, T., Choi, J. Y., Kwon, S., and Ryu, E. K. (2023). Diffusion Probabilistic Models Generalize When They Fail to Memorize. In ICML 2023 Workshop on Structured Probabilistic Inference & Generative Modeling.

[57] Zhang, C., Zhang, C., Zheng, S., Zhang, M., Qamar, M., Bae, S.-H., and Kweon, I. S. (2023). A Survey on Audio Diffusion Models: Text-to-Speech Synthesis and Enhancement in Generative AI.

[58] Xu, Z.-Q. J., Zhang, Y., Luo, T., Xiao, Y., and Ma, Z. (2020). Frequency Principle: Fourier Analysis Sheds Light on Deep Neural Networks. Communications in Computational Physics, 28(5):1746-1767.