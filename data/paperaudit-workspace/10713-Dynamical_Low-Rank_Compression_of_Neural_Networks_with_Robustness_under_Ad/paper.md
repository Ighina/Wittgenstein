
## Introduction


Dynamical Low-Rank Compression of Neural Networks with Robustness under Adversarial Attacks

Steffen Schotthöfer*, H. Lexie Yang†, and Stefan Schnake*

*Computer Science and Mathematics Division, †Geospatial Science and Human Security Division
Oak Ridge National Laboratory
Oak Ridge, TN 37831 USA
{schotthofers, yangh, schnakesr}@ornl.gov


## Abstract


Abstract

Deployment of neural networks on resource-constrained devices demands models that are both compact and robust to adversarial inputs. However, compression and adversarial robustness present conflicting trade-offs. In this work, we present a dynamical low-rank training scheme with a spectral regularizer to partially control the condition number of the low-rank core in each layer. This method reduces sensitivity of compressed models to adversarial perturbations though some degradation in clean accuracy may occur. The approach is model- and data-agnostic, efficient, and supports rank adaptivity to compress networks. Experiments on standard architectures and datasets show that networks can achieve compression rates approaching 90% while partially recovering adversarial accuracy relative to uncompressed baselines, depending on the dataset and attack strength.


## Introduction


1 Introduction

Deep neural networks have achieved state-of-the-art performance across a wide range of tasks in computer vision and data processing. However, their success comes at a cost of substantial computational and memory demands, which hinders deployment in resource-constrained environments. While significant progress has been made in scaling up models through data centers and specialized hardware, a complementary and equally important challenge lies in the opposite direction: deploying accurate and robust models on low-power platforms such as unmanned aerial vehicles (UAVs) or surveillance sensors. These platforms often operate in remote locations with limited power and compute resources, and are expected to function autonomously over extended periods without human intervention.

This setting introduces three interdependent challenges:
• Compression: Models must operate under strict memory, compute, and energy budgets.
• Accuracy: Despite being compressed, models must maintain high performance to support critical decision-making.
• Robustness: Inputs may be corrupted by noise or adversarial perturbations, requiring models to be resilient under distributional shifts.

Recent work indicates these three objectives are highly challenging to reconcile. Compression methods such as low-rank [38] or sparsity [14] generally reduce accuracy, while adversarial robustness techniques including data augmentation [24] and regularization [54] often impair clean performance. Additionally, low-rank compressed networks may become more vulnerable to attacks [35]. Many robustness methods introduce costly overheads during training [43, 8] or inference [9, 15, 28], complicating deployment on constrained devices.

Our main contributions are as follows:

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

• Low-rank compression framework. We introduce a novel regularization and integration method to modify a class of low-rank training methods that yields low-rank compressed neural networks, achieving a more than 10× reduction in both memory footprint and compute cost, while maintaining clean accuracy and adversarial robustness on par with full-rank baselines.

• Theoretical guarantees. We analyze the proposed regularizer and derive an explicit bound on the condition number κ of each regularized layer. The bound gives further confidence that the regularizer improves adversarial performance.

• Preservation of performance. We prove analytically—and verify empirically—that our regularizer neither degrades training performance nor reduces clean validation accuracy across a variety of network architectures.

• Extensive empirical validation. We conduct comprehensive experiments on multiple architectures and datasets, demonstrating the effectiveness, robustness, and broad applicability of our method.

Beyond these core contributions, our approach is model- and data-agnostic, can be integrated seamlessly with existing adversarial defenses, e.g., adversarial training [13], and never requires assembling full-rank weight matrices—the last point guaranteeing a low memory footprint during training and inference. Moreover, by connecting to dynamical low-rank integration schemes and enabling convergence analysis via gradient flow, we offer new theoretical and algorithmic insights. Finally, the use of interpretable spectral metrics enhances the trustworthiness and analyzability of the compressed models.


## Preliminaries


\n\n\n\n

## 2 Controlling the adversarial robustness of a neural network through the singular spectrum of its layers

We consider a neural network f as a concatenation of L layers z^{ℓ+1} = σ^ℓ(W^ℓ z^ℓ) with matrix-valued^1 parameters W^ℓ ∈ ℝ^{n×n}, layer input z^ℓ ∈ ℝ^{n×b}, and element-wise nonlinear activation σ^ℓ. For simplicity of notation, we do not consider biases, but they are included for the numerical experiments in Section 6. The data X constitutes the input to the first layer, i.e., z^0 = X. We assume that the layer activations σ^ℓ are Lipschitz continuous, which is the case for all popular activations [35]. The network is trained on a loss function ℒ which we assume to be locally bounded with a Lipschitz continuous gradient. Throughout this work, we call a network in the standard format a “baseline” network.

Low-rank Compression: The compression of the network for training and inference is typically facilitated by approximating the layer weight matrices by a low-rank factorization W^ℓ = U^ℓ S^ℓ V^{ℓ,⊤}, with U^ℓ, V^ℓ ∈ ℝ^{n×r} and S^ℓ ∈ ℝ^{r×r}, where r ≤ n is the rank of the factorization. In this work, we generally assume that U^ℓ and V^ℓ are orthonormal matrices at all times during training and inference. This assumption deviates from standard low-rank training approaches [17]; however, recent literature provides methods that are able to fulfill this assumption approximately [55] and even exactly [38, 37]. If r ≪ n, the low-rank factorization, with a storage and matrix–vector computation cost of 𝒪(2nr + r^2), is computationally more efficient than the standard matrix format W with a computational cost of 𝒪(n^2).

Adversarial robustness: The adversarial robustness of a neural network f, a widely used trustworthiness metric, can be measured by its relative sensitivity 𝒮 to small perturbations δ, e.g., noise, of the input data X [49, 11], i.e., 𝒮(f, X, δ) := (‖f(X+δ) − f(X)‖/‖f(X)‖)(‖X‖/‖δ‖). In this work, we consider the sensitivity in the Euclidean (ℓ^2) norm, i.e., ‖·‖ = ‖·‖_2. For neural networks consisting of layers with Lipschitz continuous activation functions σ^ℓ, 𝒮 can be bounded [35] by the product
𝒮(f, X, δ) ≤ (∏_{ℓ=1}^L κ(W^ℓ)) (∏_{ℓ=1}^L κ(σ^ℓ)) (1)
where κ(W) := ‖W‖‖W^†‖ is the condition number of a matrix W, W^† is the pseudo-inverse of W, and κ(σ) is the condition number of the layer activation function σ. The condition number of the element-wise nonlinear activation functions σ^ℓ can be computed with the standard definitions (see [45] and [35] for condition numbers of several popular activation functions). Equation (1) allows us to consider each layer individually; thus, we drop the superscript ℓ for brevity of exposition.

^1 We provide an extension to tensor-valued layers, e.g., in CNNs, in Section 5.1.

^2 Note that the difference between the baseline and low-rank singular spectrum may be less pronounced for other layers and architectures. However, we have observed in all test cases that regularization with 𝓡 makes the singular spectrum of the low-rank network more benign.

The sensitivity of a low-rank factorized network can be readily deduced from Equation (1) by leveraging orthonormality of U and V, i.e., κ(USV^⊤) = κ(S). Thus, we only consider the r × r coefficient matrix S to control the sensitivity of the network. The condition number κ(S) can be determined via a singular value decomposition (SVD) of S, which is computationally feasible when r ≪ n.

Adversarial robustness-aware low-rank training: Enhancing the adversarial robustness of the network during low-rank training thus boils down to controlling the conditioning of S, which is a non-trivial task. Moreover, the dynamics of the singular spectrum of S of adaptive low-rank training schemes such as Dynamical Low-Rank Training (DLRT) [38] become more ill-conditioned than the baseline during training, even if S is always full rank. In Figure 1, we observe that the singular values \varsigma of a rank 64 factorization of a network layer compressed with DLRT range from \varsigma_{r=1} = 2.7785 to \varsigma_{r=64} = 0.8210 yielding a condition number κ(S) = 3.3844. In comparison, the baseline network has singular values ranging from \varsigma_{r=1} = 1.8627 to \varsigma_{r=128} = 0.9445 yielding a lower condition number κ(S) = 1.9722. As a result, an ℓ^2-FGSM attack with strength ε = 0.3 reduces the accuracy of the baseline network to 54.96%, while the accuracy of the low-rank network drops to 43.39%, see Table 2.

Figure 1: The singular values \varsigma_i(W) of sequential layer 7 in VGG16 for baseline training, unregularized dynamical low-rank training, and RobustDLRT with our condition number regularizer \mathcal{R} with \beta = 0.075 (see Section 5). The matrix W is formed as the first-mode unfolding of the convolutional tensor. Conditioning of the regularized low-rank layer is significantly improved compared to the non-regularized low-rank and baseline layer.<sup>2</sup>


## Related Work


## 3 Related work

Low-rank compression reduces memory and compute costs by constraining weights to low-rank subspaces. Early work performed post-hoc matrix [12] and tensor decompositions [23]. Recent methods incorporate low-rank constraints within training for efficiency and generalization. Dynamical Low-Rank Training (DLRT) [38] formulates optimization on low-rank manifolds, enabling compression without storing full-rank matrices. Extensions to tensor layers exist but federated learning adaptations remain preliminary and unproven. Methods like Pufferfish [47] and intrinsic dimension approaches [2] leverage subspace training but differ significantly in goals and scope. Memory reductions via gradient projection, e.g., GaLore, are orthogonal to our approach.

In contrast, low-rank fine-tuning methods like low-rank adaptation (LoRA) [17] inject trainable low-rank updates into a frozen pre-trained model, enabling efficient adaptation with few parameters. Extensions such as GeoLoRA [37], AdaLoRA [55], DyLoRA [46], and DoRA [31] incorporate rank adaptation or structured updates, improving performance over static rank baselines. However, these fine-tuning methods do not reduce the cost of the full training and inference, thus are not applicable to address the need of promoting computational efficiency.

Pruning is another well studied approach to reduce the number of parameters of a trained neural network [18, 26, 40, 57, 7, 19] by either sparsifying weight matrices or layer output channels of a network. Typically sparsity pruning is performed after training a fully parametrized neural network and thus only reduces memory and compute load during inference, while treating training as an offline cost.

Improving adversarial robustness with orthogonal layers has been a recently studied topic in the literature [3, 4, 48, 10, 35]. Many of these methods can be classified as either a soft approach, where orthogonality is imposed weakly via a regularizer, or a hard approach, where orthogonality is explicitly enforced in training.

Examples of soft approaches include the soft orthogonal (SO) regularizer [48], double soft orthogonal regularizer [4], mutual coherence regularizer [4], and spectral normalization [32]. These regularization-based approaches have several advantages; namely, they are more flexible to many


![Figure 9](image_9.jpg)
<!-- Caption: Related Work, index 9 -->


problems/architectures and are amenable to transfer learning scenarios (since pretrained models are admissible in the optimization space). However, influencing the spectrum weakly via regularization cannot enforce rigorous and explicit bounds on the spectrum.

Many hard approaches strongly enforce orthogonality/well-conditioned constraints by training on a chosen manifold using Riemannian optimization methods [25, 1, 35]. A hard approach built for low-rank training is given in [35]; this method clamps the extremes of the spectrum to improve the condition number during training. The clamping gives a hard estimate on the range of the spectrum which enables a direct integration of the low-rank equations of motion with reasonable learning rates. However, this method requires a careful selection of the rank r, which is viewed as a hyperparameter in [35]. If r is chosen incorrectly, the clamping of the spectrum, a hard-thresholding technique, acts as a strong regularizer which could affect the validation metrics of the network.

Our regularization method detailed below falls neatly into a soft approach and our proposed regularizer can be seen as an extension of the soft orthogonality (SO) regularizer [48] to well-conditioned matrices in the low-rank setting. As noted in [4], the SO regularizer only works well when the input matrix is of size m × n with m ≤ n. However, we avoid this issue since the regularizer is applied to the square r × r matrix S; an extension to convolutional layers is discussed in Section 5.1. In the context of low-rank training, the soft approach enables rank-adaptivity of the method.


## Method


4 Improving conditioning via regularization

We design a computationally efficient regularizer R to control and decrease the condition number of each network layer during training. The regularizer R only acts on the small r × r coefficient matrices S of each layer and thus has a minimal memory and compute overhead over low-rank training. The regularizer is differentiable almost everywhere and compatible with automatic differentiation tools. Additionally, R has a closed-form derivative that enables an efficient and scalable implementation of ∇R. Furthermore, R is compatible with any rank-adaptive low-rank training scheme that ensures orthogonality of U, V, e.g., [55, 36, 37, 35].

Definition 1. We define the robustness regularizer R for any S ∈ R^{r×r} by
R(S) = ||S^T S − α_S^2 I||, where α_S^2 = (1/r) ||S||^2. (2)
and I = I_r is the r × r identity matrix.

The regularizer R can be viewed as an extension of the soft orthogonality regularizer [48, 4] where we penalize the distance of S^T S to the well-conditioned matrix α_S^2 I. Here α_S is chosen such that ||S|| = ||α_S I||. Moreover, R is also a scaled standard deviation of the squared singular values {σ_i(S)^2}_{i=1}^r, i.e.,
(1/r) R(S)^2 = (1/r) ∑_{i=1}^r (σ_i(S)^2)^2 − ( (1/r) ∑_{i=1}^r σ_i(S)^2 )^2. (3)
See Appendix C for the proof. Therefore, R is a unitarily invariant regularizer; namely, R(USV^T) = R(S) for orthogonal U, V. These two forms of R are useful in the properties shown below.

Proposition 1. The gradient of R in (2) is given by ∇R(S) = 2 S (S^T S − α_S^2 I) / R(S).
See Appendix C for the proof. The gradient computation consists only of r × r matrix multiplications and a Frobenius norm evaluation. Thus ∇R is computationally efficient for r ≪ m. Further, its closed-form enables a straightforward integration into existing optimizers such as Adam or SGD applied to S.

Proposition 2 (Condition number bound). For any S ∈ R^{r×r} there holds
κ(S) ≤ exp( (1 / (2 σ_r(S))) R(S) ) + c,
where c is a constant that may depend on r and the spectrum structure. Due to variations in σ_r(S) during training, the bound may not reliably predict the condition number throughout. Thus, R(S) provides a heuristic rather than a strict quantitative control on the conditioning.
See Appendix C for details.


## Experiments


Table 1: VGG16 on UCM data. Comparison of regularized LoRA and DLRT trained networks under the ℓ^2-FGSM attack. Orthogonality of U, V increases adversarial performance significantly.

| Method                      | c.r. [%] | clean Acc [%] | ℓ^2-FGSM, ε = 0.1 |
| --------------------------- | -------- | ------------- | ----------------- |
| Non-regularized DLRT        | 94.50    | 93.50         | 70.00             |
| RobustDLRT, β = 0.075       | 95.90    | 95.00         | 80.00             |
| LoRA, β = 0.075             | 95.70    | 87.50         | 75.00             |


## Method


In this section we integrate the regularizer R into a rank-adaptive, orthogonality preserving, and efficient low-rank training scheme. We seek a training framework that separates spectral dynamics of the coefficients S from bases U and V while maintaining their orthogonality during training. Popular simultaneous gradient descent methods like LoRA [17] do not preserve U and V orthogonality, violating R(USVᵀ) = R(S) and complicating regularizer evaluation. Our iterative truncation and basis augmentation steps, however, occasionally disrupt orthonormality, introducing numerical drift. This mismatch implies that some theoretical assumptions may not hold exactly in practice, necessitating further stabilization mechanisms.

See Appendix C for the proof. We note that unlike standard ridge and lasso regularization methods, R lacks convexity; thus long-time stability of the regularized dynamics is not obvious. However, ∇R possesses monotonicity properties that we leverage to show in (5) that the growth in J only depends on β, M, and the initial loss. Moreover, for large t, the change in the final loss by the regularizer only depends on β and the true solution M and not the specific path S(t). While training on the non-convex loss will not provide the same theoretical properties as the convex least-squares loss used in Proposition 3, the experiments in Figure 2 give confidence that adding our regularizer does not yield a relatively large change in the loss decay rate over moderate training regimes. Particularly, we observe empirically in Figure 2 that the condition number κ(S) decreases alongside the regularizer value R during training.

Singular value truncation used in rank-adaptive methods ensures that ζᵣ(S) is always sufficiently large. Figures 2a and 2b show the dynamics of R(S(t)) and κ(S(t)) during low-rank regularized training; we see that κ(S(t)) decays as R(S(t)) decays, validating Proposition 2.

We now study the stability of the regularizer when applied to a least squares regression problem, i.e., given a fixed M ∈ R^{r×r} we seek to minimize J(S) := β R(S) + 1/2 ||S − M||² over S ∈ R^{r×r}.

Proposition 3. Consider the dynamical system generated by the gradient flow of J; namely, Ṡ(t) + β∇R(S(t)) + S(t) = M. Then for any t > 0 we have the long-time stability estimate 1/2 ||S(t) − M||² + 2β ∫₀ᵗ e^{τ−t} R(S(τ)) dτ ≤ 1/2 e^{−t} ||S(0) − M||² + 2(1 − e^{−t}) β(1 + 2β) ||M||². (5)

Remark 1. When U, V are not orthonormal, e.g., in simultaneous gradient descent training (LoRA), the smallest n − r singular values of USVᵀ are often zero-valued; thus, the bound of Equation (4) is not useful. Table 1 shows that the clean accuracy and adversarial accuracy of regularized LoRA are significantly lower than standard or regularized training with orthonormal U, V.

Remark 2. We note R² can also be used in place of R. While R² is differentiable at R(S) = 0, we choose R as our regularizer due to the proper scaling in (4).


![Figure 14](image_14.jpg)
<!-- Caption: Method, index 14 -->



![Figure 15](image_15.jpg)
<!-- Caption: Method, index 15 -->


Thus we adapt the two-step scheme of [36] which ensures orthogonality of U, V. The method dynamically reduces or increases the rank of the factorized layers depending on the training dynamics and the complexity of the learning problem at hand. Consequently, the rank of each layer is no longer a hyper-parameter that needs fine-tuning, cf. [17, 35], but is rather an interpretable measure for the inherent complexity required for each layer.

To facilitate the discussion, we define Ĺ = ℒ + βℛ as the regularized loss function of the training process with regularization parameter β > 0. To construct the method we consider the (stochastic) gradient-descent-based update of a single weight matrix
W_{t+1} = W_t − λ ∇_W Ĺ(W_t)
for minimizing Ĺ with step size λ > 0. The corresponding continuous-time gradient flow reads Ẇ(t) = −∇_W Ĺ(W(t)), which is a high-dimensional dynamical system with a steady-state solution. We draw from established dynamical low-rank approximation (DLRA) methods, which were initially proposed for matrix-valued dynamical systems [20]. DLRA was recently extended to neural network training [38, 53, 36, 37, 22, 16] to formulate a consistent gradient flow evolution for the low-rank factors U, S, and V.

The DLRA method constrains the trajectory of W to the manifold ℳ_r, consisting of n × n matrices with rank r, by projecting the full dynamics Ẇ onto the local tangent space of ℳ_r via an orthogonal projection, see Figure 3. The low-rank matrix is represented as USV^⊤ ∈ ℳ_r, where U ∈ ℝ^{n×r} and V ∈ ℝ^{n×r} have orthonormal columns and S ∈ ℝ^{r×r} is full-rank (but not necessarily diagonal). An explicit representation of the tangent space leads to equations for the factors U, S, and V in [20, Proposition 2.1]. However, following these equations requires a prohibitively small learning rate due to the curvature of the manifold [29]. Therefore, specialized integrators have been developed to accurately navigate the manifold with reasonable learning rates [29, 6, 5].

Below we list the method of [36] with the changes introduced by adding our robustness regularizer. We call the resulting scheme RobustDLRT, and a single iteration of RobustDLRT is specified in Algorithm 1.

Basis Augmentation: The method first augments the current bases U^t, V^t at optimization step t by their gradient dynamics ∇_U ℒ, ∇_V ℒ via
Û = orth([U^t | ∇_U ℒ(U^t S^t V^{t,⊤})]) ∈ ℝ^{n×2r},
V̂ = orth([V^t | ∇_V ℒ(U^t S^t V^{t,⊤})]) ∈ ℝ^{n×2r}, (6)
to double the rank of the low-rank representation and subsequently creates orthonormal bases Û, V̂. Here orth(A) denotes an orthonormal basis for the range of A and | denotes horizontal concatenation of matrices. Since ℛ(USV^⊤) = ℛ(S), ∇_U ℛ(USV^⊤) = ∇_V ℛ(USV^⊤) = 0; hence ∇_U Ĺ = ∇_U ℒ and ∇_V Ĺ = ∇_V ℒ are used in (6). The span of Û contains U^t, which is needed to ensure the loss does not increase during augmentation, and provides a first-order approximation of span(Û^{t+1}) using the exact gradient flow for U, see [36, Theorem 2] for details. Geometrically, the latent space
𝒮 = { Û Z V̂^⊤ : Z ∈ ℝ^{2r×2r} } (7)
can be seen as a subspace^3 of the tangent plane of ℳ_r at U^t S^t V^{t,⊤}, see Figure 3.

Latent Space Training: We update the latent coefficients Ŝ via a Galerkin projection of the training dynamics onto the latent space 𝒮. The latent coefficients Ŝ are updated by integrating the projected gradient flow.

Figure 3: Geometric interpretation of Algorithm 1. First, we compute the parametrization of the tangent plane 𝒯_{ℳ_r}. Then we compute the projected gradient update with ∇_{Ŝ} Ĺ. Lastly, we retract the updated coefficients back onto the manifold ℳ_r. The regularizer ℛ steers training to regions of ℳ_r with lower curvature.

3 Technically the latent space contains extra elements not in the tangent space, but the extra information only helps the approximation.


![Figure 21](image_21.jpg)
<!-- Caption: Method, index 21 -->


Algorithm 1: Single iteration of RobustDLRT.

Input: Initial orthonormal bases U, V ∈ R^{n×r} and diagonal S ∈ R^{r×r}; ϑ: singular value threshold for rank truncation; λ: learning rate.

1. Evaluate L(USV^⊤) /* Forward evaluate */
2. G_U ← ∇_U L(USV^⊤); G_V ← ∇_V L(USV^⊤) /* Backprop on basis */
3. Û ← orth([U | G_U]); V̂ ← orth([V | G_V]) /* augmentation in parallel */
4. Ŝ ← Û^⊤ U S V^⊤ V̂ /* coefficient augmentation */
5. Ŝ ← coefficient_update(Ŝ, s_*, λ, β) /* regularized coefficient training */
6. U, S, V ← truncation(Ŝ, Û, V̂)
7. def coefficient_update(S_0: coefficient, s_*: number of local steps, λ: learning rate, β: robustness regularization weight):
8. for s = 1, …, s_* do
9. G_S ← −λ ∇_{Ŝ} L̃(Û Ŝ_{s−1} V̂^⊤) − β ∇_{Ŝ_s} R(Ŝ_s)
10. Ŝ_s ← Ŝ_{s−1} + optim(G_S) /* optimizer update, e.g., SGD or Adam */
11. return Ŝ_{s_*}
12. def truncation(Ŝ: augmented coefficient, Û: augmented basis, V̂: augmented co-basis):
13. P_{r_1}, Σ_{r_1}, Q_{r_1} ← truncated svd(Ŝ) with threshold ϑ to new rank r_1
14. U ← Û P_{r_1}; V ← V̂ Q_{r_1} /* Basis update */
15. S ← Σ_{r_1} /* Coefficient update with diagonal Σ_{r_1} */
16. return U, S, V

Ŝ̇ = −Û^⊤ ∇_W L̃ V̂ = −∇_{Ŝ} L̃ using stochastic gradient descent or another suitable optimizer for a number of s_* local iterations, i.e.,

Ŝ_{s+1} = Ŝ_s − λ ∇_{Ŝ} L̃ − β ∇_{Ŝ} R(Ŝ_s), s = 0, …, s_* − 1. (8)

Equation (8) is initialized with Ŝ_0 = Û^⊤ U^t S^t V^{t,⊤} V̂ ∈ R^{2r×2r}, and we set Ŝ = Ŝ_{s_*}.

Truncation: Finally, the latent solution Û Ŝ V̂^⊤ is retracted back onto the manifold M_r. The retraction can be computed efficiently by using a truncated SVD of Ŝ that discards the smallest r singular values. To enable rank adaptivity, the new rank r_1 instead of r can be chosen by a variety of criteria, e.g., a singular value threshold ∥[ϛ_{r_1}, …, ϛ_{2r}]∥_2 < ϑ. Once a suitable rank is determined, the bases U and V are updated by discarding the basis vectors corresponding to the truncated singular values.

Remark 3. We note that R will likely increase the smallest singular values of Ŝ to improve κ(Ŝ). This could theoretically increase the truncated rank over non-regularized DLRT and result in less compression. However, we find in the experiments in Section 6 that RobustDLRT has similar compression rates to DLRT.

Computational cost: The computational cost of RobustDLRT is asymptotically the same as LoRA, since the reconstruction of the full weight matrix W is never required. The orthonormalization, computation of the regularizer R, and the SVD for Ŝ account for O(n r^2), O(r^3), and O(r^3) floating point operations, respectively. When using multiple coefficient update steps s_* > 1, the amortized cost is lower than that of LoRA, since only the gradient with respect to Ŝ is required in most updates. While the regularizer may be applied to full-rank baseline models, its O(n^3) computational scaling significantly increases training costs.

5.1 Extension to convolutional neural networks

The convolution layer map in 2D CNNs translates a W × H image with N_I in-features to N_O out-features. Using tensors, this map is expressed as Y = C * X where X ∈ R^{N_I×W×H}, Y ∈ R^{N_O×W×H}, and C ∈ R^{N_O×N_I×S_W×S_H} is the convolutional kernel with a convolution window size S_W × S_H. Neglecting the treatment of strides and padding, C * X is given as a tensor contraction by

Y(o, w, h) = ∑_{c, s_w, s_h} C(o, c, s_w, s_h) X(c, w + s_w, h + s_h). (9)


## Experiments


Table 2: UCM and Cifar10 benchmark. Clean and adversarial accuracy means and std. devs. of the baseline and regularized low-rank networks for different architectures. We report the low-rank results for β = 0.0 (DLRT) and the best performing β that is given in Table 9. Algorithm 1 (RobustDLRT) is able to match or surpass baseline adversarial accuracy values at compression rates of up to 94% in most setups. All runs where RobustDLRT surpasses the uncompressed baseline are highlighted.

| UCM Data | Method      | c.r. [%] | Clean Acc. [%] | Acc. [%] for ℓ2-FGSM, ε 0.05 | Acc. [%] for ℓ2-FGSM, ε 0.1 | Acc. [%] for Jitter, ε 0.3 | Acc. [%] for Jitter, ε 0.035 | Acc. [%] for Mixup, ε 0.045 | Acc. [%] for Mixup, ε 0.025 | Acc. [%] for Mixup, ε 0.1 | Acc. [%] for Mixup, ε 0.75 |
| -------- | ----------- | -------- | --------------- | ---------------------------- | --------------------------- | -------------------------- | ----------------------------- | ---------------------------- | ---------------------------- | -------------------------- | --------------------------- |
| VGG16    | Baseline    | 0.0      | 94.40±0.72      | 86.71±1.90                   | 76.40±2.84                  | 54.96±2.99                 | 89.58±2.99                    | 85.05±3.40                   | 77.77±1.61                   | 37.25±3.66                 | 23.05±3.01                  |
|          | DLRT        | 95.30    | 93.92±0.23      | 87.95±1.02                   | 72.41±2.08                  | 43.39±4.88                 | 83.99±1.22                    | 67.41±1.63                   | 85.79±1.51                   | 40.42±2.89                 | 20.13±2.92                  |
|          | RobustDLRT  | 95.84    | **94.61±0.35**  | **89.12±1.33**               | **78.68±2.30**              | **53.30±3.14**             | **88.33±1.20**                | **79.81±0.93**               | **90.33±0.90**               | **70.12±3.08**             | **47.31±2.78**              |
| VGG11    | Baseline    | 0.0      | 94.23±0.71      | 89.93±1.33                   | 78.66±2.46                  | 39.45±2.98                 | 90.25±1.66                    | 85.24±1.90                   | 83.10±1.47                   | 40.34±4.88                 | 22.01±3.21                  |
|          | DLRT        | 94.89    | 93.70±0.71      | 86.58±1.22                   | 67.55±2.16                  | 28.92±2.65                 | 83.90±1.36                    | 63.41±1.39                   | 87.15±1.18                   | 40.17±4.96                 | 14.18±3.78                  |
|          | RobustDLRT  | 94.59    | 93.57±0.84      | 87.90±0.91                   | 72.96±1.55                  | 32.85±2.46                 | 86.77±0.76                    | 74.31±1.50                   | **88.00±1.13**               | **60.97±4.18**             | **28.56±3.64**              |
| ViT-16b  | Baseline    | 0.0      | 96.72±0.36      | 93.02±0.38                   | 92.18±0.31                  | 89.71±0.28                 | 93.71±1.22                    | 93.21±1.17                   | 89.62±1.81                   | 51.05±3.17                 | 43.91±3.97                  |
|          | DLRT        | 86.70    | 96.38±0.60      | 91.21±0.44                   | 82.10±0.32                  | 62.45±0.41                 | 86.67±1.05                    | 79.81±0.81                   | 80.48±1.82                   | 41.52±3.24                 | 35.91±3.76                  |
|          | RobustDLRT  | 87.90    | 96.41±0.67      | 92.57±0.34                   | 85.67±0.41                  | 69.94±0.42                 | 91.03±0.86                    | 84.19±1.39                   | 87.33±1.81                   | 46.39±2.75                 | 40.76±3.88                  |

| Cifar10 Data | Method     | c.r. [%] | Clean Acc. [%] | Acc. [%] for ℓ2-FGSM, ε 0.05 | Acc. [%] for ℓ2-FGSM, ε 0.1 | Acc. [%] for Jitter, ε 0.3 | Acc. [%] for Jitter, ε 0.035 | Acc. [%] for Mixup, ε 0.045 | Acc. [%] for Mixup, ε 0.025 | Acc. [%] for Mixup, ε 0.1 | Acc. [%] for Mixup, ε 0.75 |
| ------------- | ---------- | -------- | --------------- | ---------------------------- | --------------------------- | -------------------------- | ----------------------------- | ---------------------------- | ---------------------------- | -------------------------- | --------------------------- |
| VGG16         | Baseline   | 0.0      | 89.82±0.45      | 76.22±1.38                   | 63.78±2.01                  | 34.97±2.54                 | 78.60±1.12                    | 73.54±1.55                   | 71.51±1.31                   | 37.36±2.60                 | 16.12±2.12                  |
|               | DLRT       | 94.37    | 89.23±0.62      | 74.07±1.23                   | 59.55±1.79                  | 28.74±2.21                 | 72.51±1.04                    | 66.21±1.41                   | 79.56±1.15                   | 59.88±2.26                 | 38.98±1.94                  |
|               | RobustDLRT | 94.18    | 89.49±0.58      | 76.04±1.18                   | 62.08±1.69                  | 32.77±2.04                 | 75.53±0.98                    | 69.93±1.22                   | **87.62±1.07**               | **84.80±2.01**             | **81.26±2.15**              |
| VGG11         | Baseline   | 0.0      | 88.34±0.49      | 75.89±1.42                   | 64.21±1.96                  | 31.76±2.45                 | 74.96±1.09                    | 68.59±1.63                   | 74.77±1.26                   | 40.88±2.58                 | 8.95±1.98                   |
|               | DLRT       | 95.13    | 88.13±0.56      | 72.02±1.34                   | 55.83±1.92                  | 21.59±2.16                 | 66.98±1.05                    | 58.57±1.55                   | 79.42±1.08                   | 47.95±2.18                 | 22.92±1.77                  |
|               | RobustDLRT | 94.67    | 87.97±0.52      | **76.04±1.26**               | **63.82±1.83**              | 30.77±2.30                 | 71.06±1.00                    | 65.63±1.38                   | **84.93±1.10**               | **78.35±1.89**             | **65.93±2.04**              |
| ViT-16b       | Baseline   | 0.0      | 95.42±0.35      | 79.94±0.95                   | 63.66±1.62                  | 32.09±2.05                 | 84.65±0.88                    | 77.20±1.04                   | 52.17±1.49                   | 16.03±2.34                 | 13.29±2.01                  |
|               | DLRT       | 73.42    | 95.39±0.41      | 79.50±0.91                   | 61.62±1.48                  | 30.32±1.94                 | 83.33±0.80                    | 76.16±0.95                   | 58.32±1.44                   | 17.43±2.28                 | 14.49±1.92                  |
|               | RobustDLRT | 75.21    | 94.66±0.38      | **82.03±0.88**               | **69.29±1.43**              | **38.05±1.99**             | **87.97±0.75**                | **83.03±0.91**               | **74.49±1.32**               | **27.80±2.11**             | **18.34±1.87**              |


## Method


where s_w and s_h range from −S_W/2, …, S_W/2 and −S_H/2, …, S_H/2 respectively, and o = 1, …, N_O, w = 1, …, W, and h = 1, …, H.

DLRT was extended to convolutional layers in [53] by compressing C with a Tucker factorization. Little is gained in compressing the window modes as they are typically small. Thus, we only factorize C in the feature modes with output and input feature ranks r_O ≪ N_O and r_I ≪ N_I as
C(o, i, s_w, s_h) = ∑_{q_O=1}^{r_O} ∑_{q_I=1}^{r_I} U_O(o, q_O) U_I(i, q_I) S(q_O, q_I, s_w, s_h). (10)

Substituting (10) into (9) and rearranging indices yields
Y(o, w, h) = ∑_{q_O} U_O(o, q_O) Ẏ(q_O, w, h), (11a)
Ẏ(q_O, w, h) = ∑_{q_I, s_w, s_h} S(q_O, q_I, s_w, s_h) Ẋ(q_I, w + s_w, h + s_h), (11b)
Ẋ(q_I, w + s_w, h + s_h) = ∑_c U_I(c, q_I) X(c, w + s_w, h + s_h). (11c)

Remark 4. Aside from the prolongation (11a) and retraction (11c) from/to the low-rank latent space, the low-rank convolution map (11) features a convolution (11b) similar to (9) but in the reduced-dimension low-rank latent space.

Robustness regularization for convolutional layers. The contractions in (9) and (11b) show that the output channels arise from a tensor contraction of the input channel and window modes; hence, both (9) and (11b) can be viewed as matrix–vector multiplications where C is matricised on the output channel mode; i.e., C → Mat(C) ∈ ℝ^{N_O×N_I S_W S_H} and S → Mat(S) ∈ ℝ^{r_O×r_I S_W S_H}. Therefore, we only regularize Mat(S) with our robustness regularizer. Moreover, we assume r_O ≤ r_I S_W S_H, which is almost always the case since r_O and r_I are comparable and S_W S_H ≫ 1. Then we regularize convolutional layers by ℛ(Mat(S)ᵗ) so that S Sᵗ is an r_O × r_O matrix, which is computationally efficient.

We remark that the extension of Algorithm 1 to a tensor-valued layer with Tucker factorization only requires changing the truncation step; the SVD is replaced by a truncated Tucker decomposition of S. The Tucker bases U_O and U_I can be augmented in parallel similarly to the matrix case.


## Experiments


6 Numerical Results

We evaluate the numerical performance of Algorithm 1 compared with non-regularized low-rank training, baseline training, and several other robustness-enhancing methods on the VGG16, VGG11, and ViT-16b architectures and University of California, Merced (UCM), Cifar10, and ImageNet1k datasets. Detailed descriptions of the models, datasets, pre-processing, training hyperparameters,

Table 3: ImageNet benchmark, ViT-32l trained with baseline Adam, DLRT, and RobustDLRT. We report the low-rank results for unregularized β = 0.0 and the best performing β, given in Table 9. Algorithm 1 (RobustDLRT) is able to match or surpass baseline adversarial accuracy values in most setups. All runs where RobustDLRT surpasses the uncompressed baseline are highlighted.

| Method     | c.r. [%] | Top1/Top5 Clean Acc. [%] | Top1/Top5 Acc. [%] for ℓ²-FGSM, ϵ |                 |                 | Top1/Top5 Acc. [%] for Jitter, ϵ |                 |
| ---------- | -------- | ------------------------- | ---------------------------------- | --------------- | --------------- | --------------------------------- | --------------- |
|            |          |                           | 0.05                               | 0.1             | 0.3             | 0.035                             | 0.045           |
| Baseline   | 0        | 74.37/92.20               | 43.58/73.75                        | 31.42/63.42     | 16.03/43.41     | 43.09/78.24                       | 35.57/74.96     |
| DLRT       | 58.02    | 72.27/90.06               | 42.70/70.43                        | 30.32/60.90     | 15.47/40.58     | 43.98/74.49                       | 38.44/71.31     |
| RobustDLRT | 57.98    | 72.25/90.03               | 43.17/71.58                        | 35.11/62.82     | 25.24/50.65     | 48.22/77.35                       | 43.51/75.14     |

and competitor methods are given in Appendix B. A reference implementation is provided at https://github.com/ScSteffen/RobustDLRT. We measure the compression rate (c.r.) as the relative amount of pruned parameters of the target network, i.e. c.r. = (1 − #params low-rank net / #params baseline net) × 100. The reported numbers in the tables represent the average over 10 stochastic training runs. We observe in Table 2 that clean accuracy results exhibit a standard deviation of less than 0.8%; the standard deviation increases with the attack strength ϵ for all tests and methods. This observation holds true for all presented results; thus, we omit the error bars in the other tables for the sake of readability.

UCM dataset We observe in Table 2 that Algorithm 1 compresses VGG11, VGG16 and ViT-16b networks with up to 94% reduction in trainable parameters and slightly reduced clean accuracy compared to uncompressed baselines. The regularized low-rank networks often outperform non-regularized ones on adversarial accuracy under ℓ²-FGSM, Jitter, and Mixup attacks; for Mixup, improvements appear particularly large. However, we acknowledge scenarios where adversarial robustness may degrade or have limited effect, and note potential risks of overfitting to specific attacks. A comprehensive ethical analysis of fairness, robustness transferability, and societal impact is beyond this work’s scope but remains an important future direction.

Cifar10 dataset We repeat the methodology of the UCM dataset for Cifar10, and observe similar computational results in Table 2. Furthermore, we compare our method in Table 4 to several methods of the recent literature, see Section 3. We compare the adversarial accuracy under the ℓ¹-FGSM attack, see Appendix B.1.2 for details, for consistency with the literature results. We find that our proposed method achieves the highest adversarial validation accuracy for all attack strengths ϵ, even surpassing the baseline adversarial accuracy. Additionally, we find an at least 15% higher compression ratio with RobustDLRT than the second best compression method, CondLR [35]. A similar experiment for the Projected Gradient Descent (PGD) attack [30] is given in Appendix A.2.

Table 4: Comparison to literature on CIFAR10 with VGG16 under the ℓ¹-FGSM attack. The first three rows list the computed mean over 10 random initializations. The values of all other methods, given below the double rule, are taken from [35, Table 1]. RobustDLRT has higher adversarial accuracy at higher compression rates than all listed methods.

| Method               | c.r. [%] | 0.0   | ℓ¹-FGSM, ϵ |           |           |
| -------------------- | -------- | ----- | ---------- | --------- | --------- |
|                      |          |       | 0.002      | 0.004     | 0.006     |
| Baseline             | 0        | 89.83 | 78.61      | 64.66     | 53.71     |
| DLRT                 | 94.58    | 89.55 | 74.71      | 59.61     | 47.56     |
| RobustDLRT β = 0.15  | 94.35    | 89.35 | 78.72      | 66.02     | 54.15     |
|                      |          |       |            |           |           |
| Cayley SGD [25]      | 0        | 89.62 | 74.46      | 58.16     | 45.29     |
| Projected SGD [1]    | 0        | 89.70 | 74.55      | 58.32     | 45.74     |
| CondLR [35] τ = 0.5  | 50       | 89.97 | 72.25      | 60.19     | 50.17     |
| CondLR [35] τ = 0.5  | 80       | 89.33 | 68.23      | 48.54     | 36.66     |
| LoRA [17]            | 50       | 89.97 | 67.71      | 48.86     | 38.49     |
| LoRA [17]            | 80       | 88.10 | 64.24      | 42.66     | 29.90     |
| SVD prune [51]       | 50       | 89.92 | 67.30      | 47.77     | 36.98     |
| SVD prune [51]       | 80       | 87.99 | 63.57      | 42.06     | 29.27     |

ImageNet1k dataset Finally we repeat the methodology for the ImageNet1k dataset, using the ViT-32l vision transformer trained from an ImageNet21k checkpoint, and report the results in Table 3. The hyperparameters are obtained by

We perform an initial sweep and report the results in Tables 8 and 9. RobustDLRT consistently yields higher Top-1/Top-5 accuracy across ℓ^2-FGSM and Jitter attacks than DLRT, with especially pronounced gains at larger perturbations (e.g., +9 points in Top-1 accuracy under ℓ^2-FGSM ε = 0.3). These trends are consistent with our ViT experiments in Table 2, demonstrating that adversarial regularization enhances robustness without compromising scalability. We benchmark training runtime of one ImageNet epoch on an A100 80GB GPU. DLRT requires approximately 26m, while RobustDLRT takes about 33m, yielding a roughly 25% overhead attributable to regularizer computations and additional SVD steps. Variability in batch size and data loading may affect exact measurements. Further optimizations are needed to make RobustDLRT training costs comparable to DLRT.

Black-box attacks
We investigate the scenario where an attacker has knowledge of the used model architecture, but not of the low-rank compression. We use the ImageNet-1k pretrained VGG16 and VGG11 and retrain them with Algorithm 1 and baseline training on the UCM data using the same training hyperparameters. Then we generate adversarial examples with the baseline network and evaluate the performance on the low-rank network with and without regularization. The results are given in Table 5. In this scenario, the weights from low-rank training, being sufficiently far away from the baseline, provide an effective defense against the attack. Further, the proposed regularization significantly improves the adversarial robustness when compared to the unregularized low-rank network. Even for extreme attacks with ε = 1, the regularized network achieves 84.76% and 87.33% accuracy for VGG16 and VGG11, respectively.

Adversarial Training
We evaluate low-rank training for VGG16 on the UCM dataset with adversarial training where 50% of each batch is clean and 50% adversarially perturbed using ℓ^2-FGSM. However, baseline and DLRT models were trained on static, clean data without adversarial augmentation, while RobustDLRT was trained with adversarial batches. This difference confounds attribution of performance gains solely to the regularizer. Consequently, observed superior robustness in RobustDLRT may partly stem from increased training data diversity, challenging direct method-to-method comparisons.


## Conclusion


Conclusion
RobustDLRT enables highly compressed neural networks with strong adversarial robustness by controlling the spectral properties of low-rank factors. The method is efficient, rank-adaptive, and yields up to 94% parameter reduction across a diverse suite of models and datasets. The method achieves promising results in balancing compression, accuracy, and adversarial robustness, though improvements relative to existing approaches are incremental and dataset-dependent. While our approach shows potential to aid deployment on resource-constrained edge devices, further work is necessary to robustly validate impacts on energy efficiency, trustworthiness, and interpretability in real-world use cases. The condition number regularization offers interpretable spectral metrics but does not fully resolve transparency challenges in critical decision-making scenarios.


## Experiments


|       | Method         | c.r. [%] | ℓ^2-FGSM, ε 0.05 | ℓ^2-FGSM, ε 0.1 | ℓ^2-FGSM, ε 0.25 | ℓ^2-FGSM, ε 0.5 | ℓ^2-FGSM, ε 0.75 | ℓ^2-FGSM, ε 1.0 |
| ----- | -------------- | -------- | ---------------- | --------------- | ---------------- | --------------- | ---------------- | --------------- |
| VGG16 | Baseline       | 0.0      | 86.71            | 76.40           | 48.76            | 39.33           | 35.23            | 33.23           |
|       | β = 0          | 95.30    | 93.03            | 91.81           | 88.09            | 83.14           | 78.95            | 76.00           |
|       | β = 0.05       | 95.15    | 92.66            | 92.47           | 91.33            | 88.76           | 86.85            | 84.76           |
| VGG11 | Baseline       | 0.0      | 89.93            | 78.66           | 60.76            | 45.23           | 38.38            | 35.52           |
|       | β = 0          | 95.82    | 92.76            | 91.81           | 88.25            | 84.09           | 80.57            | 77.71           |
|       | β = 0.05       | 96.12    | 92.95            | 92.66           | 92.00            | 91.04           | 88.66            | 87.33           |

| Method    | c.r. [%] | ℓ^2-FGSM, ε 0.0 | ℓ^2-FGSM, ε 0.1 | ℓ^2-FGSM, ε 0.5 | ℓ^2-FGSM, ε 0.75 | ℓ^2-FGSM, ε 1.0 |
| --------- | -------- | ---------------- | --------------- | ---------------- | ----------------- | --------------- |
| Baseline  | 0.0      | 92.61            | 91.91           | 91.90            | 89.61             | 89.91           |
| β = 0     | 94.46    | 92.55            | 91.91           | 87.98            | 85.37             | 82.96           |
| β = 0.075 | 94.19    | 92.49            | 92.49           | 90.98            | 89.56             | 89.42           |

Table 6: UCM dataset — Adversarial Training. VGG16 is trained on 50% clean images and 50% images attacked with ℓ^2-FGSM for various ε. The displayed numbers are the mean of 5 repeated runs. RobustDLRT (β = 0.075) is superior to DLRT (β = 0) and is able to approximately match the non-compressed baseline.

Table 5: UCM dataset — Black-box attack. Adversarial images with the ℓ^2-FGSM attack are generated by the baseline network for different values of ε. The baseline, DLRT (β = 0), and RobustDLRT (β = 0.075) networks are then evaluated on these images. Regularized low-rank compression achieves high adversarial accuracy, even under strong attacks.


## Appendix


# Acknowledgments and Disclosure of Funding

This manuscript has been authored by UT-Battelle, LLC under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. The United States Government retains, and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan (http://energy.gov/downloads/doe-public-access-plan).

This material is based upon work supported by the Laboratory Directed Research and Development Program of Oak Ridge National Laboratory (ORNL), managed by UT-Battelle, LLC for the U.S. Department of Energy under Contract No. DE-AC05-00OR22725. The authors recognize that adversarial robustness research entails ethical responsibilities, including considerations of potential misuse, fairness, and reliability. Future work will include thorough assessment and mitigation of such concerns.

S. Schotthöfer, H. L. Yang, and S. Schnake were supported by the Artificial Intelligence Initiative of the Laboratory Directed Research and Development Program of Oak Ridge National Laboratory (ORNL), managed by UT-Battelle, LLC for the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.

This research used resources of the Compute and Data Environment for Science (CADES) at the Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.


## References


# References

[1] P.-A. Absil and J. Malick. Projection-like retractions on matrix manifolds. SIAM Journal on Optimization, 22(1):135–158, 2012.

[2] A. Aghajanyan, S. Gupta, and L. Zettlemoyer. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 7319–7328, 2021.

[3] C. Anil, J. Lucas, and R. Grosse. Sorting out Lipschitz function approximation. In International conference on machine learning, pages 291–301. PMLR, 2019.

[4] N. Bansal, X. Chen, and Z. Wang. Can we gain more from orthogonality regularizations in training deep networks? Advances in Neural Information Processing Systems, 31, 2018.

[5] G. Ceruti, J. Kusch, and C. Lubich. A rank-adaptive robust integrator for dynamical low-rank approximation. BIT Numerical Mathematics, pages 1–26, 2022.

[6] G. Ceruti and C. Lubich. An unconventional robust integrator for dynamical low-rank approximation. BIT Numerical Mathematics, 62(1):23–44, 2022.

[7] T. Chen, H. Zhang, Z. Zhang, S. Chang, S. Liu, P.-Y. Chen, and Z. Wang. Linearity grafting: Relaxed neuron pruning helps certifiable robustness, 2022.

[8] G. Cheng, X. Sun, K. Li, L. Guo, and J. Han. Perturbation-seeking generative adversarial networks: A defense framework for remote sensing image scene classification. IEEE Transactions on Geoscience and Remote Sensing, 60:1–11, 2022.

[9] M. Cisse, P. Bojanowski, E. Grave, Y. Dauphin, and N. Usunier. Parseval networks: Improving robustness to adversarial examples. In D. Precup and Y. W. Teh, editors, Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 854–863. PMLR, 06–11 Aug 2017.

[10] M. Cisse, P. Bojanowski, E. Grave, Y. Dauphin, and N. Usunier. Parseval networks: Improving robustness to adversarial examples. In International Conference on Learning Representations (ICLR), 2017.

[11] W. Czaja, N. Fendley, M. Pekala, C. Ratto, and I.-J. Wang. Adversarial examples in remote sensing. In Proceedings of the 26th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, SIGSPATIAL '18, page 408–411, New York, NY, USA, 2018. Association for Computing Machinery.

[12] E. L. Denton, W. Zaremba, J. Bruna, Y. LeCun, and R. Fergus. Exploiting linear structure within convolutional networks for efficient evaluation. *Advances in Neural Information Processing Systems*, 27, 2014.

[13] I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples. *arXiv preprint arXiv:1412.6572*, 2014.

[14] Y. Guo, A. Yao, and Y. Chen. Dynamic network surgery for efficient DNNs. *Advances in Neural Information Processing Systems*, 29, 2016.

[15] M. Hein and M. Andriushchenko. Formal guarantees on the robustness of a classifier against adversarial manipulation. *Advances in Neural Information Processing Systems*, 30, 2017.

[16] A. Hnatiuk, J. Kusch, L. Kusch, N. R. Gauger, and A. Walther. Stochastic aspects of dynamical low-rank approximation in the context of machine learning. *Optimization Online*, 2024.

[17] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*, 2021.

[18] T. Jian, Z. Wang, Y. Wang, J. Dy, and S. Ioannidis. Pruning adversarially robust neural networks without adversarial examples, 2022.

[19] A. Jordao and H. Pedrini. On the effect of pruning on adversarial robustness. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 1–11, 2021.

[20] O. Koch and C. Lubich. Dynamical low-rank approximation. *SIAM Journal on Matrix Analysis and Applications*, 29(2):434–454, 2007.

[21] A. Kurakin, I. J. Goodfellow, and S. Bengio. Adversarial machine learning at scale. In *International Conference on Learning Representations*, 2017.

[22] J. Kusch, S. Schotthöfer, and A. Walther. An augmented backward-corrected projector splitting integrator for dynamical low-rank training. *arXiv preprint arXiv:2502.03006*, 2025.

[23] V. Lebedev, Y. Ganin, M. Rakhuba, I. Oseledets, and V. Lempitsky. Speeding-up convolutional neural networks using fine-tuned CP-decomposition. In *International Conference on Learning Representations*, 2015.

[24] H. Lee, S. Han, and J. Lee. Generative adversarial trainer: Defense to adversarial perturbations with GAN. *arXiv preprint arXiv:1705.03387*, 2017.

[25] J. Li, F. Li, and S. Todorovic. Efficient Riemannian optimization on the Stiefel manifold via the Cayley transform. In *International Conference on Learning Representations*, 2020.

[26] Z. Li, T. Chen, L. Li, B. Li, and Z. Wang. Can pruning improve certified robustness of neural networks?, 2022.

[27] J. Lin, C. Song, K. He, L. Wang, and J. E. Hopcroft. Nesterov accelerated gradient and scale invariance for adversarial attacks. In *International Conference on Learning Representations*, 2020.

[28] X. Liu, Y. Li, C. Wu, and C.-J. Hsieh. Adv-BNN: Improved adversarial defense through robust Bayesian neural network. In *International Conference on Learning Representations*, 2019.

[29] C. Lubich and I. V. Oseledets. A projector-splitting integrator for dynamical low-rank approximation. *BIT Numerical Mathematics*, 54(1):171–188, 2014.

[30] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards deep learning models resistant to adversarial attacks. In *International Conference on Learning Representations*, 2018.

[31] Y. Mao, K. Huang, C. Guan, G. Bao, F. Mo, and J. Xu. DoRA: Enhancing parameter-efficient fine-tuning with dynamic rank distribution. In L.-W. Ku, A. Martins, and V. Srikumar, editors, *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 11662–11675, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.

[32] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida. Spectral normalization for generative adversarial networks. In International Conference on Learning Representations, 2018.
[33] J. Nagy. Über algebraische gleichungen mit lauter reellen wurzeln. Jahresbericht der Deutschen Mathematiker-Vereinigung, 27:37–43, 1918.
[34] R. Nenov, D. Haider, and P. Balazs. (Almost) smooth sailing: Towards numerical stability of neural networks through differentiable regularization of the condition number, 2024.
[35] D. Savostianova, E. Zangrando, G. Ceruti, and F. Tudisco. Robust low-rank training via approximate orthonormal constraints. Advances in Neural Information Processing Systems, 36:66064–66083, 2023.
[36] S. Schotthöfer and M. P. Laiu. Federated dynamical low-rank training with global loss convergence guarantees. arXiv preprint arXiv:2406.17887, 2024.
[37] S. Schotthöfer, E. Zangrando, G. Ceruti, F. Tudisco, and J. Kusch. GeoLoRA: Geometric integration for parameter efficient fine-tuning. In The Thirteenth International Conference on Learning Representations, 2025.
[38] S. Schotthöfer, E. Zangrando, K. Jonas, G. Ceruti, and F. Tudisco. Low-rank lottery tickets: finding efficient low-rank neural networks via matrix differential equations. In Advances in Neural Information Processing Systems, 2022.
[39] L. Schwinn, R. Raab, A. Nguyen, D. Zanca, and B. Eskofier. Exploring misclassifications of robust neural networks to enhance adversarial attacks. Applied Intelligence, 53(17):19843–19859, 2023.
[40] V. Sehwag, S. Wang, P. Mittal, and S. Jana. Hydra: Pruning adversarially robust neural networks. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 19655–19666. Curran Associates, Inc., 2020.
[41] R. Sharma, M. Gupta, and G. Kapoor. Some better bounds on the variance with applications. Journal of Mathematical Inequalities, 4(3):355–363, 2010.
[42] S. P. Singh, G. Bachmann, and T. Hofmann. Analytic insights into structure and rank of neural network Hessian maps. In Advances in Neural Information Processing Systems, volume 34, 2021.
[43] Y. Su, G. Zhang, S. Mei, J. Lian, Y. Wang, and S. Wan. Reconstruction-assisted and distance-optimized adversarial training: A defense framework for remote sensing scene classification. IEEE Transactions on Geoscience and Remote Sensing, 61:1–13, 2023.
[44] F. Tramèr, A. Kurakin, N. Papernot, I. Goodfellow, D. Boneh, and P. McDaniel. Ensemble adversarial training: Attacks and defenses. arXiv preprint arXiv:1705.07204, 2017.
[45] L. N. Trefethen and D. Bau. Numerical Linear Algebra. SIAM, Philadelphia, PA, 1997.
[46] M. Valipour, M. Rezagholizadeh, I. Kobyzev, and A. Ghodsi. DyLoRA: Parameter efficient tuning of pre-trained models using dynamic search-free low-rank adaptation. arXiv preprint arXiv:2210.07558, 2022.
[47] H. Wang, S. Agarwal, and D. Papailiopoulos. Pufferfish: Communication-efficient models at no extra cost. Proceedings of Machine Learning and Systems, 3:365–386, 2021.
[48] D. Xie, J. Xiong, and S. Pu. All you need is beyond a good init: Exploring better solution for training extremely deep convolutional neural networks with orthonormality and modulation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6176–6185, 2017.
[49] Y. Xu and P. Ghamisi. Universal adversarial examples in remote sensing: Methodology and benchmark. IEEE Transactions on Geoscience and Remote Sensing, 60:1–15, 2022.

[50] Y. Xu and P. Ghamisi. Universal adversarial examples in remote sensing: Methodology and benchmark. IEEE Trans. Geosci. Remote Sens., 60:1–15, 2022.

[51] H. Yang, M. Tang, W. Wen, F. Yan, D. Hu, A. Li, H. Li, and Y. Chen. Learning low-rank deep neural networks via singular vector orthogonality regularization and singular value sparsification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 678–679, 2020.

[52] Y. Yang and S. Newsam. Bag-of-visual-words and spatial extensions for land-use classification. In Proceedings of the 18th SIGSPATIAL International Conference on Advances in Geographic Information Systems, GIS '10, pages 270–279, New York, NY, USA, 2010. Association for Computing Machinery.

[53] E. Zangrando, S. Schotthöfer, G. Ceruti, J. Kusch, and F. Tudisco. Rank-adaptive spectral pruning of convolutional layers during training. In Advances in Neural Information Processing Systems, 2024.

[54] H. Zhang, Y. Yu, J. Jiao, E. Xing, L. El Ghaoui, and M. Jordan. Theoretically principled trade-off between robustness and accuracy. In International Conference on Machine Learning, pages 7472–7482. PMLR, 2019.

[55] Q. Zhang, M. Chen, A. Bukharin, P. He, Y. Cheng, W. Chen, and T. Zhao. AdaLoRA: Adaptive budget allocation for parameter-efficient fine-tuning. In The Eleventh International Conference on Learning Representations, 2023.

[56] J. Zhao, Z. Zhang, B. Chen, Z. Wang, A. Anandkumar, and Y. Tian. GaLore: Memory-efficient LLM training by gradient low-rank projection. In International Conference on Machine Learning, pages 61121–61143. PMLR, 2024.

[57] Q. Zhao, T. Königl, and C. Wressnegger. Non-uniform adversarially robust pruning. In I. Guyon, M. Lindauer, M. van der Schaar, F. Hutter, and R. Garnett, editors, Proceedings of the First International Conference on Automated Machine Learning, volume 188 of Proceedings of Machine Learning Research, pages 1–16. PMLR, 25–27 Jul 2022.


## Appendix


# A Additional Numerical Results

## A.1 UCM Dataset

The numerical results for the whitebox $\ell^2$-FGSM, Jitter, and Mixup adversarial attacks on the VGG16 and VGG11 architectures can be found in Figure 4, Figure 5, and Figure 6. The regularizer confidently increases the adversarial validation accuracy of the networks.

In Table 10, we observe that the regularizer $\mathcal{R}(W)$ applied to the full weight matrices (and flattened tensors) $W$ in baseline format is able to increase the adversarial robustness of the baseline network in the UCM/VGG16 test case. However, the increased adversarial robustness comes at the expense of some of the clean validation accuracy.

## A.2 CIFAR10 Dataset

We run the same experiment in Table 4 but with the $\ell^2$-PGD attack, which is an iterative version of $\ell^2$-FGSM with a random perturbation of the input image as the initial condition [30]. Overall, we see that RobustDLRT is competitive with the other robustness-improving methods when the compression rate is taken into account.

Table 7: Comparison to literature on CIFAR10 with VGG16 under the $\ell^2$-PGD attack. The first three rows list the computed mean over 10 random initializations. The values of all other methods, given below the double rule, are taken from [35, Table 5]. RobustDLRT has competitive adversarial accuracy to all methods with a compression rate $\geq$ 80%.

| Method                    | c.r. [%] | $\ell^2$-PGD, $\epsilon$<br/>0.0 | $\ell^2$-PGD, $\epsilon$<br/>0.1 | $\ell^2$-PGD, $\epsilon$<br/>0.13 | $\ell^2$-PGD, $\epsilon$<br/>0.16 | $\ell^2$-PGD, $\epsilon$<br/>0.2 | $\ell^2$-PGD, $\epsilon$<br/>0.23 | $\ell^2$-PGD, $\epsilon$<br/>0.26 | $\ell^2$-PGD, $\epsilon$<br/>0.3 |
| ------------------------- | --------- | -------------------------------- | -------------------------------- | --------------------------------- | --------------------------------- | -------------------------------- | --------------------------------- | --------------------------------- | -------------------------------- |
| RobustDLRT $\beta = 0.15$ | 94.18     | 88.80                            | 62.58                            | 53.47                             | 44.95                             | 34.75                            | 28.33                             | 22.64                             | 16.59                            |
| DLRT                      | 94.53     | 88.58                            | 59.34                            | 50.06                             | 41.50                             | 31.82                            | 25.67                             | 20.48                             | 15.04                            |
| Baseline                  | 0         | 90.48                            | 63.01                            | 54.66                             | 47.87                             | 40.77                            | 36.75                             | 33.51                             | 29.93                            |
| ***                       |           |                                  |                                  |                                   |                                   |                                  |                                   |                                   |                                  |
| Cayley SGD [25]          | 0         | 89.62                            | 67.68                            | 59.38                             | 51.09                             | 40.87                            | 34.46                             | 29.21                             | 23.62                            |
| Projected SGD [1]        | 0         | 89.70                            | 67.64                            | 59.25                             | 51.06                             | 40.86                            | 34.51                             | 29.19                             | 23.64                            |
| CondLR [35] $\tau = 0.1$ | 50        | 90.93                            | 67.03                            | 62.08                             | 59.15                             | 56.92                            | 55.96                             | 55.28                             | 54.58                            |
| CondLR [35] $\tau = 0.5$ | 50        | 89.97                            | 64.84                            | 60.25                             | 57.75                             | 56.03                            | 55.21                             | 54.75                             | 54.25                            |
| CondLR [35] $\tau = 0.1$ | 80        | 90.48                            | 61.00                            | 50.84                             | 42.19                             | 33.70                            | 29.44                             | 26.55                             | 23.97                            |
| CondLR [35] $\tau = 0.5$ | 80        | 89.33                            | 57.45                            | 46.35                             | 37.20                             | 28.30                            | 23.82                             | 20.65                             | 17.84                            |
| LoRA [17]                | 50        | 89.97                            | 55.74                            | 45.11                             | 36.86                             | 29.62                            | 26.28                             | 24.02                             | 21.84                            |
| LoRA [17]                | 80        | 88.10                            | 51.40                            | 39.70                             | 30.12                             | 20.97                            | 16.29                             | 13.15                             | 10.37                            |
| SVD prune [51]           | 50        | 89.92                            | 54.87                            | 43.85                             | 35.23                             | 27.95                            | 24.38                             | 22.06                             | 19.94                            |
| SVD prune [51]           | 80        | 87.99                            | 50.64                            | 39.06                             | 29.57                             | 20.16                            | 15.49                             | 12.22                             | 9.57                             |

# B Details to the numerical experiments of this work

## B.1 Recap of adversarial attacks

In the following we provide the definitions of the used adversarial attacks. We use the implementation of [50] for the $\ell^2$-FGSM, Jitter, and Mixup attack. For the $\ell^1$-FGSM attack, we use the implementation of https://github.com/COMPiLELab/CondLR.

### B.1.1 $\ell^2$-FGSM attack

The Fast Gradient Sign Method (FGSM) [21] is a single-step adversarial attack that perturbs an input in the direction of the gradient of the loss with respect to the input. Given a neural network classifier $f_\theta$ with parameters $\theta$, an input $x$, and its corresponding label $y$, the attack optimizes the cross-entropy loss $\mathcal{L}_{\text{CE}}(f_\theta(x), y)$ by modifying $x$ along the gradient's sign. The adversarial example is computed

as:

$$x' = x + \alpha \cdot \frac{\nabla_x \mathcal{L}_{\text{CE}}(f_\theta(x), y)}{\|\nabla_x \mathcal{L}_{\text{CE}}(f_\theta(x), y)\|_\infty},$$
(12)

where $\alpha$ controls the perturbation magnitude. To ensure the perturbation remains bounded, the difference $x' - x$ is clamped by an $\epsilon$ bound, i.e.,

$$x' = x + \max(-\epsilon, \min(x' - x, \epsilon)).$$
(13)

In this work we fix $\alpha = \epsilon$. The attack can be iterated to increase its strength.

## B.1.2 $\ell^1$-FGSM attack

The $\ell^1$-FGSM attack [44] is used in the reference work of [35] and uses the same workflow as (B.1.1), where (12) is changed to

$$x' = x + \alpha \cdot \frac{\text{sign}(\nabla_x \mathcal{L}_{\text{CE}}(f_\theta(x), y))}{\Sigma},$$
(14)

where $\Sigma$ denotes the standard deviation of the data points in the training dataset and the sign of the gradient matrix is taken element-wise.

## B.1.3 Jitter attack

The Jitter attack [39] is an adversarial attack that perturbs an input by modifying the softmax-normalized output of the model with random noise before computing the loss. Given a neural network classifier $f_\theta$ with parameters $\theta$, an input $x$, and its corresponding label $y$, the attack first computes the network output $z = f_\theta(x)$ and normalizes it using the $\ell^\infty$ norm:

$$\hat{z} = \text{Softmax}\left(\frac{s \cdot z}{\|z\|_\infty}\right),$$
(15)

where $s$ is a scaling factor. A random noise term $\eta \sim \mathcal{N}(0, \sigma^2)$ is added to $\hat{z}$, i.e.,

$$\tilde{z} = \hat{z} + \sigma \cdot \eta.$$
(16)

The attack loss function is a mean squared error between perturbed output and target, given by

$$\mathcal{L} = \|\tilde{z} - y\|_2^2.$$
(17)

The adversarial example is then computed using the gradient of $\mathcal{L}$ with respect to $x$:

$$x' = x + \alpha \cdot \frac{\nabla_x \mathcal{L}}{\|\nabla_x \mathcal{L}\|_\infty}.$$
(18)

To ensure the perturbation remains bounded, the modification $x' - x$ is clamped within an $\epsilon$ bound:

$$x' = x + \max(-\epsilon, \min(x' - x, \epsilon)).$$
(19)

In this work, we fix $\alpha = \epsilon$ and set $\sigma = 0.1$. The Jitter attack can be performed iteratively. For each iteration $k > 0$, the attack loss is normalized by the perturbation of the input image,

$$\mathcal{L} = \frac{\|\tilde{z} - y\|_2^2}{\|x - x'_k\|_\infty}, \quad k > 0.$$
(20)

In this work, we use 5 iterations of the Jitter attack for each image.

## B.1.4 Mixup attack

The Mixup attack [49] is an adversarial attack that generates adversarial samples that share similar feature representations with a given virtual example. Inspired by the Mixup data augmentation technique, this attack aims to create adversarial examples that maintain characteristics of both the original sample and its adversarial counterpart. Given a neural network classifier $f_\theta$ with parameters $\theta$, an input $x$, and its corresponding label $y$, the attack first computes a linear combination of cross-entropy and negative KL-divergence loss,

$$\mathcal{L}_{\text{mixup}} = \beta \sum_{k=1}^{5} \mathcal{L}_{\text{CE}}\left(f_\theta\left(\frac{x}{2^k}\right), y\right) - \mathcal{L}_{\text{KL}}.$$
(21)

Table 8: Training hyperparameters for the UCM, CIFAR-10, and ImageNet benchmarks. The first set of hyperparameters applies to both DLRT and baseline training, and we train DLRT with the same hyperparameters as the full-rank baseline models. The second set of hyperparameters is specific to DLRT. The DLRT hyperparameters are selected by an initial parameter sweep. We choose the DLRT truncation tolerance relative to the Frobenius norm of S, i.e., ϑ = τ‖Ṡ‖F, as suggested in [38].

| Hyperparameter                        | VGG16 | VGG11 | ViT16b | ViT32l |
| ------------------------------------- | ----- | ----- | ------ | ------ |
| Batch Size (UCM)                      | 16    | 16    | 16     | n.a.   |
| Batch Size (CIFAR-10)                 | 128   | 128   | 128    | n.a.   |
| Batch Size (ImageNet)                 | n.a.  | n.a.  | n.a.   | 256    |
| Learning Rate                         | 0.001 | 0.001 | 0.001  | 0.001  |
| Number of Epochs                      | 20    | 20    | 5      | 10     |
| L2 regularization                     | 0     | 0     | 0.001  | 0.0001 |
| Optimizer                             | AdamW | AdamW | AdamW  | AdamW  |
| DLRT rel. truncation tolerance τ      | 0.1   | 0.05  | 0.08   | 0.013  |
| Coefficient Steps s_*                 | 10    | 10    | 10     | 75     |
| Initial Rank                          | 150   | 150   | 150    | 200    |
| Parameters                            | 138M  | 132M  | 86M    | 304M   |

δ = α · ∇x LCE(fθ(x), y) / ‖∇x LCE(fθ(x), y)‖∞. (22)

Equation (21) features a scale-invariance attack applied to the loss [27, Section 3.3].

The final adversarial example is computed as a convex combination of the original input and its perturbed version:

x′ = λx + (1 − λ)(x + δ), (23)

where λ ∼ Beta(β, β) is sampled from a Beta distribution with hyperparameter β, controlling the interpolation between clean and perturbed inputs. The perturbation is further constrained within an ε-ball to ensure bounded adversarial modifications:

x′ = x + max(−ε, min(x′ − x, ε)). (24)

In this work, we fix α = 1 and set β = 10^−3. The attack can be iterated to increase its effectiveness, refining the adversarial perturbation at each step. We use 5 iterations of the Mixup Attack for each image.

B.2 Network architecture and training details

In this paper, we use the PyTorch implementation and take pretrained weights from the ImageNet-1k dataset as initialization. The data loader randomly samples a batch for each batch update, which is the only source of randomness in our training setup. Below is an overview of the used network architectures:

- VGG16 is a deep convolutional neural network architecture that consists of 16 layers, including 13 convolutional layers and 3 fully connected layers.
- VGG11 is a convolutional neural network architecture similar to VGG16 but with fewer layers, consisting of 11 layers: 8 convolutional layers and 3 fully connected layers. It follows the same design principle as VGG16, using small 3×3 convolution filters and 2×2 max-pooling layers.
- ViT16b is a Vision Transformer with 16×16 patch size, a deep learning architecture that leverages transformer models for image classification tasks.
- ViT32l is a Vision Transformer with 32×32 patch size, a deep learning architecture that leverages transformer models for image classification tasks. We use the ImageNet-21k weights from the Hugging Face endpoint google/vit-large-patch32-224-in21k as weight initialization.

The full training setup is described in Table 8. We train DLRT with the same hyperparameters as the full-rank baseline models. It is known [37] that DLRT methods are robust w.r.t. common hyperparameters.

| Architecture | UCM Dataset<br/>FGSM | UCM Dataset<br/>Jitter | UCM Dataset<br/>Mixup | CIFAR-10 Dataset<br/>FGSM | CIFAR-10 Dataset<br/>Jitter | CIFAR-10 Dataset<br/>Mixup | ImageNet Dataset<br/>FGSM | ImageNet Dataset<br/>Jitter | ImageNet Dataset<br/>Mixup |
| ------------ | -------------------- | ---------------------- | --------------------- | ------------------------ | -------------------------- | ------------------------- | ------------------------- | --------------------------- | -------------------------- |
| VGG16        | 0.075                | 0.2                    | 0.15                  | 0.05                     | 0.05                       | 0.05                      | n.a.                      | n.a.                        | n.a.                       |
| VGG11        | 0.1                  | 0.05                   | 0.15                  | 0.15                     | 0.05                       | 0.2                       | n.a.                      | n.a.                        | n.a.                       |
| ViT16b       | 0.1                  | 0.15                   | 0.15                  | 0.01                     | 0.01                       | 0.05                      | n.a.                      | n.a.                        | n.a.                       |
| ViT32l       | n.a.                 | n.a.                   | n.a.                  | n.a.                     | n.a.                       | n.a.                      | 0.075                     | 0.075                       | 0.075                      |

| β      | Acc [%] under the ℓ²-FGSM attack with ε<br/>0 | Acc [%] under the ℓ²-FGSM attack with ε<br/>0.01 | Acc [%] under the ℓ²-FGSM attack with ε<br/>0.025 | Acc [%] under the ℓ²-FGSM attack with ε<br/>0.05 | Acc [%] under the ℓ²-FGSM attack with ε<br/>0.075 | Acc [%] under the ℓ²-FGSM attack with ε<br/>0.1 | Acc [%] under the ℓ²-FGSM attack with ε<br/>0.2 | Acc [%] under the ℓ²-FGSM attack with ε<br/>0.3 | Acc [%] under the ℓ²-FGSM attack with ε<br/>0.4 | Acc [%] under the ℓ²-FGSM attack with ε<br/>0.5 |
| ------ | --------------------------------------------- | ------------------------------------------------ | ------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- |
| 0      | 92.40                                         | 91.72                                            | 90.65                                             | 86.71                                            | 81.32                                             | 76.40                                           | 64.52                                           | 54.96                                           | 49.38                                           | 45.14                                           |
| 0.0001 | 91.69                                         | 91.69                                            | 91.10                                             | 87.73                                            | 83.14                                             | 78.43                                           | 63.21                                           | 53.31                                           | 47.18                                           | 42.99                                           |
| 0.001  | 88.81                                         | 88.78                                            | 87.90                                             | 84.40                                            | 80.00                                             | 76.34                                           | 62.61                                           | 53.77                                           | 48.09                                           | 44.38                                           |
| 0.01   | 88.22                                         | 88.19                                            | 87.12                                             | 82.78                                            | 77.52                                             | 72.72                                           | 58.32                                           | 48.89                                           | 42.83                                           | 38.61                                           |
| 0.05   | 90.45                                         | 90.43                                            | 89.63                                             | 87.23                                            | 84.11                                             | 80.55                                           | 68.66                                           | 59.29                                           | 52.62                                           | 46.61                                           |
| 0.1    | 92.51                                         | 92.51                                            | 92.11                                             | 90.45                                            | 88.43                                             | 86.32                                           | 76.91                                           | 68.01                                           | 61.29                                           | 55.52                                           |
| 0.2    | 89.20                                         | 89.18                                            | 88.85                                             | 86.66                                            | 84.36                                             | 81.96                                           | 73.25                                           | 65.20                                           | 58.61                                           | 53.29                                           |

hyperparameters such as learning rate, batch size, and initial rank. The truncation tolerance τ is chosen between 0.05 and 0.1 per an initial parameter study. These values are good default values, as per recent literature [36, 42]. In general, there is a trade-off between target compression ratio and accuracy, as illustrated e.g., in [38] for matrix-valued and [42] for tensor-valued (CNN) layers.

## B.3 UCM Test Case

The University of California, Merced (UCM) Land Use Dataset is a benchmark dataset in remote sensing and computer vision, introduced in [52]. It comprises 2,100 high-resolution aerial RGB images, each measuring 256 × 256 pixels, categorized into 21 land use classes with 100 images per class. The images were manually extracted from the USGS National Map Urban Area Imagery collection, covering various urban areas across the United States. The dataset contains images with spatial resolution approximately 0.3 meters per pixel (equivalent to 1 foot), providing detailed visual information suitable for fine-grained scene classification tasks.

We normalize the training and validation data with mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225] for the RGB image channels. The convolutional neural networks used in this work are applied to the original 256 × 256 image size. The vision transformer data pipeline resizes the image to a resolution of 224 × 224 pixels. The adversarial attacks for this dataset are performed on the resized images.

## B.4 CIFAR-10

The CIFAR-10 dataset consists of 10 classes, with a total of 60,000 RGB images with a resolution of 32 × 32 pixels.

We use standard data augmentation techniques. That is, for CIFAR-10, we augment the training dataset by a random horizontal flip of the image, followed by a normalization using mean [0.4914, 0.4822, 0.4465] and standard deviation [0.2470, 0.2435, 0.2616]. The test dataset is only normalized. The convolutional neural networks used in this work are applied to the original 32 × 32 image size. The vision transformer data pipeline resizes the image to a resolution of 224 × 224 pixels. The adversarial attacks for this dataset are performed on the resized images.

| Regularization (β) | 0.0   | 0.01  | 0.025 | 0.05  | 0.075 | 0.1   | 0.2   | 0.3   | 0.4   | 0.5   | c.r. [%] |
| ------------------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | --------- |
| Full-Rank          | 92.40 | 91.72 | 90.65 | 86.71 | 81.32 | 76.40 | 64.52 | 54.96 | 49.38 | 45.14 | 0.00      |
| 0.0                | 93.92 | 93.86 | 93.00 | 87.95 | 80.14 | 72.41 | 52.12 | 43.39 | 39.07 | 35.81 | 95.30     |
| 0.001              | 94.03 | 93.94 | 93.04 | 88.70 | 81.51 | 74.10 | 53.70 | 43.94 | 38.04 | 34.26 | 95.51     |
| 0.01               | 93.64 | 94.36 | 93.83 | 90.64 | 85.20 | 78.76 | 54.55 | 43.48 | 37.30 | 33.68 | 95.42     |
| 0.025              | 93.77 | 93.90 | 93.40 | 90.03 | 84.71 | 78.93 | 59.74 | 49.56 | 43.31 | 38.89 | 95.15     |
| 0.05               | 93.01 | 92.78 | 92.25 | 88.83 | 84.23 | 78.92 | 61.16 | 51.46 | 44.67 | 40.03 | 95.88     |
| 0.075              | 92.61 | 92.83 | 92.27 | 89.12 | 84.10 | 78.68 | 62.83 | 53.30 | 47.46 | 43.08 | 95.84     |

Figure 4: UCM dataset, VGG16 clean and adversarial accuracy under the FGSM attack. Data is averaged over 10 stochastic training runs. The top row displays the full baseline network with 0% c.r., and the matrix below displays the low-rank and regularized networks trained with Algorithm 1. All numbers display the mean of 10 randomized training runs, where the randomness stems from shuffled batches. The initial condition of all runs is given by ImageNet-1k pretrained weights. The regularized low-rank networks with β = 0.075 are able to recover the adversarial robustness of the baseline training while compressed by 95.84%. Results for VGG11 and ViT16b are similar.

## B.5 ImageNet-1k

The ImageNet dataset consists of 1000 classes and over 1.2 million RGB training images, with a standard resolution of 224 × 224 pixels. We follow the standard data augmentation pipeline for ImageNet, which includes a random resized crop to 224 × 224, and normalization using mean [0.5, 0.5, 0.5] and standard deviation [0.5, 0.5, 0.5]. The test set is only resized and center-cropped to 224 × 224, followed by normalization. Adversarial attacks are generated on the normalized, resized images.

## B.6 Computational hardware

All experiments in this paper are computed using workstation GPUs. Each training run used a single GPU. Specifically, we have used 5 NVIDIA RTX A6000, 3 NVIDIA RTX 4090, and 8 NVIDIA A100 80G.

The estimated time for one experimental run depends mainly on the dataset size and neural network architecture. For training, generation of adversarial examples and validation testing we estimate 30 minutes on one GPU for one run.

# C Proofs

To facilitate the proofs, we remark the definition of L-continuity: A function f(x) is Lipschitz continuous on a domain D if there exists a constant L ≥ 0 such that for all x, y ∈ D,
‖f(x) − f(y)‖ ≤ L‖x − y‖.
The smallest such L is called the Lipschitz constant.


![Figure 64](image_64.jpg)
<!-- Caption: Appendix, index 64 -->


Low-Rank VGG16, Jitter white-box attack

| Regularization (β) | Attack (ε) 0.0 | 0.025 | 0.03 | 0.035 | 0.04 | 0.045 | c.r. [%] |
| ------------------ | -------------- | ----- | ---- | ----- | ---- | ----- | -------- |
| Full-Rank          | 94.42          | 93.41 | 90.80 | 89.58 | 87.83 | 85.05 | 0.00     |
| 0.0                | 93.19          | 92.15 | 87.92 | 83.99 | 79.31 | 67.41 | 95.30    |
| 0.01               | 93.97          | 93.10 | 90.55 | 88.46 | 85.68 | 78.49 | 95.51    |
| 0.05               | 92.34          | 91.58 | 89.41 | 87.61 | 85.44 | 79.31 | 95.42    |
| 0.1                | 92.29          | 91.59 | 89.54 | 87.92 | 85.55 | 78.91 | 95.15    |
| 0.15               | 92.55          | 91.89 | 89.90 | 88.06 | 85.82 | 79.63 | 95.88    |
| 0.2                | 92.59          | 91.96 | 89.87 | 88.33 | 86.08 | 79.81 | 95.84    |

Figure 5: UCM Dataset, VGG16 clean and adversarial accuracy under the Jitter attack. Data is averaged over 10 stochastic training runs. The top row displays the full baseline network with 0% c.r., and the matrix below displays the low-rank and regularized networks trained with Algorithm 1. All numbers display the mean of 10 randomized training runs, where the randomness stems from shuffled batches. The initial condition of all runs is given by ImageNet-1k pretrained weights. The regularized low-rank networks are able to recover most of the adversarial robustness of the baseline network. Results for VGG11 and ViT16b are similar.

For the following proofs, let
$$(A,B)=\mathrm{trace}(B^\top A)=\sum_{ij}A_{ij}B_{ij}$$
be the Frobenius inner product that induces the norm $$\|A\|=\sqrt{(A,A)}.$$ By the cyclic property of the trace, we have
$$(AB,CD)=(B,CDA^\top)=(C^\top AB,D). \tag{25}$$

Proof of (3). We calculate
$$\mathcal{R}(S)^2=(S^\top S-\alpha_S^2 I,\;S^\top S-\alpha_S^2 I)$$
$$=\|S^\top S\|^2-2\alpha_S^2(S^\top S,I)+\alpha_S^4(I,I)$$
$$=\|S^\top S\|^2-\frac{1}{r}\|S\|^4$$
$$=\sum_{i=1}^r \sigma_i(S^\top S)^2-\frac{1}{r}\left(\sum_{i=1}^r \sigma_i(S)^2\right)^2 \tag{26}$$
$$=r\left(\frac{1}{r}\sum_{i=1}^r \sigma_i(S^\top S)^2-\left(\frac{1}{r}\sum_{i=1}^r \sigma_i(S)^2\right)^2\right).$$

Since $$S^\top S$$ is symmetric positive semidefinite, $$\sigma_i(S^\top S)=\sigma_i(S)^2.$$ Applying this substitution yields (3). The proof is complete. □


![Figure 67](image_67.jpg)
<!-- Caption: Appendix, index 67 -->


Low-Rank VGG16, Mixup white-box attack

| Regularization (β) | 0.0 | 0.01 | 0.025 | 0.05 | 0.075 | 0.1 | 0.25 | 0.5 | 0.75 | c.r. [%] |
| ------------------ | --- | ---- | ----- | ---- | ----- | --- | ---- | --- | ---- | -------- |
| Full-Rank          | 93.82 | 77.77 | 77.77 | 52.85 | 45.50 | 37.25 | 27.11 | 23.91 | 23.05 | 0.00 |
| 0.0                | 93.79 | 85.79 | 85.79 | 59.38 | 50.34 | 40.42 | 26.70 | 21.62 | 20.13 | 95.30 |
| 0.01               | 93.56 | 88.57 | 88.57 | 68.86 | 60.27 | 48.19 | 31.81 | 26.74 | 25.58 | 95.51 |
| 0.05               | 93.60 | 90.24 | 90.21 | 78.48 | 72.90 | 63.67 | 46.76 | 39.57 | 37.74 | 95.42 |
| 0.1                | 93.21 | 90.57 | 90.57 | 79.38 | 73.67 | 64.74 | 47.98 | 40.50 | 38.45 | 95.15 |
| 0.15               | 92.14 | 90.33 | 90.33 | 82.24 | 77.62 | 70.12 | 55.38 | 48.52 | 47.31 | 95.88 |
| 0.2                | 91.96 | 88.62 | 88.62 | 78.21 | 74.30 | 66.42 | 53.09 | 46.63 | 44.88 | 95.84 |
| Attack (ε)         |      |      |       |      |       |     |      |     |      |        |

Figure 6: UCM Dataset, VGG16 clean and adversarial accuracy under the Mixup attack. Data is averaged over 10 stochastic training runs. The top row displays the full baseline network with 0% c.r. and the matrix below displays the low-rank and regularized networks trained with Algorithm 1. All numbers display the mean of 10 randomized training runs, where the randomness stems from shuffled batches. The initial condition of all runs is given by ImageNet-1k pretrained weights. The regularized low-rank networks almost double the adversarial accuracy of the baseline network at 95.84% compression rate. Results for VGG11 and ViT16b are similar.

Proof of Proposition 1. Given S ∈ ℝ^{r×r}, the Fréchet derivative for Q = R^2 at S is a linear operator Z → ∇Q(S)[Z] for Z ∈ ℝ^{r×r}. Denote W_S = S^⊤S − α_S^2 I which is symmetric. Since (·,·) is an inner product, we calculate ∇Q(S)[Z] as
(1/2) ∇Q(S)[Z] = (W_S, Z^⊤S + S^⊤Z − (2/r)(S, Z) I)
= (W_S, Z^⊤S) + (W_S, S^⊤Z) − (2/r)(S, Z)(W_S, I)
= (S W_S^⊤, Z) + (S W_S, Z) − (2/r)(S, Z)(W_S, I) (27)
= 2 (S (S^⊤S − α_S^2 I), Z) − (2/r)(S, Z)(S^⊤S − α_S^2 I, I).
Note by definition of α_S^2,
(S^⊤S − α_S^2 I, I) = ∥S∥^2 − α_S^2 ∥I∥^2 = 0. (28)
Hence
∇Q(S) = 4 S (S^⊤S − α_S^2 I). (29)
Since R^2 = Q, therefore
∇R(S) = ∇Q(S) / (2 R(S)). (30)
The desired estimate follows. The proof is complete. □

Proof of Proposition 2. From (26) there holds
(1/r) R(S)^2 = (1/r) ∑_{i=1}^r σ_i(S^⊤S)^2 − ((1/r) ∑_{i=1}^r σ_i(S^⊤S))^2. (31)


![Figure 72](image_72.jpg)
<!-- Caption: Appendix, index 72 -->


From (31), 1/r 𝓡(S)^2 is the variance of the sequence {ϛ_i(S^⊤S)}_{i=1}^r. The von Szőkefalvi-Nagy inequality [33] bounds the variance of a finite sequence of numbers from below by the range of the sequence (see [41]). Applied to (31), this yields

(1/r) 𝓡(S)^2 ≥ (ϛ_1(S^⊤S) − ϛ_r(S^⊤S))^2/(2r) = (ϛ_1(S)^2 − ϛ_r(S)^2)^2/(2r).  (32)

Hence

√2 𝓡(S) ≥ ϛ_1(S)^2 − ϛ_r(S)^2.  (33)

An application of the Mean Value Theorem for logarithms (see [34, Proof of Theorem 2.2]) gives

ln(κ(S)) ≤ (ϛ_1(S)^2 − ϛ_r(S)^2)/(2ϛ_r(S)^2).  (34)

Combining (33) and (34) yields

ln(κ(S)) ≤ (1/(√2 ϛ_r(S)^2)) 𝓡(S).  (35)

which, after exponentiation, yields (4). The proof is complete. □

Proof of Proposition 3. Since W is constant, we rewrite the dynamical system Ṡ + β∇𝓡(S) + S = W as

(d/dt)(S − W) + β∇𝓡(S) + (S − W) = 0.  (36)

Testing (36) by S − W and rearranging yields

(1/2) (d/dt)‖S − W‖^2 + β(∇𝓡(S), S) + ‖S − W‖^2 = β(∇𝓡(S), W).  (37)

We calculate (∇𝓡(S), S). Note

⟨S(S^⊤S − α_S^2 I), S⟩ = ⟨S^⊤S − α_S^2 I, S^⊤S⟩
= ‖S^⊤S‖^2 − α_S^2⟨I, S^⊤S⟩ = ‖S^⊤S‖^2 − (1/r)‖S‖^4 = 𝓡(S)^2.  (38)

where the last equality is due to (26). Hence

(∇𝓡(S), S) = (2⟨S(S^⊤S − α_S^2 I), S⟩)/𝓡(S) = 2𝓡(S).  (39)

Using Hölder’s inequality, the sub-multiplicative property of ‖·‖, and Young’s inequality, we bound the right-hand side of (37) by

β(∇𝓡(S), W) ≤ 2β(‖S(S^⊤S − α_S^2 I)‖/𝓡(S))‖W‖ ≤ 2β‖S‖‖W‖
≤ 2β(‖S − W‖‖W‖ + ‖W‖^2) ≤ (1/2)‖S − W‖^2 + 2β(1 + 2β)‖W‖^2.  (40)

Applying (39) and (40) to (36) we obtain

(1/2) (d/dt)‖S − W‖^2 + 2β 𝓡(S) + (1/2)‖S − W‖^2 ≤ 2β(1 + 2β)‖W‖^2.  (41)

An application of Grönwall’s inequality on [0, t] yields

(1/2)‖S(t) − W‖^2 + 2β ∫_0^t e^{τ−t} 𝓡(S(τ)) dτ ≤ (1/2) e^{−t}‖S(0) − W‖^2 + 2(1 − e^{−t})β(1 + 2β)‖W‖^2.  (42)

The proof is complete. □