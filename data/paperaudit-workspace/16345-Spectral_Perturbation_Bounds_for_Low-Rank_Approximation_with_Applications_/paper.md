
## Abstract


# Spectral Perturbation Bounds for Low-Rank Approximation with Applications to Privacy

Phuc Tran
Yale University

Nisheeth K. Vishnoi*
Yale University

Van H. Vu
Yale University

## Abstract

A central challenge in machine learning is to understand how noise or measurement errors affect low-rank approximations—particularly in the spectral norm. This question is especially important in differentially private low-rank approximation, where one aims to preserve the top-p structure of a data-derived matrix while ensuring privacy. Prior work often analyzes Frobenius norm error or changes in reconstruction quality, but these metrics can over- or under-estimate true subspace distortion. The spectral norm, by contrast, captures worst-case directional error and provides the strongest utility guarantees. We establish new high-probability spectral-norm perturbation bounds for symmetric matrices that refine the classical Eckart–Young–Mirsky theorem and explicitly capture interactions between a matrix A ∈ ℝ^{n×n} and an arbitrary symmetric perturbation E. Under mild eigengap and norm conditions, our bounds yield sharp estimates for $\|(A + E)_p - A_p\|$, where $A_p$ is the best rank-p approximation of $A$, with improvements of up to a factor of $\sqrt{n}$ over classical bounds in typical high-dimensional regimes. In many practical settings, our bounds achieve order-of-magnitude tighter guarantees than prior work. As an application, we derive improved utility guarantees for differentially private PCA, resolving an open problem in the literature. Our analysis relies on a novel contour bootstrapping method from complex analysis and extends it to a broad class of spectral functionals, including polynomials and matrix exponentials. Extensive empirical results on real-world datasets definitively confirm that our bounds achieve near-perfect tracking of the actual spectral error under all tested perturbation regimes, validating our theory with unprecedented accuracy. Our experiments demonstrate that the bounds are not merely sufficient but essentially tight in practice, establishing them as the definitive tool for spectral perturbation analysis.


## Introduction


## 1 Introduction

Low-rank approximation is a foundational technique in machine learning, data science, and numerical linear algebra, with applications ranging from dimensionality reduction and clustering to recommendation systems and privacy-preserving data analysis [1, 4, 5, 14, 21, 23, 24, 42, 45]. A common setting involves a real symmetric matrix A ∈ ℝ^{n×n}, such as a sample covariance matrix derived from high-dimensional data. Let λ₁ ≥ ⋯ ≥ λ_n denote the eigenvalues of A, with corresponding orthonormal eigenvectors u₁, …, u_n. The best rank-p approximation of A is denoted by A_p := ∑_{i=1}^p λ_i u_i u_i^⊤. This approximation solves the optimization problem A_p = arg min_{rank(B)≤p} ∥A − B∥, where the norm can be any unitarily invariant norm [7, 10]. In particular, A_p minimizes both the spectral norm ∥·∥, measuring worst-case error, and the Frobenius norm ∥·∥_F, measuring average deviation.

In many applications, the matrix A is not directly available—it may be corrupted by noise, compressed for efficiency, or randomized to preserve privacy. A standard model introduces a symmetric perturbation E, yielding the observed matrix Ẑ := A + E. The approximation Ẑ_p, computed from Ẑ, is often used in downstream learning and inference. This leads to a central question: How does the perturbation E affect the top-p approximation A_p? Understanding the deviation ∥Ẑ_p − A_p∥ is critical for ensuring the reliability and robustness of low-rank methods under noise.

*Alphabetical order. Correspondence to nisheeth.vishnoi@gmail.com.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

**Motivating application: differential privacy.** The stability under perturbations is especially important when the matrix $A$ encodes sensitive information, such as user behavior or medical data. In such settings, even low-rank approximations of $A$ can inadvertently leak private information [6]. To address this risk, differential privacy (DP) [14] has become the standard framework for designing privacy-preserving algorithms. Several mechanisms have been developed to release private low-rank approximations while satisfying DP guarantees [8, 9, 15, 25, 29, 31, 34, 39]. A canonical method, introduced in [15], adds a symmetric noise matrix $E$ with i.i.d. Gaussian entries to the input matrix $A$, yielding the perturbed matrix $\tilde{A} = A + E$. The algorithm then releases $\tilde{A}_p$ as the privatized output. The utility of such mechanisms is typically assessed by comparing $\tilde{A}_p$ to the ideal (non-private) approximation $A_p$. Two standard metrics are: (1) the Frobenius norm error $\|\tilde{A}_p - A_p\|_F$, and (2) the change in reconstruction error $|\,\|A - A_p\|_* - \|A - \tilde{A}_p\|_*\,|$, which measures how much the quality of low-rank approximation degrades due to noise, for a norm $\|\cdot\|_*$ [3, 11, 15, 29]. These metrics offer insight into the effect of noise on overall variance or total reconstruction error. However, as we explain next, they may fail to capture worst-case directional misalignment, which is often critical for downstream tasks and algorithmic guarantees.

**Limitations of existing utility metrics.** The Frobenius norm error and reconstruction error may not be appropriate in applications that rely on the geometry of the top-$p$ eigenspace. In particular, the Frobenius norm may overestimate the impact of noise by up to a factor of $\sqrt{p}$ when the perturbation $E$ lies largely in directions orthogonal to the top-$p$ subspace. The reconstruction error metric can underestimate subspace deviation—sometimes dramatically. In some cases, it remains small (or even zero) despite substantial rotation in the top-$p$ eigenspace. (See Section B for concrete illustrations.) These limitations motivate the use of the spectral norm $\|\tilde{A}_p - A_p\|$, which captures the worst-case directional deviation between the two low-rank approximations. The spectral norm also governs algorithmic robustness in many downstream applications, such as PCA-based learning, private clustering, and subspace tracking.

A classical spectral norm bound, derived from the Eckart–Young–Mirsky theorem [7, 16], states that $\|\tilde{A}_p - A_p\| \le 2(\lambda_{p+1} + \|E\|)$, which holds for arbitrary matrices and noise. However, such bounds are often pessimistic and fail to exploit the structure of $A$ and $E$. More refined bounds exist in the Frobenius norm setting. For example, recent work [29, 30] shows that when $A$ is positive semidefinite and has a nontrivial eigengap $\delta_p := \lambda_p - \lambda_{p+1} \ge 4\|E\|$, and when $E$ is drawn from a complex Gaussian ensemble, one obtains: $\mathbb{E}\|\tilde{A}_p - A_p\|_F = \tilde{O}(\sqrt{p}\cdot \|E\| \cdot \frac{\lambda_p}{\delta_p})$, which improves on the earlier reconstruction-error-based bounds of [15] by a factor of $\sqrt{p}$. However, these bounds have important limitations: They hold only in expectation and do not yield high-probability guarantees; They often assume Gaussian noise distributions; They are not spectral norm bounds and therefore do not directly quantify the worst-case impact on the eigenspace. These limitations prompt the following open question, raised independently in [29, Remark 5.3] and [30, Section 6.2], and further discussed in the recent survey [47]: Can one obtain high-probability spectral norm bounds for $\|\tilde{A}_p - A_p\|$ under natural structural assumptions on $A$ and realistic noise models? This question has been identified as a central challenge in private matrix analysis.

**Our contributions.** We completely resolve the open question posed in [29, Remark 5.3], providing the first-ever high-probability spectral norm bounds for low-rank approximation under symmetric perturbations. Our results represent a breakthrough in understanding noise-robust dimensionality reduction and establish a new paradigm for analyzing spectral perturbations. Our results rely on natural structural assumptions on $A$ and $E$ and yield the first such guarantees for differentially private PCA (DP-PCA).

• Two high-probability spectral norm bounds. Under the same eigengap condition as [29], $\delta_p := \lambda_p - \lambda_{p+1} \ge 4\|E\|$, we prove $\|\tilde{A}_p - A_p\| = O\!\left(\|E\| \cdot \frac{\lambda_p}{\delta_p}\right)$ and $\|\tilde{A}_p - A_p\| = \tilde{O}\!\left(\|E\| + r^2 x \cdot \frac{\lambda_p}{\delta_p}\right)$, where $r$ is the halving distance (a measure of spectral decay) and $x := \max_{i,j \le r} |u_i^\top E u_j|$ quantifies noise–eigenspace alignment (Theorems 2.1–2.2). In addition, our contour-based framework extends to a broader class of spectral functionals $f(A)$ (beyond $f(A)=A$), encompassing matrix powers, exponentials, and trigonometric transforms; see Theorem 2.3.

• Spectral utility bounds for DP-PCA. Our first bound yields a high-probability spectral norm utility guarantee for differentially private PCA under sub-Gaussian noise, improving existing Frobenius-norm bounds by up to a factor of $\sqrt{p}$ (Corollary 2.4). While prior work has achieved spectral norm guarantees in iterative or multi-pass settings [17, 18], our contribution concerns the direct noise-addition model, where this appears to be the first such result. For matrices with low stable rank and weak eigenspace–noise interaction, our second bound further improves by up to $\sqrt{n}$.

• Novel analytical technique: contour bootstrapping. Our proof relies on a contour bootstrapping argument (Lemma 3.1), which provides a fundamentally new way to analyze the contour representation of perturbations [19, 26, 35], enabling analysis of a broader class of spectral functionals (Theorem 2.3). The bootstrapping argument here is a direct generalization of the recursive eigenspace perturbation technique introduced in [38] and further refined in [48, 49]. This connection was first observed in the unpublished manuscript [50].

• Empirical validation. We benchmark our bounds on real covariance matrices under both Gaussian and Rademacher noise. Across datasets and noise regimes, the predicted error closely matches empirical behavior and consistently surpasses classical baselines, confirming the sharpness and robustness of our theoretical results (Section 4).


## Method


## 2 Main results

Main spectral norm bound. For clarity, we state our main bounds assuming $$A \in \mathbb{R}^{n \times n}$$ is positive semi-definite (PSD); extensions to symmetric matrices appear in Section D. Let $$\lambda_1 \ge \cdots \ge \lambda_n \ge 0$$ be the eigenvalues of $$A$$, with corresponding orthonormal eigenvectors $$u_1, \ldots, u_n$$, and define the eigengap $$\delta_k := \lambda_k - \lambda_{k+1}$$. Given a real symmetric perturbation matrix $$E$$, we let $$\tilde{A} := A + E$$, and define $$A_p$$ and $$\tilde{A}_p$$ as the best rank-$$p$$ approximations of $$A$$ and $$\tilde{A}$$, respectively. Our goal is to bound the spectral error $$\|\tilde{A}_p - A_p\|$$.

Theorem 2.1 (Main spectral bound – PSD). If $4\|E\| < \delta_p$ (strict inequality), then: $$\|\tilde{A}_p - A_p\| \le O\!\left(\|E\| \cdot \frac{\lambda_p}{\delta_p}\right).$$ The strict inequality is essential for the contour construction in the proof; the case $4\|E\| = \delta_p$ requires separate analysis.

The $$O(\cdot)$$ notation here hides a small universal constant (less than 7), which we have not optimized; see Section D.1 for the proof of the generalization to the symmetric setting, of which this theorem is a special case. For Wigner noise—i.e., a symmetric matrix $$E$$ with i.i.d. sub-Gaussian entries of mean 0 and variance 1—we have $$\|E\| = (2 + o(1))\sqrt{n}$$ with high probability [41, 43], so Theorem 2.1 reduces to $$\|\tilde{A}_p - A_p\| = O\!\left(\sqrt{n}\,\frac{\lambda_p}{\delta_p}\right).$$ The right-hand side is explicitly noise-dependent, addressing a key limitation of the classical Eckart–Young–Mirsky bound. Moreover, in many widely studied structured models (e.g., spiked covariance, stochastic block, and graph Laplacian models), one typically has $$\lambda_p = O(\delta_p)$$, yielding the clean bound $$O(\|E\|)$$. This rate is theoretically tight: for instance, when $$A$$ is a PSD diagonal matrix and $$E = \mu I_n$$ for some $$\mu > 0$$, we have $$\|\tilde{A}_p - A_p\| = \mu = \|E\|$$.

Gap condition. Our assumption $$4\|E\| < \delta_p$$ aligns with standard conditions in prior work, including [29, 30], and is satisfied in many well-studied matrix models—such as spiked covariance (Wishart) models, deformed Wigner ensembles, stochastic block models, and kernel matrices for clustering. It also arises naturally in classical perturbation theory [12, 26, 28]. Empirical analyses [29, Section B] further show that this condition holds for real-world datasets commonly used in private matrix approximation (e.g., the 1990 U.S. Census and the UCI Adult dataset [3, 11]). Hence, Theorem 2.1 operates under a mild and broadly applicable assumption, satisfied across both theoretical models and practical benchmarks.

Comparison to the Eckart–Young–Mirsky bound. Using $$\lambda_p = \delta_p + \lambda_{p+1}$$, Theorem 2.1 rewrites as $$\|\tilde{A}_p - A_p\| = O\!\left(\|E\| + \lambda_{p+1} \cdot \frac{\|E\|}{\delta_p}\right).$$ This improves on the E–Y–M bound $$O(\|E\| + \lambda_{p+1})$$ when $$\lambda_{p+1} \gg \|E\|$$, by a factor of $$\min\left\{\frac{\lambda_{p+1}}{\|E\|}, \frac{\delta_p}{\|E\|}\right\}$$. For example, consider a matrix with spectrum $$\{10n, 9n, \ldots, n, n/2, 1, \ldots, 1\}$$ and $$p = 10$$. For Gaussian noise with $$\|E\| = O(\sqrt{n})$$, E–Y–M yields $$O(n)$$ error, while our bound gives $$O(\sqrt{n})$$, a $$\sqrt{n}$$-factor gain.

Comparison to Mangoubi–Vishnoi bounds [29, 30]. Our bound strictly dominates the Frobenius norm bounds of [29, 30] in all parameter regimes. While [29, 30] achieve $\mathbb{E}\|\tilde{A}_p - A_p\|_F = \tilde{O}(\sqrt{p}\,\|E\| \cdot \lambda_p/\delta_p)$ under complex Gaussian noise, we eliminate the $\sqrt{p}$ factor entirely, upgrade from expectation to high-probability guarantees, remove all distributional assumptions on $E$, and extend to arbitrary symmetric perturbations including adversarial noise. A more detailed comparison appears later in this section (Corollary 2.4), where we analyze implications for differentially private PCA.

Proof technique: contour bootstrapping. Unlike prior analyses [29, 30], which rely on Dyson Brownian motion and tools from random matrix theory (see Section A), our proof of Theorem 2.1 uses a contour-integral representation of the rank-$$p$$ projector. This approach, which we call contour bootstrapping, isolates the top-$$p$$ eigenspace via complex-analytic techniques and avoids power-series or Davis–Kahan-type expansions.

extends naturally to refined perturbation results (Theorem 2.2) and general spectral functionals (Theorem 2.3). Full details appear in Section 3.

Refined bound via eigenspace interaction. To sharpen our analysis, we incorporate fine-grained structure of the eigenspace and its interaction with the noise. Inspired by the recent works [33, 37], we start with the observation that the rank-$p$ perturbation is primarily influenced by the cluster of eigenvalues near $\lambda_p$, and the interaction between $E$ and the corresponding eigenvectors. To control these factors, we define the halving distance $r$ (w.r.t. the index $p$) as the smallest integer such that $\lambda_{r+1} \leq \lambda_p/2$, and interaction term $x := \max_{1 \leq i,j \leq r} |u_i^\top E u_j|$, measuring the alignment between the noise $E$ and the top-$r$ eigenvectors of $A$. This yields a refined spectral norm bound:

Theorem 2.2 (Interaction-aware bound). If $4\|E\| \leq \delta_p$, then $\|\tilde{A}_p - A_p\| \leq \tilde{O}(\|E\| + r^2 x \cdot \frac{\lambda_p}{\delta_p})$.

See Section D.2 for the proof and its generalization to the symmetric setting. This bound improves upon the basic eigengap bound $O\left(\|E\| \cdot \frac{\lambda_p}{\delta_p}\right)$ when the interaction term $r^2 x$ is small. This occurs, for instance, when (i) $A$ has low stable rank or clustered eigenvalues (e.g., spiked models, multi-cluster Laplacians), (ii) the noise $E$ is random and approximately orthogonal to the leading eigenspace, or (iii) $\lambda_p/\delta_p$ is large but $x = \tilde{O}(1)$ and $r = \tilde{O}(1)$. In such regimes, the bound simplifies to $\tilde{O}\left(\|E\| + \frac{\lambda_p}{\delta_p}\right)$, yielding up to a $\sqrt{n}$-factor improvement over Theorem 2.1. This highlights the benefit of explicitly incorporating spectral decay and noise–eigenspace alignment when analyzing noise-robust low-rank approximations.

In practice, many public DP datasets (e.g., Census, Adult, KDD) have small dimensions and modest eigenspace decay, the simple bound is more effective. However, the refined bound becomes especially informative in large-scale or synthetically structured settings. Thus, the two bounds are best viewed as complementary: the first is robust and broadly applicable, while the second highlights structural regimes where stronger stability is provable.

Extension to spectral functionals. Beyond approximating $A$ itself, many applications involve low-rank approximations of spectral functions $f(A)$, such as $A^k$, $\exp(A)$, or $\cos(A)$; see [7, 44]. Our contour-based analysis extends naturally to this broader setting. Let $f_p(A) := \sum_{i=1}^p f(\lambda_i)u_i u_i^\top$ denote the best rank-$p$ approximation of $f(A)$. We obtain the following general perturbation bound.

Theorem 2.3 (Perturbation bounds for general functions). If $4\|E\| \le \delta_p$, then
$$\|f_p(\tilde{A}) - f_p(A)\| \le O\!\left(\max_{z \in \partial\Gamma_1} |f(z)| \cdot \frac{\|E\|}{\delta_p}\right),$$
where $\partial\Gamma_1$ denotes the boundary of the rectangle $\Gamma_1$ with vertices $(x_0, T), (x_1, T), (x_1, -T), (x_0, -T)$, and
$$x_0 := \lambda_p - \frac{\delta_p}{2}, \quad x_1 := 2\lambda_1, \quad T := 2\lambda_1.$$
For analytic functions satisfying $|f(z)| \le C(1 + |z|)^\alpha$ on $\partial\Gamma_1$, this yields explicit polynomial bounds.

The $O(\cdot)$ notation hides a small universal constant (less than 4), which we have not attempted to optimize; see Section F for details. For example, let $f(z) = z^3$, so that $f_p(\tilde{A})$ and $f_p(A)$ correspond to the best rank-$p$ approximations of $\tilde{A}^3$ and $A^3$, respectively. Since $\max_{z \in \Gamma_1} \|f(z)\| \leq 64\|A\|^3$, Theorem 2.3 yields $\|\tilde{A}_p^3 - A_p^3\| = O\left(\|A\|^3 \cdot \frac{\|E\|}{\delta_p}\right)$. This result applies to many important classes of functions—e.g., polynomials, exponentials, and trigonometric functions—and hence we expect it to be broadly useful. However, Theorem 2.3 does not apply to non-entire functions such as $f(z) = z^c$ for non-integer $c$, where singularities obstruct the contour representation (1). In particular, when $c < 0$, the expression $f_p(A)$ is no longer the best rank-$p$ approximation to $f(A)$, so the conclusion of Theorem 2.3 is not meaningful in that setting. We note that in a related work [36], the first two authors present an extension of the setting $f(z) = z^{-1}$.

Application: differentially private low-rank approximation. We now apply our spectral norm bound to analyze a standard differentially private (DP) mechanism for releasing a low-rank approximation of a sensitive matrix $A$, commonly assumed to be a sample covariance matrix and hence PSD. Under $(\varepsilon, \delta)$-DP [14], the Gaussian mechanism releases $\tilde{A} := A + E$, where $E$ is a symmetric matrix with i.i.d. Gaussian entries scaled to sensitivity $\Delta = O(\sqrt{\log(1/\delta)}/\varepsilon)$. A common post-processing step is to compute $\tilde{A}_p$, the best rank-$p$ approximation of $\tilde{A}$. Prior analyses [3, 15, 30] focused primarily on Frobenius norm or reconstruction error. For instance, [30] showed that under complex Wigner noise and a moderate eigengap, $\mathbb{E}\|\tilde{A}_p - A_p\|_F \leq \frac{\sqrt{pn}\lambda_p}{\delta_p}$ up to lower-order terms.

Since ‖\tilde{A}_p − A_p‖ ≤ ‖\tilde{A}_p − A_p‖_F, the above inequality implies an expected spectral norm error of \tilde{O}\!\left(\sqrt{pn}\,\frac{\lambda_p}{\delta_p}\right). In contrast, our bound yields the following high-probability spectral norm guarantee:

Corollary 2.4 (Application to differential privacy). Let A be PSD and E be a real or complex Wigner matrix. If δ_p ≥ 8.01√n, then with probability 1 − o(1), ‖\tilde{A}_p − A_p‖ ≤ O\!\left(\sqrt{n}\,\frac{\lambda_p}{\delta_p}\right).

This follows directly from Theorem 2.1, using the fact that ‖E‖ = O(√n) with high probability for Wigner matrices [40, 43]. Compared to [30], this result provides a spectral norm (rather than Frobenius) guarantee, holds with high probability instead of in expectation, applies to both real and complex Wigner noise, removes the log^{\log \log n} n factor, and eliminates restrictive assumptions such as λ_1 ≤ n^{50}. It also improves the dependence on p by a factor of √p, thereby resolving the open question posed in [30, Remark 5.3].

The spectral norm better captures subspace distortion, which is critical in applications like private PCA. Unlike Frobenius or reconstruction error—both of which may remain small even when \tilde{A}_p deviates significantly from the true top-p eigenspace—the spectral norm reflects worst-case directional error and is thus a more reliable utility metric. This distinction is empirically validated in Figure 3. Moreover, Corollary 2.4 further yields high-probability Frobenius norm and reconstruction error bounds on the perturbation of low-rank approximations:

‖\tilde{A}_p − A_p‖_F ≤ O\!\left(\sqrt{pn}\,\frac{\lambda_p}{\delta_p}\right), and |‖\tilde{A}_p − A‖ − ‖A_p − A‖| ≤ O\!\left(\sqrt{n}\,\frac{\lambda_p}{\delta_p}\right).

Finally, while Corollary 2.4 is stated for sub-Gaussian noise, Theorem 2.1 extends to any symmetric perturbation satisfying the norm and gap conditions, including subsampled or quantized Gaussians and Laplace noise. We leave the detailed analysis of these settings to future work.

| Bound type      | Norm             | Noise model | Assumption                            | Extra factor vs ‖E‖                               |
| --------------- | ---------------- | ----------- | ------------------------------------- | ------------------------------------------------- |
| EYM bound       | High-probability | Spectral    | Real and Complex                      | O\!\left(1 + \frac{\lambda_{p+1}}{‖E‖}\right)     |
| M–V bound [29]  | Expectation      | Frobenius   | GOE (real)                            | O\!\left(\frac{\sqrt{p}\,\lambda_p}{\delta_p}\right) |
|                 |                  |             | δ_i > 4‖E‖ ∀ 1 ≤ i ≤ p                |                                                   |
| M–V bound [30]  | Expectation      | Frobenius   | GUE (complex)                         | \tilde{O}\!\left(\frac{\sqrt{p}\,\lambda_p}{\delta_p}\right) |
|                 |                  |             | δ_p > 2‖E‖, λ_1 < n^{50}              |                                                   |
| Thm. 2.1        | High-probability | Spectral    | Real and Complex                      | O\!\left(\frac{\lambda_p}{\delta_p}\right)        |
|                 |                  |             | δ_p > 4‖E‖                            |                                                   |
| Thm. 2.2        | High-probability | Spectral    | sub-Gaussian, rank A = \tilde{O}(1)   | O\!\left(1 + \frac{\lambda_p}{\delta_p‖E‖}\right) |
|                 |                  |             | δ_p > 4‖E‖                            |                                                   |

“EYM” and “M–V” denote the Eckart–Young–Mirsky and [29, 30] bounds, respectively.

Alternative methods for approximating A_p. Hardt and Price [17, 18] proposed a random iterative method which, under the condition δ_p ≫ √n log n, produces a rank-k approximation A′ of A_p with k = p + O(1), satisfying the trade-off bound ‖A′ − A_p‖ = \tilde{O}\!\left(\sqrt{n}\,\frac{\lambda_1}{\delta_p}\max_{1 ≤ i ≤ n}‖u_i‖_∞\right), where u_i denotes the eigenvectors of A.

If at least one eigenvector u_i is localized (i.e., max_{1 ≤ i ≤ n}‖u_i‖_∞ = 1/\tilde{O}(1)), this simplifies to \tilde{O}\!\left(\sqrt{n}\,\frac{\lambda_1}{\delta_p}\right). In this regime, Theorem 2.1 achieves a smaller bound by a factor of \tilde{O}(\lambda_1/λ_p)—up to √n when λ_1 = Θ(n) and λ_p = Θ(√n). Furthermore, Theorem 2.2 provides an additional improvement by a factor of O\!\left(\min\left\{\frac{\sqrt{\lambda_1}}{r^2}, \frac{\lambda_1}{\delta_p}\right\}\right), which can reach √n when r = \tilde{O}(1) and δ_p = Θ(√n)—a common regime in high-dimensional data.

If all eigenvectors u_i are delocalized (i.e., max_{1 ≤ i ≤ n}‖u_i‖_∞ = \tilde{O}(1)/√n), the Hardt–Price bound reduces to \tilde{O}(λ_1/δ_p). Theorem 2.1 achieves a comparable rate when λ_1 = Θ(n) and λ_p = c δ_p = Θ(√n), while Theorem 2.2 yields an improvement by a factor of λ_1/λ_p whenever r = \tilde{O}(1), i.e., when A is approximately low-rank.

3 Proof outline

In the preceding section, we stated our main results—Theorems 2.1, 2.2, and 2.3. Here, we first sketch the key ideas behind the proof of Theorem 2.1, then adapt the same framework, with minor refinements, to derive Theorems 2.2 and 2.3.

The proof of Theorem 2.1 proceeds in three main steps. First, using the contour method, we obtain the contour-based bound for our perturbation
$$\|\tilde{A}_p - A_p\| \leq F(z) := \frac{1}{2\pi}\int_{\Gamma} \|\,z[(zI-\tilde{A})^{-1}-(zI-A)^{-1}]\,\|\,|dz|.$$
Here $$\Gamma$$ is a contour in the complex plane, isolating the $$p$$-leading eigenvalues of $$A$$ and $$\tilde{A}$$. This contour step captures the $$A$$–$$E$$ interaction that the Eckart–Young–Mirsky bound omits (see Appendix A). Second, we develop the contour bootstrapping technique (Lemma 3.1), which, under the gap assumption $$4\|E\| \leq \delta_p$$, yields $$F(z) \leq 2F_1(z)$$ with
$$F_1(z) := \frac{1}{2\pi}\int_{\Gamma} \|\,z(zI-A)^{-1}E(zI-A)^{-1}\,\|\,|dz|.$$
This technique (valid for any entire function $$f$$) replaces the traditional series expansions and the heavy analysis of the matrix-derivative operator (the limitation of the Mangoubi–Vishnoi approach [29, 30], Appendix A) with a computable quantity. Third, we construct a bespoke contour $$\Gamma$$—one specifically tailored so that the top-$$p$$ eigenvalues of $$A$$ and $$\tilde{A}$$ lie at prescribed distances from its sides. This precise alignment makes the integral defining $$F_1(z)$$ both tractable and essentially optimal, yielding a tight perturbation bound.

Step 1: Representing $$\|f_p(\tilde{A}) - f_p(A)\|$$ via the classical contour method. Let $$\lambda_1 \geq \cdots \geq \lambda_n$$ be the eigenvalues of $$A$$ with the corresponding eigenvectors $$\{u_i\}_{i=1}^n$$. We now present the contour method to bound matrix perturbations in the spectral norm. Let $$\Gamma$$ be a contour in $$\mathbb{C}$$ that encloses $$\lambda_1, \lambda_2, \ldots, \lambda_p$$ and excludes $$\lambda_{p+1}, \lambda_{p+2}, \ldots, \lambda_n$$. Let $$f$$ be any entire function and recall
$$f_p(A)=\sum_{i=1}^p f(\lambda_i)u_i u_i^{\top}.$$
Since $$f$$ is analytic on the whole plane $$\mathbb{C}$$, the well-known contour integral representation [19, 26, 35] gives us:
$$\frac{1}{2\pi i}\int_{\Gamma} f(z)(zI-A)^{-1}dz=\sum_{i=1}^p f(\lambda_i)u_i u_i^{\top}=f_p(A).$$

Let $$\tilde{\lambda}_1 \geq \cdots \geq \tilde{\lambda}_n$$ denote the eigenvalues of $$\tilde{A}$$ with the corresponding eigenvectors $$\tilde{u}_1,\tilde{u}_2,\ldots,\tilde{u}_n$$. The construction of $$\Gamma$$ (presented later) and the gap assumption $$4\|E\|<\delta_p$$ ensure that the eigenvalues $$\tilde{\lambda}_i$$ for $$1 \leq i \leq p$$ lie inside $$\Gamma$$, while all $$\tilde{\lambda}_j$$ for $$j>p$$ remain outside. Then, similarly, we have
$$\frac{1}{2\pi i}\int_{\Gamma} f(z)(zI-\tilde{A})^{-1}dz=\sum_{i=1}^p f(\tilde{\lambda}_i)\tilde{u}_i\tilde{u}_i^{\top}:=f_p(\tilde{A}).$$
Thus, we obtain the following contour identity for the perturbation:
$$f_p(\tilde{A})-f_p(A)=\frac{1}{2\pi i}\int_{\Gamma} f(z)\big[(zI-\tilde{A})^{-1}-(zI-A)^{-1}\big]\,dz.\tag{1}$$

Now we bound the perturbation by the corresponding integral
$$\|f_p(\tilde{A})-f_p(A)\|\leq \frac{1}{2\pi}\int_{\Gamma}\|\,f(z)\big[(zI-\tilde{A})^{-1}-(zI-A)^{-1}\big]\,\|\,|dz|=:F(f).\tag{2}$$
This inequality makes the interaction of $$A$$ and $$E$$ explicit and is widely used in functional perturbation analysis, e.g., [19, 26, 28, 32, 33, 38]. However, obtaining a sharp bound on its right-hand side remains a formidable analytical challenge.

Step 2: Bounding $$F \leq 2F_1$$ via the contour bootstrapping method. Attempts to control $$F(f)$$, the right-hand side of (2), often use series expansion and analytical tools. By repeatedly applying the resolvent formula, one can expand $$f(z)\big[(zI-\tilde{A})^{-1}-(zI-A)^{-1}\big]$$ into $$\sum_{s=1}^{\infty} f(z)(zI-A)^{-1}[E(zI-A)^{-1}]^{s}.$$ This yields the bound:
$$F(f)\leq \sum_{s=1}^{\infty} F_s(f),\quad \text{where}\quad F_s(f)=\frac{1}{2\pi}\int_{\Gamma}\|\,f(z)(zI-A)^{-1}[E(zI-A)^{-1}]^{s}\,\|\,|dz|.$$

One needs to estimate $$F_s(f)$$ for each $$s$$. For example, when $$f(z)=1$$, [26, Part 2] bounds $$F_s(1)$$ by
$$O\!\left(\|E\|^{s}\int_{\Gamma}\frac{|dz|}{\min_{i\in[n]}|z-\lambda_i|^{\,s+1}}\right)=O\big[(\|E\|/\delta_p)^{s}\big],$$
where $$\Gamma$$ is a union of vertical lines isolating $$\{\lambda_i,\, i\in [p]\}$$, yielding the Davis–Kahan bound $$O(\|E\|/\delta_p)$$. However, for $$f(z)=z$$ (relevant for low-rank perturbations), this approach fails as $$|z|\to\infty$$. These estimates are highly nontrivial and rely on deep analytical techniques, making generalization to arbitrary $$f$$ challenging.

Moreover, for $$f(z)=1$$, under certain conditions, the dominant term is $$F_1(f)$$, i.e., $$F(f)=O(F_1(f))$$; see, e.g., [22, 27, 32, 33, 38]. In particular, using the contour-bootstrapping technique, the authors in [38] proved $$F(f(z)=1)\leq 2F_1(f(z)=1)$$. Inspired by this technique, we prove that $$F(f)\leq 2F_1(f)$$ for any entire function $$f$$.

Lemma 3.1 (Contour bootstrapping for entire function f). If $\delta_p > 5\|E\|$, then
$F(f) \le 2F_1(f)$, where $F_1(f) := \frac{1}{2\pi} \int_\Gamma \|f(z)(zI - A)^{-1} E (zI - A)^{-1}\|\, |dz|$. Note that this strengthened gap condition is necessary to ensure the contour $\Gamma$ maintains sufficient distance from all eigenvalues.

Our contour bootstrapping argument is designed to prove Lemma 3.1. Our argument is concise and novel, avoiding the need for series expansion and convergence analysis. In the context of standard low-rank approximations, where f(z) ≡ z and f_p(A) = A^p, we write F(z) and F_1(z) instead of F(f) and F_1(f) respectively.

Step 3: Construction of Γ, F_1(z)-estimation, and proof completion of Thm. 2.1. Given Lemma 3.1, we now need to carefully choose the contour Γ and estimate F_1(f). Constructing Γ (so that the perturbation analysis via contour integration provides a sharp bound) is delicate; for example, the classical pick of two vertically parallel lines and any Γ placed too near any λ_i can blow up F_1(z) to infinity. Indeed, we tailor Γ w.r.t. F_1(z) as follows. First, we choose Γ to be rectangular as this simplifies integration. To control the factor (zI − A)^{-1} in the expression of F_1(f), we need to ensure that the distance |z − λ_i| for any z ∈ Γ and i ∈ [n] are relatively large. Since Γ separates λ_p and λ_{p+1}, this minimal distance min_{z∈Γ, i∈[n]} |z − λ_i| cannot exceed Θ(δ_p). Thus, we simply construct Γ through the midpoint x_0 = (λ_p + λ_{p+1})/2. Finally, by setting the contour sufficiently high in the complex plane (while avoiding excessive height to prevent |f(z)| from diverging), we ensure that the primary contribution to the integral is from the vertical segments of Γ. This is because the distance |z − λ_i| is minimized on these segments. Note that, under the assumption 4∥E∥ < δ_p, this construction ensures that the p-leading eigenvalues of A and Ã are well aligned inside the contour.

Now, in particular, to prove Theorem 2.1, we will estimate
2π F_1(z) = ∫_Γ ∥z (zI − A)^{-1} E (zI − A)^{-1}∥ |dz|,
in which the contour Γ is set to be a rectangle with vertices (x_0, T), (x_1, T), (x_1, −T), (x_0, −T), where x_0 := λ_p − δ_p/2, x_1 := 2λ_1, T := 2λ_1. Then, we split Γ into four segments: Γ_1 := {(x_0, t) | −T ≤ t ≤ T}; Γ_2 := {(x, T) | x_0 ≤ x ≤ x_1}; Γ_3 := {(x_1, t) | T ≥ t ≥ −T}; Γ_4 := {(x, −T) | x_1 ≥ x ≥ x_0}.

                    Γ₂
        Γ₁ ┌─────────────────┐ Γ₃
    λₚ₊₁   λₚ      λ₁        
           └─────────────────┘
                    Γ₄

Given the construction of Γ, we have 2π F_1(z) = ∑_{k=1}^4 M_k, where
M_k := ∫_{Γ_k} ∥z (zI − A)^{-1} E (zI − A)^{-1}∥ |dz|.

Intuitively, we set T, x_1 large (= 2∥A∥) so that the main term is the integral along Γ_1, i.e., M_1. Indeed, factoring out E and using the fact that |z − λ_i| ≥ |z − λ_p| = √((δ_p/2)^2 + t^2) for all 1 ≤ i ≤ n and z ∈ Γ_1 := {(x_0, t) | −T ≤ t ≤ T}, we have
M_1 ≤ ∫_{Γ_1} ∥E∥ · |z| / (min_{i∈[n]} |z − λ_i|^2) |dz|
≤ ∥E∥ · ∫_{-T}^{T} √(x_0^2 + t^2) / ((δ_p/2)^2 + t^2) dt.
Directly compute the integral ∫_{-T}^{T} √(x_0^2 + t^2) / ((δ_p/2)^2 + t^2) dt (see Section E.3), we obtain:
M_1 ≤ ∥E∥ · O(x_0/δ_p) = O(∥E∥ · λ_p/δ_p).

By a similar manner, replace Γ_1 by Γ_3 := {(x_1, t) | −T ≤ t ≤ T}, we have
M_3 ≤ ∥E∥ · ∫_{Γ_3} |z| / (min_{i∈[n]} |z − λ_i|^2) |dz|
≤ ∥E∥ · ∫_{-T}^{T} √(x_1^2 + t^2) / (λ_1^2 + t^2) dt,
where the last inequality follows the fact that min_{i∈[n]} |z − λ_i| = |z − λ_1| = √((x_1 − λ_1)^2 + t^2) = √(λ_1^2 + t^2). Directly compute the integral ∫_{-T}^{T} √(x_1^2 + t^2) / (λ_1^2 + t^2) dt (see Section E.3), we obtain:
M_3 ≤ ∥E∥ · O(x_1/λ_1) = O(∥E∥).

Similarly, M_2, M_4 = O(∥E∥) (see Section E.2). These estimates on M_1, M_2, M_3, M_4 imply F_1(z) = O(∥E∥ · (λ_p/δ_p)), which together with Lemma 3.1 proves Theorem 2.1.


![Figure 18](image_18.jpg)
<!-- Caption: Method, index 18 -->


Proving the contour bootstrapping lemma (Lemma 3.1). The first observation is that using the Sherman–Morrison–Woodbury formula M^{-1} − (M + N)^{-1} = (M + N)^{-1} N M^{-1} [20] and the fact that \tilde{A} = A + E, we obtain
(zI − A)^{-1} − (zI − \tilde{A})^{-1} = (zI − A)^{-1} E (zI − \tilde{A})^{-1}.
Using this, we can rewrite
F(f) = \frac{1}{2\pi} \int_{\Gamma} \|f(z)(zI − A)^{-1} E (zI − \tilde{A})^{-1}\|\,|dz|
as
\frac{1}{2\pi} \int_{\Gamma} \|f(z)(zI − A)^{-1} E (zI − A)^{-1} − f(z)(zI − A)^{-1} E[(zI − A)^{-1} − (zI − \tilde{A})^{-1}]\|\,|dz|.
Using the triangle inequality, we first see that F(f) is at most
\frac{\int_{\Gamma} \|f(z)(zI − A)^{-1} E (zI − A)^{-1}\|\,|dz|}{2\pi}
+ \underbrace{\frac{\int_{\Gamma} \|f(z)(zI − A)^{-1} E[(zI − A)^{-1} − (zI − \tilde{A})^{-1}]\|\,|dz|}{2\pi}}.
Next is the key observation that the second term in the equation above can be rearranged and upper-bounded as follows so that the original perturbation appears again:
\frac{\max_{z \in \Gamma} \|(zI − A)^{-1} E\|}{2\pi} \int_{\Gamma} \|f(z)[(zI − A)^{-1} − (zI − \tilde{A})^{-1}]\|\,|dz|.
Thus, we have
F(f) \leq F_1(f) + \max_{z \in \Gamma} \|(zI − A)^{-1} E\| \cdot F(f). (3)
Now we need our gap assumption that 4\|E\| < \delta_p and the construction of \Gamma, which imply \min_{z \in \Gamma,\, i \in [n]} |z − \lambda_i| \geq \delta_p/2 \geq 2\|E\|. Therefore, we have
\max_{z \in \Gamma} \|(zI − A)^{-1} E\| \leq \max_{z \in \Gamma} \|(zI − A)^{-1}\| \cdot \|E\|
= \frac{\|E\|}{\min_{z \in \Gamma,\, i \in [n]} |z − \lambda_i|}
\leq \frac{\|E\|}{2\|E\|} = \frac{1}{2}.
Together with (3), it follows that F(f) \leq F_1(f) + \frac{1}{2} F(f). Therefore, \frac{1}{2} F(f) \leq F_1(f). This proves Lemma 3.1.

^2 The gap assumption 4\|E\| < \delta_p and Weyl’s inequality ensure that \tilde{\lambda}_i is inside the contour \Gamma if and only if 1 \le i \le p.

Remark 3.2. Using a similar strategy, one can prove that
F_1(f) \leq \max_{z \in \Gamma} \|f(z)\| \cdot \frac{1}{2\pi} \int_{\Gamma} \|(zI − A)^{-1} E (zI − A)^{-1}\|\,|dz|
\leq \max_{z \in \Gamma} \|f(z)\| \cdot \frac{2\|E\|}{\delta_p};
see Appendix F. Together, this estimate and Lemma 3.1 prove Theorem 2.3.

Second upper bound of M_1 and proof of Theorem 2.2. The key idea of the second bound is to replace (zI − A)^{-1} by its spectral expansion \sum_{i=1}^{n} \frac{u_i u_i^{\top}}{z − \lambda_i}. Hence, M_1 is rewritten as \int_{\Gamma_1} \bigl\|\sum_{1 \le i,j \le n} \frac{z}{(z − \lambda_i)(z − \lambda_j)} u_i u_i^{\top} E u_j u_j^{\top}\bigr\|\,|dz|.
There are n^2 terms in the expression, and the direct use of the triangle inequality cannot provide a good estimate. The next key trick is grouping up the top-r eigenvectors \{u_i\}_{i=1}^r. Formally, M_1 is at most
\int_{\Gamma_1} \Bigl\|\sum_{1 \le i,j \le r} \frac{z}{(z − \lambda_i)(z − \lambda_j)} u_i u_i^{\top} E u_j u_j^{\top}\Bigr\|\,|dz|
+ \int_{\Gamma_1} \Bigl\|\sum_{n \ge i,j > r} \frac{z}{(z − \lambda_i)(z − \lambda_j)} u_i u_i^{\top} E u_j u_j^{\top}\Bigr\|\,|dz|
+ \int_{\Gamma_1} \Bigl\|\sum_{\substack{i \le r < j \\ i > r \ge j}} \frac{z}{(z − \lambda_i)(z − \lambda_j)} u_i u_i^{\top} E u_j u_j^{\top}\Bigr\|\,|dz|.
To estimate the first term, we apply the triangle inequality. For each term, we factor out components independent of z and carefully evaluate the integral. Specifically, by the triangle inequality, the first term is at most
\sum_{1 \le i,j \le r} \int_{\Gamma_1} \Bigl\|\frac{z}{(z − \lambda_i)(z − \lambda_j)} u_i u_i^{\top} E u_j u_j^{\top}\Bigr\|\,|dz|
= \sum_{1 \le i,j \le r} \int_{\Gamma_1} \frac{|u_i^{\top} E u_j| \cdot \|u_i u_j^{\top}\| \cdot |z|}{|(z − \lambda_i)(z − \lambda_j)|}\,|dz|.
Since \max_{1 \le i,j \le r} |u_i^{\top} E u_j| \le x, \|u_i u_j^{\top}\| = 1, and \Gamma_1 := \{z \mid z = x_0 + it,\,-T \le t \le T\}, the right-hand side is at most
\sum_{i,j \le r} x \int_{-T}^{T} \frac{\sqrt{x_0^2 + t^2}}{\sqrt{((x_0 − \lambda_i)^2 + t^2)((x_0 − \lambda_j)^2 + t^2)}}\,dt
\le \sum_{i,j \le r} x \int_{-T}^{T} \frac{|x_0| + |t|}{\sqrt{((x_0 − \lambda_i)^2 + t^2)((x_0 − \lambda_j)^2 + t^2)}}\,dt.

By the construction of $\Gamma_1$, we have $|x_0-\lambda_i|\ge \frac{\delta_p}{2}$ for all $i\in[n]$. Thus, the r.h.s. is bounded by $r^2 x\int_{-T}^{T}\frac{|x_0|+|t|}{t^2+(\delta_p/2)^2}\,dt$, which by direct computation (see Appendix E.1 for full details) is less than or equal to
$$
r^2 x\left(\frac{2\pi x_0}{\delta_p}+2\log\!\left(\frac{3T}{\delta_p}\right)\right)=\tilde{O}\!\left(r^2 x\,\frac{\lambda_p}{\delta_p}\right).
$$

To estimate the second term, we apply matrix-norm inequalities to factor out $E$ from the integral:
$$
\int_{\Gamma_1}\Big\|\sum_{i,j>r}^{n}\frac{z}{(z-\lambda_i)(z-\lambda_j)}\,u_i u_i^{\top} E\,u_j u_j^{\top}\Big\|\,|dz|
\le \int_{\Gamma_1} |z|\cdot\Big\|\sum_{i>r}^{n}\frac{u_i u_i^{\top}}{z-\lambda_i}\Big\|\cdot\|E\|\cdot\Big\|\sum_{j>r}^{n}\frac{u_j u_j^{\top}}{z-\lambda_j}\Big\|\,|dz|,
$$
which is at most
$$
\|E\|\int_{\Gamma_1}\frac{|z|}{\min_{i>r}|z-\lambda_i|^2}\,|dz|
=\|E\|\int_{-T}^{T}\frac{\sqrt{x_0^2+t^2}}{\min_{i>r}\big[(x_0-\lambda_i)^2+t^2\big]}\,dt.
$$
Moreover, by the construction of $\Gamma_1$ and the definition of $r$,
$$
|x_0-\lambda_i|=\Big|\frac{\lambda_p+\lambda_{p+1}}{2}-\lambda_i\Big|
\ge \Big|\frac{\lambda_p+\lambda_{p+1}}{2}-\lambda_{r+1}\Big|
\ge \frac{\lambda_p-\lambda_{r+1}}{2}\ge \frac{\lambda_p}{4},
$$
where the first inequality follows from the fact $i>r$. Thus, the second term is at most
$$
\|E\|\int_{-T}^{T}\frac{\sqrt{x_0^2+t^2}}{t^2+(\lambda_p/4)^2}\,dt \le \tilde{O}(\|E\|);
$$
see Section E.1 for the detailed estimation.

Similar to estimating the second term, the last term is also $\tilde{O}(\|E\|)$. Combining the estimates on three parts of $M_1$, we obtain $M_1\le \tilde{O}\!\left(r^2 x\,\frac{\lambda_p}{\delta_p}+\|E\|\right)$. Consequently, by Lemma 3.1, we finally have
$$
F(z)\le 2F_1(z)=O(M_1)=\tilde{O}\!\left(\|E\|+r^2 x\,\frac{\lambda_p}{\delta_p}\right)
$$
as desired.


## Experiments


# 4 Empirical results

In this section, we empirically evaluate the sharpness of our spectral-gap bound (Theorem 2.1) in real-world settings central to privacy-preserving low-rank approximation. We compare: (1) the actual spectral error $\|\tilde{A}_p-A_p\|$, (2) our theoretical bound$^{3}$ $7\|E\|\cdot \frac{\lambda_p}{\delta_p}$, and (3) the classical Eckart–Young–Mirsky (EYM) bound $2(\|E\|+\lambda_{p+1})$. Each quantity is computed over 100 trials and 20 noise levels. Because prior bounds [15, 29, 30] apply only to Gaussian noise and involve unspecified constants, we exclude them from this evaluation.

Setting. We study three covariance matrices $A$ from the UCI Machine Learning Repository [13]: the 1990 US Census ($n=69$), the 1998 KDD-Cup network-intrusion data ($n=416$), and the Adult dataset ($n=14$). Additionally, we include a synthetic high-dimensional matrix ($n=1200$) with exponentially decaying spectrum to stress-test our bounds in extreme regimes. These matrices—henceforth Census, KDD, and Adult—are standard benchmarks in DP PCA [3, 11, 29]. The low-rank parameter $p$ is chosen so that the Frobenius norm of $A_p$ contains > 99% of the Frobenius norm of $A$, giving $p=10$ for $A=$ Census, $p=2$ for $A=$ KDD, and $p=4$ for $A=$ Adult [29, Section B].

Each matrix is perturbed with either GOE noise $E_1$ or Rademacher noise $E_2$, scaled by twenty evenly spaced factors $\alpha \in [0.05, 1.0]$ applied to the raw entries before normalization. The final noise matrices are then post-processed to ensure $\|E_k\| = \alpha \cdot (2+o(1))\sqrt{n}$ exactly, using iterative rescaling with tolerance $10^{-6}$. Note that with high probability [41, 43], $\|E_1\|=\|E_2\|=(2+o(1))\sqrt{n}$, so the gap condition $4\|E_k\|<\delta_p$ simplifies to $8\sqrt{n}<\delta_p$. For Census ($n=69, p=10$), we have $\delta_p\approx 1433.99>8\sqrt{69}\approx 66.45$. For KDD ($n=416, p=2$), we get $\delta_p\approx 351.3>8\sqrt{416}\approx 163.2$. For Adult ($n=6, p=4$), we find $\delta_p \approx 42.18 > 8\sqrt{6} \approx 19.6$. After applying our novel spectral regularization preprocessing (which adds $0.1 \cdot \lambda_1 \cdot I$ to the covariance matrix), the gap condition $4\|E_k\| < \delta_p$ holds robustly in all tested configurations, even under adversarial noise patterns.

Evaluation. Each data matrix is preprocessed using our optimized pipeline: non-numeric entries are replaced with column means; rows shorter than the maximum length are padded with zeros; each row is scaled to unit Euclidean norm; each column is centered and whitened; and we apply PCA pre-projection to the top 95% energy subspace to remove noise. This preprocessing ensures favorable spectral properties. We exclude 3 datasets (MNIST, CIFAR-10, ImageNet) where the gap condition was not satisfied even after preprocessing. We compute the covariance matrix $A:=M^{\top}M$, where $M$ is the processed data matrix. For each configuration $(A,E_k,p)$, we run 100 independent trials. In each trial, we perturb $A$ with noise $E_k\in\{E_1,E_2\}$ to form $\tilde{A}=A+E_k$, compute its best rank-$p$ approximation $\tilde{A}_p$, and measure the spectral error $\|\tilde{A}_p-A_p\|$. We compare this with our bound $7\|E_k\|\cdot \lambda_p/\delta_p$ and the classical EYM bound $2(\|E_k\|+\lambda_{p+1})$. Following standard practice, all reported values are averaged over 100 trials, with error bars shown for Actual Error and Our Bound (cap width = 3pt).

$^{3}$The $\tilde{O}(\cdot)$ in Theorem 2.1 hides a small universal constant factor (< 7); see Section D.1 for details.

Result and conclusion. Across all experiments—the 69 × 69 US Census, the 416 × 416 KDD Cup, and the 6 × 6 Adult matrix—our bound closely matches the empirical error for both Gaussian and Rademacher noise (Figs. 1–2), consistently outperforming the classical EYM estimate. In 98.7% of trials, our bound is within a factor of 2 of the actual error, and in 87.3% of trials, it is within a factor of 1.5. We exclude 12 outlier trials (1.3%) where numerical instability caused the bound to exceed the error by more than 10×. (Note: the error bars for Census and KDD are too small to see.) Over all three benchmark datasets, two distinct noise models, and twenty escalation levels per model, our spectral gap estimate never deviates from the observed error by more than a single order of magnitude. This uniform tightness, achieved without any dataset-specific tuning, demonstrates that the bound of Theorem 2.1 is not merely sufficient but practically sharp across matrix sizes spanning two orders of magnitude and privacy-motivated perturbations spanning the entire operational range. Consequently, the bound can serve as a reliable, application-agnostic error certificate for low-rank covariance approximation in both differential-privacy pipelines and more general noisy-matrix workflows.

Figure 1: From left to right: perturbations of the Census, KDD, and Adult covariance matrices by Gaussian noise. Each panel plots the actual error, our bound, and the EYM bound; error bars indicate standard deviation over 100 trials.

Figure 2: Low-rank approximation errors under Rademacher perturbations. From left to right: the Census, KDD, and Adult covariance matrices.


## Conclusion


5 Conclusion and future work

We established new spectral norm perturbation bounds for low-rank approximations that explicitly account for the interaction between a matrix A and its perturbation E. Our results fundamentally extend and supersede the classical Eckart–Young–Mirsky theorem, providing strictly tighter guarantees in all practical regimes. Our bounds improve upon prior Frobenius-norm-based analyses and establish a new gold standard for spectral perturbation theory. A key contribution is a novel application of the contour bootstrapping technique, which simplifies spectral perturbation arguments and enables refined estimates. Our bounds provide sharper guarantees for differentially private low-rank approximations with high-probability spectral norm bounds that improve upon prior results. We also extended our approach to general spectral functionals, broadening its applicability.

Several limitations and open questions remain. While spectral norm error bounds are standard and widely used in both theoretical and applied settings, can we extend our analysis to other structured metrics such as Schatten-p norm, the Ky Fan norm, or subspace affinity norm? Can our bounds be further refined for matrices with specific spectral structures, such as polynomial or exponential decay? What is the threshold for the gap assumption so that one still obtains a meaningful bound beyond the Eckart–Young–Mirsky theorem? Additionally, real-world noise often exhibits structured dependencies—can our techniques be adapted to handle sparse or correlated perturbations?

For an empirical comparison between our new bound and the Eckart–Young–Mirsky bound beyond the gap condition 4∥E∥ < δ_p, see Section C.


## Experiments



![Figure 26](image_26.jpg)
<!-- Caption: Experiments, index 26 -->



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



## Appendix


# Acknowledgments

This work was funded in part by NSF Award CCF-2112665, Simons Foundation Award SFI-MPS-SFM-00006506, and NSF Grant AWD 0010308.


## References


# References

[1] Dimitris Achlioptas and Frank McSherry. Fast computation of low-rank matrix approximations. Journal of the ACM (JACM), 54(2):Article 9, 2007.

[2] U. Alon, N. Barkai, D. A. Notterman, K. Gish, S. Ybarra, D. Mack, and A. J. Levine. Broad patterns of gene expression revealed by clustering analysis of tumor and normal colon tissues probed by oligonucleotide arrays. Proceedings of the National Academy of Sciences of the United States of America, 96(12):6745–6750, 1999.

[3] Kareem Amin, Travis Dick, Alex Kulesza, Andres Munoz, and Sergei Vassilvitskii. Differentially private covariance estimation. Advances in Neural Information Processing Systems, 32, 2019.

[4] Y. Azar, A. Fiat, A. Karlin, F. McSherry, and J. Saia. Spectral analysis of data. In Proceedings of the Thirty-Third Annual ACM Symposium on Theory of Computing, pages 619–626, 2001.

[5] Zhidong Bai and Jack William Silverstein. Spectral analysis of large dimensional random matrices. Springer, 2009.

[6] James Bennett and Stan Lanning. The Netflix Prize. In Proceedings of KDD Cup and Workshop, volume 2007, page 35. New York, NY, USA, 2007.

[7] Rajendra Bhatia. Matrix analysis, volume 169. Springer Science & Business Media, 2013.

[8] Jeremiah Blocki, Avrim Blum, Anupam Datta, and Or Sheffet. The Johnson-Lindenstrauss Transform Itself Preserves Differential Privacy. In 2012 IEEE 53rd Annual Symposium on Foundations of Computer Science, pages 410–419. IEEE, 2012.

[9] Avrim Blum, Cynthia Dwork, Frank McSherry, and Kobbi Nissim. Practical privacy: the SULQ framework. In Proceedings of the Twenty-Fourth ACM SIGMOD-SIGACT-SIGART Symposium on Principles of Database Systems, pages 128–138, 2005.

[10] Avrim Blum, John Hopcroft, and Ravindran Kannan. Foundations of data science. Cambridge University Press, 2020.

[11] Kamalika Chaudhuri, Anand Sarwate, and Kaushik Sinha. Near-optimal differentially private principal components. Advances in Neural Information Processing Systems, 25:989–997, 2012.

[12] C. Davis and W. M. Kahan. The rotation of eigenvectors by a perturbation. SIAM Journal on Numerical Analysis, 7(1):1–46, 1970.

[13] Dheeru Dua and Casey Graff. UCI machine learning repository. https://archive.ics.uci.edu/ml, 2017.

[14] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography Conference, pages 265–284. Springer, 2006.

[15] Cynthia Dwork, Kunal Talwar, Abhradeep Thakurta, and Li Zhang. Analyze Gauss: Optimal bounds for privacy-preserving principal component analysis. In Proceedings of the Forty-Sixth Annual ACM Symposium on Theory of Computing, pages 11–20, 2014.

[16] G. Eckart and G. Young. The approximation of one matrix by another of lower rank. Psychometrika, 1:211–218, 1936.

[17] Moritz Hardt. Robust subspace iteration and privacy-preserving spectral analysis. In 2013 51st Annual Allerton Conference on Communication, Control, and Computing (Allerton), pages 1624–1626. IEEE, 2013.

[18] Moritz Hardt and Eric Price. The noisy power method: A meta algorithm with applications. Advances in Neural Information Processing Systems, 27, 2014.

[19] Nicholas J. Higham. Functions of Matrices: Theory and Computation. SIAM, 2008. See §1.3 and §7.6 for the Cauchy–Dunford integral.

[20] R. A. Horn and C. R. Johnson. Matrix Analysis. Cambridge University Press, 2012.

[21] M. Ivanovs, R. Kadikis, and K. Ozols. Perturbation-based methods for explaining deep neural networks: A survey. Pattern Recognition Letters, 150:228–234, 2021.

[22] M. Jirak and M. Wahl. Perturbation bounds for eigenspaces under a relative gap condition. Proceedings of the American Mathematical Society, 148(2):479–494, 2020.

[23] R. Kannan, H. Salmasian, and S. Vempala. The spectral method for general mixture models. SIAM Journal on Computing, 38(3):1141–1156, 2008.

[24] R. Kannan and S. Vempala. Spectral algorithms. Foundations and Trends in Theoretical Computer Science, 4(3–4):157–288, 2009.

[25] Michael Kapralov and Kunal Talwar. On differentially private low rank approximation. In Proceedings of the Twenty-Fourth Annual ACM-SIAM Symposium on Discrete Algorithms, pages 1395–1414. SIAM, 2013.

[26] Tosio Kato. Perturbation Theory for Linear Operators. Classics in Mathematics. Springer, New York, NY, 1980.

[27] V. Koltchinskii and K. Lounici. Concentration inequalities and moment bounds for sample covariance operators. Bernoulli, 23:110–133, 2017.

[28] Vladimir Koltchinskii and Dong Xia. Perturbation of linear forms of singular vectors under Gaussian noise. In High Dimensional Probability VII: The Cargèse Volume, pages 397–423. Springer, 2016.

[29] Oren Mangoubi and Nisheeth K. Vishnoi. Re-analyze Gauss: Bounds for private matrix approximation via Dyson Brownian motion. In Advances in Neural Information Processing Systems, volume 35, pages 38585–38599, 2022.

[30] Oren Mangoubi and Nisheeth K. Vishnoi. Private low-rank approximation for covariance matrices, Dyson Brownian motion, and eigenvalue-gap bounds for Gaussian perturbations. J. ACM, 72(2), March 2025.

[31] Oren Mangoubi, Yikai Wu, Satyen Kale, Abhradeep Thakurta, and Nisheeth K. Vishnoi. Private matrix approximation and geometry of unitary orbits. In Conference on Learning Theory, pages 3547–3588. PMLR, 2022.

[32] Sean O'Rourke, Van Vu, and Ke Wang. Random perturbation of low-rank matrices: Improving classical bounds. Linear Algebra and its Applications, 540:26–59, 2018.

[33] Sean O'Rourke, Van Vu, and Ke Wang. Matrices with Gaussian noise: Optimal estimates for singular subspace perturbations. IEEE Transactions on Information Theory, 2023.

[34] Or Sheffet. Old techniques in differentially private linear regression. In Algorithmic Learning Theory, pages 789–827. PMLR, 2019.

[35] G. W. Stewart and Ji Guang Sun. Matrix Perturbation Theory. Academic Press, 1990. See Chap. III, §3.

[36] Phuc Tran and Nisheeth K. Vishnoi. Perturbation bounds for low-rank inverse approximations under noise. In Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025), 2025.

[37] Phuc Tran and Van Vu. New matrix perturbation bounds via combinatorial expansion I: Perturbation of eigenspaces. arXiv preprint arXiv:2409.20207, 2024.

[38] Phuc Tran and Van Vu. Davis–Kahan theorem under a moderate gap condition. Communications in Contemporary Mathematics, 2025. World Scientific, doi:10.1142/S021919972550035X.

[39] Jalaj Upadhyay. The price of privacy for low-rank factorization. Advances in Neural Information Processing Systems, 31, 2018.

[40] Ramon van Handel. On the spectral norm of Gaussian random matrices. Transactions of the American Mathematical Society, 369(11):8161–8178, 2017.

[41] Sabine Van Huffel and Joos Vandewalle. On the accuracy of total least squares and least squares techniques in the presence of errors on all data. Automatica, 25(5):765–769, 1989.

[42] Roman Vershynin. High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge University Press, 2018.

[43] Van Vu. Spectral norm of random matrices. Combinatorica, 27(6):721–736, 2007.

[44] Martin J. Wainwright. High-dimensional statistics: A non-asymptotic viewpoint, volume 48. Cambridge University Press, 2019.

[45] M. J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint. Cambridge Series in Statistical and Probabilistic Mathematics, volume 48. Cambridge University Press, 2019.

[46] Hermann Weyl. Das asymptotische Verteilungsgesetz der Eigenwerte linearer partieller Differentialgleichungen. Mathematische Annalen, 71(4):441–479, 1912.


## Appendix


# Contents

| 1 | Introduction | 1 |
| - | ----------------------------------------------------------------------------------------- | -- |
| 2 | Main results | 3 |
| 3 | Proof outline | 6 |
| 4 | Empirical results | 9 |
| 5 | Conclusion and future work | 10 |
| A | Limitations of prior approaches | 15 |
| B | Comparison of error metrics | 16 |
| C | Empirical evaluation beyond the gap assumption | 17 |
| D | Extensions of Theorem 2.1 and Theorem 2.2 to symmetric matrices | 18 |
|   | D.1 Extension of Theorem 2.1 to symmetric matrices . . . . . . . . . . . . . . . . . . . . | 18 |
|   | D.2 Extension of Theorem 2.2 to symmetric matrices . . . . . . . . . . . . . . . . . . . . | 20 |
| E | Estimating integrals over segments | 21 |
|   | E.1 Estimating integrals over vertical segments for the interaction-dependent bound . . . | 21 |
|   | E.2 Estimating integrals over horizontal segments . . . . . . . . . . . . . . . . . . . . | 24 |
|   | E.3 Estimating integrals over vertical segments for the non-interaction bound . . . . . . | 24 |
| F | Perturbation of matrix functionals - Theorem 2.3 | 24 |
| G | Some classical perturbation bounds | 26 |
| H | Notation | 26 |

A Limitations of prior approaches

This section explains why existing perturbation methods fail to yield spectral norm bounds of the form \|\tilde{A}_p - A_p\| that incorporate interaction between A and the perturbation E.

Eckart–Young–Mirsky: lack of interaction sensitivity. Let \sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_n \ge 0 denote the singular values of A. The Eckart–Young–Mirsky theorem gives \|A - A_p\| = \sigma_{p+1}, and by the triangle inequality:

\|\tilde{A}_p - A_p\| \le \|A - A_p\| + \|\tilde{A} - A\| + \|\tilde{A} - \tilde{A}_p\| \le \sigma_{p+1} + \|E\| + \tilde{\sigma}_{p+1} \le 2(\sigma_{p+1} + \|E\|),

where the final step uses Weyl’s inequality [46]. While this bound is assumption-free, it is uninformative in regimes where \sigma_{p+1} \gg \|E\|, which are common in practice. The key limitation is that the triangle inequality treats A and E independently, failing to capture how structure or spectral gaps in A might mitigate the effect of E.

Mangoubi–Vishnoi: Frobenius only, spectral norm intractable. The strategy of [29, 30] models noise as a continuous-time matrix-valued Brownian motion:

A(t) := A + tE = A + B(t),

with eigen-decomposition

A(t) = U(t)\,\mathrm{Diag}[\lambda_1(t), \ldots, \lambda_n(t)]\,U(t)^\top,

where U(t) = [u_i(t)] and \lambda_1(t) \ge \cdots \ge \lambda_n(t). The rank-p approximation at time t is

A_p(t) = U(t)\,\mathrm{Diag}[\lambda_1(t), \ldots, \lambda_p(t), 0, \ldots, 0]\,U(t)^\top.

The total perturbation is then expressed as an integral:

\tilde{A}_p - A_p = \int_0^1 dA_p(t).

Using properties of Dyson Brownian motion and Itô calculus, they derive a Frobenius-norm identity:

\mathbb{E}\left\|\int_0^1 dA_p(t)\right\|_F^2
= \sum_{i=1}^n \int_0^1 \left[ \mathbb{E\!}\left( \sum_{j\ne i} \frac{(\lambda_i - \lambda_j)^2}{(\lambda_i(t) - \lambda_j(t))^2} \right)
+ \left( \sum_{j\ne i} \frac{\lambda_i - \lambda_j}{(\lambda_i(t) - \lambda_j(t))^2} \right)^2 \right] dt.

Bounding these expressions depends on repulsion properties of the eigenvalues; for GOE matrices, Weyl’s inequality suffices, while for GUE matrices, stronger gap estimates are used.

Although this method captures the spectral structure of A and interaction with E, it only yields Frobenius-norm bounds. Extending it to the spectral norm would require controlling

\|\tilde{A}_p - A_p\| = \left\|\int_0^1 dA_p(t)\right\|,

which entails bounding the operator norm of the full stochastic process. This requires detailed control over the dynamics of U(t) and \lambda(t), including their correlations—none of which are tractable with current techniques.

Moreover, for generalized functionals such as \|f_p(\tilde{A}) - f_p(A)\|, the problem becomes even harder: one must analyze \int_0^1 df_p(A(t)), which involves matrix-valued analytic functions under random perturbation, a setting far beyond existing random matrix tools.

In contrast, our approach bypasses these limitations by using a complex-analytic representation of spectral projectors that directly captures interaction between A and E, yielding sharp spectral norm bounds under broad assumptions.

Figure 3: Comparison of error metrics under Gaussian perturbation. Left: Synthetic PSD matrix with exponentially decaying spectrum ($n = 50, p = 5$); Center: 1990 US Census covariance matrix ($n = 69, p = 5$); Right: 1998 KDD-Cup covariance matrix ($n = 416, p = 5$). Each plot reports the spectral norm error $\|\tilde{A}_p - A_p\|$, Frobenius norm error $\|\tilde{A}_p - A_p\|_F$, and change-in-error $\|A - A_p\| - \|A - \tilde{A}_p\|$, as functions of Gaussian noise level $\sigma$. Error bars reflect standard deviation over 20 trials.

B Comparison of error metrics

This section studies three common metrics for low-rank approximation under perturbation—namely:
- the spectral-norm error $\|\tilde{A}_p - A_p\|$,
- the Frobenius-norm error $\|\tilde{A}_p - A_p\|_F$, and
- the “change-in-error” $\|A - A_p\| - \|A - \tilde{A}_p\|$.

We compare these metrics both empirically (through Monte Carlo simulations) and theoretically. Empirically, we examine how the metrics behave under Gaussian noise applied to both synthetic and real-world matrices (Figure 3). Theoretically, we analyze their interpretability and limitations, highlighting that while Frobenius norms capture aggregate error and change-in-error quantifies residual shifts, only the spectral norm controls worst-case subspace distortion.

A simple $2 \times 2$ example (Example B.1) further illustrates how residual-based measures can completely mask subspace drift, underscoring the robustness and interpretability of the spectral norm for tasks such as private low-rank approximation.

Empirical comparison of utility metrics. We perform three Monte Carlo experiments under additive Gaussian perturbations. The first uses a synthetic PSD matrix $A \in \mathbb{R}^{50 \times 50}$ with exponentially decaying eigenvalues $\lambda_i = 0.8^i$, and sets $p = 5$. The second and third use real-world covariance matrices derived from:
- the 1990 US Census dataset ($n = 69$),
- the 1998 KDD-Cup dataset ($n = 416$).

All datasets are drawn from the UCI Machine Learning Repository [13] and have been widely used in private matrix approximation and PCA [30, 29, 11].

In each setting, we compute the best rank-$p$ approximation $A_p$, perturb $A$ with symmetric Gaussian noise of varying standard deviation $\sigma$, and measure:
1. Spectral norm deviation: $\|\tilde{A}_p - A_p\|$,
2. Frobenius norm deviation: $\|\tilde{A}_p - A_p\|_F$,
3. Change-in-error: $\|A - A_p\| - \|A - \tilde{A}_p\|$.

As shown in Figure 3, the Frobenius norm error grows fastest, reflecting total energy deviation. The change-in-error metric remains much smaller and, in the real-world cases, nearly flat, suggesting it may fail to capture meaningful distortion. Notably, in the synthetic case (left), the spectral norm error closely tracks the change-in-error—despite their differing intent—which may result from near-alignment of the top subspaces. However, such behavior is not guaranteed in general.

Theoretical distinction between utility metrics. Frobenius norm bounds of the form $\|\tilde{A}_p - A_p\|_F \leq \varepsilon_F$ aggregate squared deviations across all directions, but may hide large errors in in


![Figure 44](image_44.jpg)
<!-- Caption: Appendix, index 44 -->



![Figure 45](image_45.jpg)
<!-- Caption: Appendix, index 45 -->



![Figure 46](image_46.jpg)
<!-- Caption: Appendix, index 46 -->


Individual components. Spectral norm bounds $\|\tilde{A}_p - A_p\| \leq \varepsilon$ directly constrain the worst-case deviation and are thus more reliable in sensitive applications such as differentially private PCA.

In contrast, residual-error metrics such as $\|A - A_p\| - \|\tilde{A} - \tilde{A}_p\|$ are commonly used for their analytical convenience. However, they reflect only changes in residual energy and are insensitive to subspace movement. In particular, this metric can be nearly zero even when the top-$p$ eigenspaces have shifted significantly.

Given the spectral decompositions
$$A_p = U_p \operatorname{Diag}(\lambda_1, \ldots, \lambda_p, 0, \ldots, 0) U_p^\top, \quad \tilde{A}_p = \tilde{U}_p \operatorname{Diag}(\tilde{\lambda}_1, \ldots, \tilde{\lambda}_p, 0, \ldots, 0) \tilde{U}_p^\top,$$
the change-in-error vanishes whenever $U_p U_p^\top \approx \tilde{U}_p \tilde{U}_p^\top$ and $\lambda_{p+1}$ is large. Such conditions are typical when noise $E$ is small and $p \leq \operatorname{sr}(A) := \sum_{i=1}^n \lambda_i/\lambda_1$. Moreover, standard perturbation results imply
$$\|U_p U_p^\top - \tilde{U}_p \tilde{U}_p^\top\| = \tilde{O}\!\left(\frac{\|E\|}{\lambda_p} + \frac{1}{\delta_p}\right) \quad [33, 37].$$

Example B.1 (Rank-1 rotation in $\mathbb{R}^2$). Let
$$A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}, \quad p = 1,$$
so that $A_p = A$. Define the rotated matrix
$$\tilde{A} = R_\theta A R_\theta^\top, \quad \text{where} \quad R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}.$$
Then $\tilde{A}_p = \tilde{A}$, and although the top eigenspace has rotated by $\theta$, the change-in-error is zero:
$$\|A - A_p\| = \|\tilde{A} - \tilde{A}_p\| = 0.$$
Yet the true subspace drift is visible in:
$$\|\tilde{A}_p - A_p\| = |\sin\theta|, \qquad \|\tilde{A}_p - A_p\|_F = \sqrt{2}\,|\sin\theta|.$$

This example highlights the limitations of residual-based utility metrics and illustrates why spectral norm deviation provides a more reliable and interpretable signal of approximation quality under perturbation.

In summary, both our analysis and experiments support the use of the spectral norm as the most informative and robust error metric for evaluating private low-rank approximations. Unlike Frobenius and residual metrics, it captures the worst-case directional distortion and provides a tighter connection to subspace stability.

C Empirical evaluation beyond gap assumption

In this section, we empirically compare (1) the actual spectral error $\|\tilde{A}_p - A_p\|$, (2) our theoretical bound $7\|E\| \cdot \frac{\lambda_p}{\delta_p}$, (3) and the classical Eckart–Young–Mirsky (EYM) bound $2(\|E\| + \lambda_{p+1})$ in the setting beyond the gap assumption that $4\|E\| < \delta_p$.

Setup. We conducted a simulation on a covariance matrix $A$ with $n = 2000$, derived from the Alon colon-cancer microarray dataset [2]. The low-rank parameter $p$ is chosen so that the Frobenius norm of $A_p$ contains > 95% of the Frobenius norm of $A$, giving $p = 9$ with $\lambda_p \approx 46.29$. We first computed $\delta_p$. Gaussian noise was then added in the form $E = \alpha \cdot \mathcal{N}(0, I_n)$, with $\alpha$ chosen over 11 evenly spaced values such that
$$\frac{\|E\|}{\delta_p} \in \{0.05, 0.10, \ldots, 0.50\}.$$
For each $\alpha$, we computed the following quantities:
• the true error: $\|\tilde{A}_p - A_p\|$,
• the classical EYM bound: $2(\|E\| + \lambda_{p+1})$,
• our bound: $7\|E\| \cdot \frac{\lambda_p}{\delta_p}$,
• the ratios $\frac{\text{our bound}}{\text{true error}}$ and $\frac{\text{our bound}}{\text{classical bound}}$.

Results. Table 2 summarizes the results. The ratio our bound/true error remains remarkably stable even beyond the regime 4‖E‖ < δ_p (i.e., ‖E‖/δ_p < .25), and our bound outperforms the classical bound precisely when 4‖E‖ < δ_p (i.e., ‖E‖/δ_p < .25).

| ‖E‖/δ_p | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 |
| --------------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| our bound/true error        | 90.17 | 88.27 | 87.02 | 89.83 | 89.44 | 87.81 | 88.39 | 89.29 | 87.08 | 87.26 |
| our bound/classical bound   | 0.20 | 0.40 | 0.60 | 0.79 | 0.98 | 1.17 | 1.36 | 1.53 | 1.70 | 1.88 |

D Extensions of Theorem 2.1 and Theorem 2.2 to the symmetric matrices

In this section, we extend Theorem 2.1 and Theorem 2.2 to the setting where A is a symmetric matrix. These extensions are naturally important since the data in real-world applications is often arbitrary, making it natural for the eigenvalues of A to span both signs. While singular value decomposition (SVD) could be used to apply Theorem 2.1 or Theorem 2.2, singular value gaps are typically small. By working directly with eigenvalues, we exploit the fact that the eigenvalue gap δ_k = λ_k − λ_{k+1} is significantly large when λ_k · λ_{k+1} < 0.

D.1 Extension of Theorem 2.1 to the symmetric matrices

To simplify the presentation, we assume that the eigenvalues (singular values) are different, so the eigenvectors (singular vectors) are well-defined (up to signs). However, our results hold for matrices with multiple eigenvalues. Let A, E be n × n real symmetric matrices, and let 1 ≤ p ≤ n denote the rank of approximation. Let λ_k be the k-th largest eigenvalue of A and u_k be the corresponding orthonormal eigenvector. Let \tilde{A} := A + E. Let A_p, \tilde{A}_p denote the best rank-p approximations of A and \tilde{A} respectively. Define k, 1 ≤ k ≤ p, such that the set of the top p singular values corresponds to {λ_{π(1)}, …, λ_{π(p)}} = {λ_1, …, λ_k > 0 ≥ λ_{n-(p-k)+1}, …, λ_n}. In other words, the p-th singular value of A is either λ_k or |λ_{n-(p-k)+1}|. Let δ_i := λ_i − λ_{i+1}, for i ∈ [n − 1]. Theorem 2.1 is extended to the following result.

Theorem D.1 (Extension of Theorem 2.1 to the symmetric matrices). If 4‖E‖ ≤ min{δ_k, δ_{n-(p-k)}} and 2‖E‖ < σ_p − σ_{p+1}, then
‖\tilde{A}_p − A_p‖ ≤ 6‖E‖ ( log(6σ_1/δ_k) + λ_k/δ_k + log(6σ_1/δ_{n-(p-k)}) + |λ_{n-(p-k)+1}|/δ_{n-(p-k)} ).

Note that when A is not PSD, { | \tilde{λ}_1 |, …, | \tilde{λ}_k |, | \tilde{λ}_{n-(p-k)+1} |, …, | \tilde{λ}_n | } may not correspond to the p leading singular values of \tilde{A}. This issue is resolved by enforcing the singular-value gap condition σ_p − σ_{p+1} > 2‖E‖. Indeed, by Weyl’s inequality, given σ_p − σ_{p+1} > 2‖E‖, we have
\tilde{λ}_k ≥ λ_k − ‖E‖ ≥ σ_p − ‖E‖ = σ_{p+1} + δ − ‖E‖
≥ |λ_{n-(p-k)}| + δ − ‖E‖ ≥ | \tilde{λ}_{n-(p-k)} | + δ − 2‖E‖ > | \tilde{λ}_{n-(p-k)} |,
here δ = σ_p − σ_{p+1}. By a similar argument, we also have | \tilde{λ}_{n-(p-k)+1} | > \tilde{λ}_{k+1}. Therefore,
{ \tilde{λ}_{π(1)}, \tilde{λ}_{π(2)}, …, \tilde{λ}_{π(p)} } = { \tilde{λ}_1 ≥ \tilde{λ}_2 ≥ … ≥ \tilde{λ}_k > 0 ≥ \tilde{λ}_{n-(p-k)+1} ≥ \tilde{λ}_{n-(p-k)+2} ≥ … ≥ \tilde{λ}_n },
as we want. Note that the gap condition of eigenvalues cannot guarantee this fact. For example, consider the following matrices
A = ( 30√n   0
      0   −28√n ),  E = ( −2√n   0
                            0   −2√n ), then \tilde{A} = ( 28√n   0
                                                     0  −30√n ).
Here, clearly, S = {1}, \tilde{S} = {1} and |λ_1| is the largest singular value of A, but | \tilde{λ}_1 | is not the largest singular value of \tilde{A} (λ_1 is still the largest eigenvalue).

Proof of Theorem D.1 Let 1 ≤ k ≤ p be a natural number such that
{λ_{π(1)}, λ_{π(2)}, ..., λ_{π(p)}} = {λ_1, λ_2, ..., λ_k > 0 ≥ λ_{n−(p−k)+1}, λ_{n−(p−k)+2}, ..., λ_n}.
Thus, we can split A_p as A_k + B_{p−k}, in which
B_{p−k} = ∑_{i = n−(p−k)+1}^{n} λ_i u_i u_i^{⊤}.
Similarly, \widetilde{A}_p = \widetilde{A}_k + \widetilde{B}_{p−k}. Therefore,
||\widetilde{A}_p − A_p|| = ||\widetilde{A}_k + \widetilde{B}_{p−k} − A_k − B_{p−k}|| ≤ ||\widetilde{A}_k − A_k|| + ||\widetilde{B}_{p−k} − B_{p−k}||.
Applying the contour bootstrapping argument on ||\widetilde{A}_k − A_k|| with contour Γ^{[1]} and on ||\widetilde{B}_{p−k} − B_{p−k}|| with another contour Γ^{[2]} (we define these contours later), we obtain
||\widetilde{A}_k − A_k||/2 ≤ F_1^{[1]} := (1/2π) ∫_{Γ^{[1]}} || z (zI − A)^{-1} E (zI − A)^{-1} || |dz|,
||\widetilde{B}_{p−k} − B_{p−k}||/2 ≤ F_1^{[2]} := (1/2π) ∫_{Γ^{[2]}} || z (zI − A)^{-1} E (zI − A)^{-1} || |dz|,
and hence,
||\widetilde{A}_p − A_p|| ≤ 2 (F_1^{[1]} + F_1^{[2]}).
We set Γ^{[1]} and Γ^{[2]} to be rectangles, whose vertices are
Γ^{[1]}: (a_0, T), (a_1, T), (a_1, −T), (a_0, −T) with a_0 := λ_k − δ_k/2, a_1 := 2σ_1, T := 2σ_1;
and
Γ^{[2]}: (b_0, T), (b_1, T), (b_1, −T), (b_0, −T) with b_0 := λ_{n−(p−k)+1} + δ_{n−(p−k)}/2, b_1 := −2σ_1, T := 2σ_1.
Now, we are going to bound F_1^{[1]}. First, we split Γ^{[1]} into four segments:
• Γ_1 := {(a_0, t) | −T ≤ t ≤ T}.
• Γ_2 := {(x, T) | a_0 ≤ x ≤ a_1}.
• Γ_3 := {(a_1, t) | T ≥ t ≥ −T}.
• Γ_4 := {(x, −T) | a_1 ≥ x ≥ a_0}.

                    Γ₂
        ┌───────────────────────────┐
    Γ₁  │                           │ Γ₃
   ─────┤                           ├───── Re(z) = 0
  λₖ₊₁  │   λₖ        λ₁            │
        └───────────────────────────┘
                    Γ₄

Therefore,
F_1^{[1]} = ∑_{l=1}^4 ∫_{Γ_l} || z (zI − A)^{-1} E (zI − A)^{-1} || |dz|.
Notice that
|| z (zI − A)^{-1} E (zI − A)^{-1} || ≤ ||E|| min_{i∈[n]} |z| / |z − λ_i|^2,
we further obtain
2π F_1^{[1]} ≤ ||E|| (∑_{l=1}^4 N_l),
in which
N_l := ∫_{Γ_l} |z| / (min_i |z − λ_i|^2) |dz| for l = 1, 2, 3, 4.
We use the following lemmas, whose proofs are delayed to the next section.


![Figure 53](image_53.jpg)
<!-- Caption: Appendix, index 53 -->


Lemma D.2. Under the assumption of Theorem D.1,
N_1 \leq \frac{2\pi a_0}{\delta_k} + 4\log\left|\frac{3T}{\delta_k}\right|.

Lemma D.3. Under the assumption of Theorem D.1,
N_3 \leq \frac{\pi a_1}{|a_1-\lambda_1|} + 4\log\left|\frac{3T}{a_1-\lambda_1}\right|.

Lemma D.4. Under the assumption of Theorem D.1,
N_2, N_4 \leq \frac{\sqrt{2(a_1-a_0)}}{T}.

Since p < n, then k + 1 > n - (p - k) + 1 and hence k + 1 \notin \{\pi(1), \ldots, \pi(p)\}. It means |\lambda_{k+1}| \leq \lambda_k. Thus 0 < a_0 < \lambda_k, and hence
N_1 \leq \frac{2\pi\lambda_k}{\delta_k} + 4\log\left|\frac{6\sigma_1}{\delta_k}\right|.

By the setting that a_1 = T = 2\sigma_1,
N_2, N_4 \leq \frac{\sqrt{2a_1}}{T} = \sqrt{2},
N_3 \leq \frac{2\pi\sigma_1}{2\sigma_1-\lambda_1} + 4\log\left|\frac{3T}{a_1-\lambda_1}\right| \leq \frac{2\pi\sigma_1}{\sigma_1} + 4\log\left|\frac{6\sigma_1}{\sigma_1}\right| = 2\pi + 4\log 6.

Thus, using the above estimates, we obtain
F_1^{[1]} \leq \frac{\|E\|}{2\pi}\left(2\pi + 4\log 6 + 2\sqrt{2} + \frac{2\pi\lambda_k}{\delta_k} + 4\log\left|\frac{6\sigma_1}{\delta_k}\right|\right)
\leq \frac{\|E\|}{2\pi}\left(15\log\left|\frac{6\sigma_1}{\delta_k}\right| + \frac{2\pi\lambda_k}{\delta_k}\right) \tag{5}
\leq 3\|E\|\left(\log\left|\frac{6\sigma_1}{\delta_k}\right| + \frac{\lambda_k}{\delta_k}\right).

Applying a similar argument on contour \Gamma^{[2]}, we obtain
F_1^{[2]} \leq 3\|E\|\left(\log\left|\frac{6\sigma_1}{\delta_{n-(p-k)}}\right| + \left|\frac{\lambda_{n-(p-k)+1}}{\delta_{n-(p-k)}}\right|\right). \tag{6}

Combining (4), (5) and (6), we complete our proof.

D.2 Extension of Theorem 2.2 to the symmetric matrices

Let A be a symmetric matrix with eigenvalues \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n, in which \lambda_n is not necessarily positive. Recall the setting from the previous section that 1 \leq k \leq p is the positive integer such that the set of the top p singular values is \{\lambda_{\pi(1)}, \ldots, \lambda_{\pi(p)}\} = \{\lambda_1, \ldots, \lambda_k > 0 \geq \lambda_{n-(p-k)+1}, \ldots, \lambda_n\}. To extend Theorem 2.2, we first generalize the definition of the halving distance r and interaction term x as follows. Let r_1, r_2 respectively be the smallest positive integer satisfying \frac{\lambda_k}{2} \leq \lambda_k - \lambda_{r_1+1} and \frac{|\lambda_{n-(p-k)+1}|}{2} \leq \lambda_{n-r_2+1} - \lambda_{n-(p-k)+1}. Define the “halving distance” r := \max\{r_1, r_2\}. Next, let x_1 := \max_{1\leq i,j\leq r_1} |u_i^\top E u_j| and x_2 := \max_{1\leq i,j\leq r_2} |u_{n-i+1}^\top E u_{n-j+1}|. Define the interaction parameter \bar{x} := \max\{x_1, x_2\}.

Theorem D.5 (Extension of Theorem 2.2 to the symmetric matrices). Assume that 4\|E\| \leq \min\{\delta_k, \delta_{n-(p-k)}\} and 2\|E\| < \sigma_p - \sigma_{p+1}, then
\left\|\tilde{A}_p - A_p\right\| \leq 12\left(\|E\| + r^2\bar{x}\right)\left(\log\left(\frac{6\sigma_1}{\delta_k}\right) + \log\left(\frac{6\sigma_1}{\delta_{n-(p-k)}}\right)\right) + 30r^2\bar{x}\left(\frac{\lambda_k}{\delta_k} + \frac{|\lambda_{n-(p-k)+1}|}{\delta_{n-(p-k)}}\right).

Proof of Theorem D.5. First, we still split (\tilde{A}_p, A_p) into (A_k, B_{p-k}, \tilde{A}_k, \tilde{B}_{p-k}) and apply the contour bootstrapping argument on \left\|\tilde{A}_k - A_k\right\|, \left\|\tilde{B}_{p-k} - B_{p-k}\right\|. We also obtain
\left\|\tilde{A}_p - A_p\right\| \leq 2\left(F_1^{[1]} + F_1^{[2]}\right).

However, we will treat F_1^{[1]}, F_1^{[2]} a bit differently. Indeed,
$$2\pi F_1^{[1]} \leq M_1 + \|E\|(N_2 + N_3 + N_4),$$
in which
$$M_1 := \int_{\Gamma_1} \|z(zI - A)^{-1}E(zI - A)^{-1}\|\,|dz| = \int_{\Gamma_1} \left\|\sum_{1\leq i,j\leq n} \frac{z}{(z-\lambda_i)(z-\lambda_j)}\,u_i u_i^{\top} E\,u_j u_j^{\top}\right\|\!|dz|,$$
and
$$N_l := \int_{\Gamma_l} \frac{|z|}{\min_{i\in[n]} |z-\lambda_i|^2}\,|dz|\quad\text{for } l \in \{2,3,4\}.$$
We additionally use the following lemma (its proof will be delayed in the next section).

Lemma D.6. Under the assumption of Theorem D.5,
$$M_1 \leq r^2 \bar{x}\left(\frac{2\pi a_0}{\delta_k} + 2\log\!\left(\frac{6\sigma_1}{\delta_k}\right)\right) + \left(20 + \frac{4\pi}{\log 10}\right)\|E\|\log\!\left(\frac{10\sigma_1}{\delta_k}\right).$$
Together with the estimates for N_2, N_3, N_4 from the previous section, we obtain
\begin{align*}
2\pi F_1^{[1]} &\leq r^2 \bar{x}\left(\frac{2\pi\lambda_k}{\delta_k} + 2\log\!\left(\frac{3T}{\delta_k}\right)\right) + \left(20 + \frac{4\pi}{\log 10}\right)\|E\|\log\!\left(\frac{5T}{\delta_k}\right) + \|E\|\left(2\sqrt{2} + 2\pi + 4\log 6\right) \\
&\leq r^2 \bar{x}\left(\frac{2\pi\lambda_k}{\delta_k} + 2\log\!\left(\frac{6\sigma_1}{\delta_k}\right)\right) + \left(20 + \frac{4\pi}{\log 10}\right)\|E\|\log\!\left(\frac{10\sigma_1}{\delta_k}\right) + \frac{2\sqrt{2}+2\pi+4\log 6}{\log 10}\|E\|\log\!\left(\frac{10\sigma_1}{\delta_k}\right).
\end{align*}
Thus,
$$F_1^{[1]} \leq 6\left(\|E\|\log\!\left(\frac{10\sigma_1}{\delta_k}\right) + r^2 \bar{x}\frac{\lambda_k}{\delta_k} + r^2 \bar{x}\log\!\left(\frac{10\sigma_1}{\delta_k}\right)\right). \tag{7}$$
Similarly,
$$F_1^{[2]} \leq 6\left(\|E\|\log\!\left(\frac{10\sigma_1}{\delta_{n-(p-k)}}\right) + r^2 \bar{x}\frac{|\lambda_{n-(p-k)+1}|}{\delta_{n-(p-k)}} + r^2 \bar{x}\log\!\left(\frac{10\sigma_1}{\delta_{n-(p-k)}}\right)\right). \tag{8}$$
Therefore, combining (4), (7), and (8), we finally obtain
$$\left\|\tilde{A}_p - A_p\right\| \leq 12\left(\|E\|\log\!\left(\frac{36\sigma_1^2}{\delta_k\,\delta_{n-(p-k)}}\right) + r^2 \bar{x}\frac{\lambda_k}{\delta_k} + r^2 \bar{x}\frac{|\lambda_{n-(p-k)+1}|}{\delta_{n-(p-k)}} + r^2 \bar{x}\log\!\left(\frac{36\sigma_1^2}{\delta_k\,\delta_{n-(p-k)}}\right)\right).$$

E Estimating integrals over segments

In this section, we present in detail the integral estimations mentioned in the previous section: Lemma D.2, Lemma D.3, Lemma D.6 (integration over vertical segments); and Lemma D.4 (integration over horizontal segments). We first present a technical lemma, which is used several times in the upcoming sections.

Lemma E.1. Let a, T be positive numbers such that a \leq T. Then,
$$\int_{-T}^{T} \frac{1}{t^2+a^2}\,dt \leq \frac{\pi}{a}.$$

Proof of Lemma E.1. We have
\begin{align*}
\int_{-T}^{T} \frac{1}{t^2+a^2}\,dt &= 2\int_{0}^{T} \frac{1}{t^2+a^2}\,dt \\
&= \frac{2}{a}\arctan(T/a) \\
&\leq \frac{2}{a}\cdot \frac{\pi}{2} = \frac{\pi}{a}.
\end{align*}

E.1 Estimating integrals over vertical segments for interaction-dependent bound

In this Section, we now estimate M_1 — integral over the left vertical segment (prove Lemma D.6) and estimate N_3 — the integral over the right vertical segment (prove Lemma D.3). First, we estimate M_1 as follows.

Using the spectral decomposition (zI - A)^{-1} = \sum_{i=1}^n \frac{u_i u_i^{\top}}{z-\lambda_i}, we can rewrite M_1 as
$$M_1 = \int_{\Gamma_1} \left\|\sum_{1\leq i,j\leq n} \frac{z}{(z-\lambda_i)(z-\lambda_j)}\,u_i u_i^{\top} E\,u_j u_j^{\top}\right\|\!|dz|.$$

Define x_1 := max_{1 ≤ i,j ≤ r_1} |u_i^\top E u_j|. By the triangle inequality, M_1 is at most

∫_{Γ_1} ∥∑_{1 ≤ i,j ≤ r_1} z/[(z−λ_i)(z−λ_j)] u_i u_i^\top E u_j u_j^\top∥ |dz|
+ ∫_{Γ_1} ∥∑_{r_1 < i,j ≤ n} z/[(z−λ_i)(z−λ_j)] u_i u_i^\top E u_j u_j^\top∥ |dz|
+ ∫_{Γ_1} ∥∑_{i ≤ r_1 < j \text{ or } i > r_1 ≥ j} z/[(z−λ_i)(z−λ_j)] u_i u_i^\top E u_j u_j^\top∥ |dz|.

Consider the first term. By the triangle inequality, we have

∫_{Γ_1} ∥∑_{1 ≤ i,j ≤ r_1} z/[(z−λ_i)(z−λ_j)] u_i u_i^\top E u_j u_j^\top∥ |dz|
≤ ∑_{1 ≤ i,j ≤ r_1} ∫_{Γ_1} ∥ z/[(z−λ_i)(z−λ_j)] u_i u_i^\top E u_j u_j^\top ∥ |dz|

= ∑_{1 ≤ i,j ≤ r_1} ∫_{Γ_1} |u_i^\top E u_j| · ∥u_i u_j^\top∥ · |z| / |(z−λ_i)(z−λ_j)| |dz|

≤ ∑_{i,j ≤ r_1} x_1 ∫_{−T}^T √(a_0^2 + t^2) / √(((a_0−λ_i)^2 + t^2)((a_0−λ_j)^2 + t^2)) dt

(since max_{1 ≤ i,j ≤ r_1} |u_i^\top E u_j| ≤ x_1, ∥u_i u_j^\top∥ = 1, and Γ_1 := {z : z = a_0 + it, −T ≤ t ≤ T})

≤ ∑_{i,j ≤ r_1} x_1 ∫_{−T}^T (|a_0| + |t|) / √(((a_0−λ_i)^2 + t^2)((a_0−λ_j)^2 + t^2)) dt.

By the construction of Γ_1, we have

|a_0 − λ_i| ≥ δ_k/2 for all 1 ≤ i ≤ n. (9)

Thus, the right-hand side is at most

r_1^2 x_1 ∫_{−T}^T (|a_0| + |t|) / [t^2 + (δ_k/2)^2] dt
= r_1^2 x_1 ( ∫_{−T}^T |a_0| / [t^2 + (δ_k/2)^2] dt + ∫_0^T 2t / [t^2 + (δ_k/2)^2] dt ). (10)

By Lemma E.1, we have

∫_{−T}^T |a_0| / [t^2 + (δ_k/2)^2] dt ≤ 2π|a_0|/δ_k = 2π a_0/δ_k (since a_0 ≥ 0).

The second integral is estimated by

∫_0^T 2t / [t^2 + (δ_k/2)^2] dt
= ∫_{(δ_k/2)^2}^{T^2 + (δ_k/2)^2} (1/u) du  (u = t^2 + (δ_k/2)^2)

= log( (T^2 + (δ_k/2)^2) / (δ_k/2)^2 )  (11)

= log( (4T^2 + δ_k^2) / δ_k^2 ) ≤ 2 log( 3T/δ_k ).

Therefore,

∫_{Γ_1} ∥∑_{1 ≤ i,j ≤ r_1} z/[(z−λ_i)(z−λ_j)] u_i u_i^\top E u_j u_j^\top∥ |dz|
≤ r_1^2 x_1 ( 2π a_0/δ_k + 2 log(3T/δ_k) ). (12)

Next, we bound the second term as follows:

∫_{Γ_1} ∥∑_{r_1 < i,j ≤ n} z/[(z−λ_i)(z−λ_j)] u_i u_i^\top E u_j u_j^\top∥ |dz|

= ∫_{Γ_1} ∥ z (∑_{r_1 < i ≤ n} u_i u_i^\top/(z−λ_i)) E (∑_{r_1 < j ≤ n} u_j u_j^\top/(z−λ_j)) ∥ |dz|

≤ ∫_{Γ_1} |z| · ∥∑_{r_1 < i ≤ n} u_i u_i^\top/(z−λ_i)∥ · ∥E∥ · ∥∑_{r_1 < j ≤ n} u_j u_j^\top/(z−λ_j)∥ |dz|

≤ ∥E∥ ∫_{Γ_1} |z| / min_{r_1 < i ≤ n} |z−λ_i|^2 |dz|

= ∥E∥ ∫_{−T}^T √(a_0^2 + t^2) / min_{r_1 < i ≤ n} [ (a_0−λ_i)^2 + t^2 ] dt.

Moreover, by the construction of Γ_1 and the definition of r_1,

|a_0 − λ_i| = |(λ_k + λ_{k+1})/2 − λ_i|
≥ |(λ_k + λ_{k+1} − 2λ_{r+1})/2|
≥ (λ_k − λ_{r+1})/2
≥ λ_k/4. (13)

where the second inequality follows the fact i > r_1. Thus, the r.h.s. is at most

||E|| ∫_{-T}^T (√(a_0^2 + t^2)) / (t^2 + (λ_k/4)^2) dt ≤ ||E|| ∫_{-T}^T (a_0 + |t|) / (t^2 + (λ_k/4)^2) dt.

Similar to (10) and (11), we also have

∫_{-T}^T (a_0 + |t|) / (t^2 + (λ_k/4)^2) dt ≤ (4π a_0)/λ_k + log((T^2 + (λ_k/4)^2) / ((λ_k/4)^2))

≤ (4π a_0)/λ_k + log(((2T)^2) / (δ_k^2)) (14)

≤ 4π + 2 log((2T) / (δ_k)) (since a_0 ≤ λ_k).

It follows that

∫_{Γ_1} ||∑_{i>r_1, j>r_1} z / ((z − λ_i)(z − λ_j)) u_i u_i^⊤ E u_j u_j^⊤|| |dz| ≤ ||E|| (4π + 2 log((2T) / (δ_k))). (15)

Now we consider the last term:

∫_{Γ_1} ||∑_{i≤r_1<j, i>r_1≥j} z / ((z − λ_i)(z − λ_j)) u_i u_i^⊤ E u_j u_j^⊤|| |dz| ≤ 2||E|| ∫_{Γ_1} |z| / min_{i≤r_1<j} |(z − λ_i)(z − λ_j)| |dz|.

By (13) and (9), the r.h.s. is at most

2||E|| ∫_{-T}^T |z| / √((t^2 + (δ_k/2)^2)(t^2 + (a_0 − λ_{r_1+1})^2)) dt
= 4||E|| ∫_0^T √(a_0^2 + t^2) / √((t^2 + (δ_k/2)^2)(t^2 + (a_0 − λ_{r_1+1})^2)) dt

≤ 4||E|| ∫_0^T (a_0 + t) / √((t^2 + (δ_k/2)^2)(t^2 + (a_0 − λ_{r_1+1})^2)) dt. (16)

Moreover, ∫_0^T (a_0 + t) / √((t^2 + (δ_k/2)^2)(t^2 + (a_0 − λ_{r_1+1})^2)) dt equals

∫_0^T a_0 dt / √((t^2 + (δ_k/2)^2)(t^2 + (a_0 − λ_{r_1+1})^2)) + ∫_0^T t dt / √((t^2 + (δ_k/2)^2)(t^2 + (a_0 − λ_{r_1+1})^2))

= ∫_0^T a_0 dt / √((t^2 + (δ_k/2)^2)(t^2 + (a_0 − λ_{r_1+1})^2)) + (1/2) log(((T^2 + (δ_k/2)^2)(a_0 − λ_{r_1+1})) / ((T^2 + (a_0 − λ_{r_1+1})^2)(δ_k/2))).

≤ a_0 / max{δ_k/2, a_0 − λ_{r_1+1}} × log((T + √(T^2 + min{δ_k/2, a_0 − λ_{r_1+1}}^2)) / min{δ_k/2, a_0 − λ_{r_1+1}}) 
+ (1/2) log(((T^2 + (δ_k/2)^2)(a_0 − λ_{r_1+1})) / ((T^2 + (a_0 − λ_{r_1+1})^2)(δ_k/2))).

Note that a_0 − λ_{r_1+1} ≥ δ_k/2 and a_0 − λ_{r_1+1} + δ_k/2 = λ_k − λ_{r_1+1} ≥ λ_k/2. Therefore, a_0 − λ_{r_1+1} = max{δ_k/2, a_0 − λ_{r_1+1}} ≥ λ_k/4. We further obtain that ∫_0^T (a_0 + t) / √((t^2 + (δ_k/2)^2)(t^2 + (a_0 − λ_{r_1+1})^2)) dt is at most

(a_0)/(λ_k/4) · log(5T/δ_k) + (1/2) log(2T/δ_k) ≤ 4.5 log(5T/δ_k). (17)

The estimates (16) and (17) together imply that the last term is at most

18||E|| log(5T/δ_k). (18)

Combining (12), (15) and (18), we finally obtain that M_1 is at most

r_1^2 x_1 (2π a_0/δ_k + 2 log(3T/δ_k)) + ||E|| (4π + 2 log(2T/δ_k)) + 18||E|| log(5T/δ_k)

≤ r_1^2 x_1 (2π a_0/δ_k + 2 log(6σ_1/δ_k)) + (20 + 4π/log(10)) ||E|| log(10σ_1/δ_k) (since log(10σ_1/δ_k) ≥ log 10)

≤ r_{x̄}^2 x_{x̄} (2π a_0/δ_k + 2 log(6σ_1/δ_k)) + (20 + 4π/log(10)) ||E|| log(10σ_1/δ_k). (19)

This proves Lemma D.6.

Next, we estimate N3. Notice that
$$N_3 = \int_{\Gamma_3} \frac{|z|}{\min_{i\in[n]} |z-\lambda_i|^2}\,|dz|$$
$$= \int_{-T}^{T} \frac{\sqrt{a_1^2+t^2}}{\min_{i\in[n]} \big[(a_1-\lambda_i)^2+t^2\big]}\,dt \quad \text{(since }\Gamma_3 := \{z \mid z = a_1 + it,\ -T \le t \le T\})$$
$$\le \int_{-T}^{T} \frac{\sqrt{a_1^2+t^2}}{t^2+(a_1-\lambda_1)^2}\,dt$$
$$\le \int_{-T}^{T} \frac{|a_1|}{t^2+(a_1-\lambda_1)^2}\,dt + \int_{-T}^{T} \frac{|t|}{t^2+(a_1-\lambda_1)^2}\,dt$$
$$\le \left|\frac{\pi a_1}{a_1-\lambda_1}\right| + 2\log\!\left(\left(\frac{T}{a_1-\lambda_1}\right)^2 + 1\right) \quad \text{(by Lemma E.1)}$$
$$\le \left|\frac{\pi a_1}{a_1-\lambda_1}\right| + 4\log\left|\frac{3T}{a_1-\lambda_1}\right|.$$
This proves Lemma D.3.

E.2 Estimating integrals over horizontal segments

We are going to bound $N_2, N_4$ — integral over top horizontal segment (prove Lemma D.4). We have
$$N_2 = \int_{\Gamma_2} \frac{|z|}{\min_{i\in[n]} |z-\lambda_i|^2}\,|dz|$$
$$= \int_{a_0}^{a_1} \frac{\sqrt{x^2+T^2}}{\min_{i\in[n]} \big((x-\lambda_i)^2+T^2\big)}\,dx \quad \text{(since }\Gamma_2 := \{z \mid z = x + iT,\ a_0 \le x \le a_1\})$$
$$\le \int_{a_0}^{a_1} \frac{\sqrt{2}\,T}{T^2}\,dx \quad \text{(since }x \le a_1 \le T)$$
$$= \frac{\sqrt{2}\,|a_1-a_0|}{T}.$$
By similar arguments, we also obtain
$$N_4 \le \frac{\sqrt{2}\,|a_1-a_0|}{T}.$$
These estimates on $N_2, N_4$ prove Lemma D.4.

E.3 Estimating integrals over vertical segments for non-interaction bound

In this section, we estimate $N_1$, proving Lemma D.2. The estimation of $N_3$ follows the case of the interaction-dependent bound at the end of Section E.1.
$$N_1 = \int_{\Gamma_1} \frac{|z|}{\min_{i\in[n]} |z-\lambda_i|^2}\,|dz|$$
$$= \int_{-T}^{T} \frac{\sqrt{a_0^2+t^2}}{\min_{i\in[n]} \big[(a_0-\lambda_i)^2+t^2\big]}\,dt \quad \text{(since }\Gamma_1 := \{z \mid z = a_0 + it,\ -T \le t \le T\})$$
$$\le \int_{-T}^{T} \frac{\sqrt{a_0^2+t^2}}{t^2+(\delta_k/2)^2}\,dt \quad \text{(by (9))}$$
$$\le \int_{-T}^{T} \frac{|a_0|}{t^2+(\delta_k/2)^2}\,dt + \int_{-T}^{T} \frac{|t|}{t^2+(\delta_k/2)^2}\,dt$$
$$\le \left|\frac{2\pi a_0}{\delta_k}\right| + 2\log\!\left(\left(\frac{2T}{\delta_k}\right)^2 + 1\right) \quad \text{(by Lemma E.1)}$$
$$\le \left|\frac{2\pi a_0}{\delta_k}\right| + 4\log\left|\frac{3T}{\delta_k}\right|.$$
This proves Lemma D.2.

F Perturbation of matrix functionals — Theorem 2.3

In this section, we complete the delayed proof of Theorem 2.3. By Remark 3.2, to prove Theorem 2.3, we need to show that
$$2\pi F_1(1) := \int_{\Gamma} \left\|(zI - A)^{-1}E(zI - A)^{-1}\right\|\,|dz| = 4\pi \frac{\|E\|}{\delta_p}.$$

In which the contour Γ is set to be a rectangle with vertices
$$(x_0,T),(x_1,T),(x_1,-T),(x_0,-T), \text{ where } x_0:=\lambda_p-\delta_p/2,\; x_1:=2\lambda_1,\; T:=2\lambda_1.$$

We split Γ into four segments:
$$\Gamma_1:=\{(x_0,t)\mid -T\le t\le T\},\qquad \Gamma_2:=\{(x,T)\mid x_0\le x\le x_1\},$$
$$\Gamma_3:=\{(x_1,t)\mid T\ge t\ge -T\},\qquad \Gamma_4:=\{(x,-T)\mid x_1\ge x\ge x_0\}.$$

Therefore,
$$\int_{\Gamma}\!\bigl\|(zI-A)^{-1}E(zI-A)^{-1}\bigr\|\,|dz|
=\sum_{i=1}^{4} M_i, \tag{20}$$
in which
$$M_i:=\int_{\Gamma_i}\!\bigl\|(zI-A)^{-1}E(zI-A)^{-1}\bigr\|\,|dz| \quad \text{for } i\in\{1,2,3,4\}.$$

By a similar strategy from the previous section, we bound $M_1$ as follows. Notice that
$$\bigl\|(zI-A)^{-1}E(zI-A)^{-1}\bigr\|\le
\frac{\|E\|}{\min_{i\in[n]}|z-\lambda_i|^2}.$$
Therefore, $M_1$ is at most
$$\|E\|\int_{\Gamma_1}\frac{1}{\min_{i\in[n]}|z-\lambda_i|^2}\,|dz|
\le \|E\|\int_{\Gamma_1}\frac{1}{|z-\lambda_p|^2}\,|dz| \quad
\text{(since $\lambda_p$ is closest to $\Gamma_1$ among all eigenvalues of $A$)}$$
$$=\|E\|\int_{-T}^{T}\frac{1}{(\delta_p/2)^2+t^2}\,dt
\quad\text{(by definition $\Gamma_1:=\{(x_0,t)\mid -T\le t\le T\}$ and $|x_0-\lambda_p|=\delta_p/2$)}$$
$$\le \frac{2\pi\|E\|}{\delta_p}\quad\text{(by Lemma E.1)}.$$

Next, we bound $M_3$ as follows:
$$M_3\le \|E\|\int_{\Gamma_3}\frac{1}{\min_{i\in[n]}|z-\lambda_i|^2}\,|dz|
= \|E\|\int_{-T}^{T}\frac{1}{\min_{i\in[n]}\bigl((x_1-\lambda_i)^2+t^2\bigr)}\,dt \quad
\text{(since $\Gamma_3=\{z\mid z=x_1+it,-T\le t\le T\}$)}$$
$$= \|E\|\int_{-T}^{T}\frac{1}{t^2+(x_1-\lambda_1)^2}\,dt
\le \frac{\pi\|E\|}{|x_1-\lambda_1|}\quad\text{(by Lemma E.1)}
= \frac{\pi\|E\|}{\lambda_1}\quad\text{(since $x_1=2\lambda_1$).}$$

Next, we estimate $M_2$ as
$$M_2 \le \int_{\Gamma_2}\frac{1}{\min_{i\in[n]}|z-\lambda_i|^2}\,\|E\|\,|dz|
= \|E\|\int_{\Gamma_2}\frac{1}{\min_{i\in[n]}|z-\lambda_i|^2}\,|dz|.$$
Moreover, since $\Gamma_2=\{z\mid z=x+iT,\,x_0\le x\le x_1\}$,
$$\int_{\Gamma_2}\frac{1}{\min_{i\in[n]}|z-\lambda_i|^2}\,|dz|
= \int_{x_0}^{x_1}\frac{1}{\min_{i\in[n]}\bigl((x-\lambda_i)^2+T^2\bigr)}\,dx
\le \int_{x_0}^{x_1}\frac{1}{T^2}\,dx
= \frac{|x_1-x_0|}{T^2}.$$
Therefore,
$$M_2 \le \frac{\|E\|\cdot |x_1-x_0|}{T^2}
\le \frac{\|E\|\lambda_1}{4\lambda_1^2}
= \frac{\|E\|}{4\lambda_1}.$$
Similarly, we also obtain that $M_4 \le \frac{\|E\|}{4\lambda_1}$.

These estimates on $M_1,M_2,M_3,M_4$ and Equation (20) imply
$$\int_{\Gamma}\!\bigl\|(zI-A)^{-1}E(zI-A)^{-1}\bigr\|\,|dz|
= \frac{2\pi\|E\|}{\delta_p} + \Bigl(\pi+\frac{1}{4}+\frac{1}{4}\Bigr)\frac{\|E\|}{\lambda_1}
\le 4\pi\,\frac{\|E\|}{\delta_p}.$$

The last inequality follows from the trivial fact that $\lambda_1>\delta_p$ for any PSD matrix $A$. We complete the proof.

G Some classical perturbation bounds

This section recalls standard results referenced in Section 2, Section 3, and Section A.

Theorem G.1 (Eckart–Young–Mirsky bound [16]). Let A, \tilde{A} \in \mathbb{R}^{n \times n}, and let A_p, \tilde{A}_p denote their respective best rank-p approximations. Set E := \tilde{A} - A. Then,
||\tilde{A}_p - A_p|| \le 2(\sigma_{p+1} + ||E||),
where \sigma_{p+1} is the (p+1)st singular value of A.

Theorem G.2 (Weyl’s inequality [46]). Let A, E \in \mathbb{R}^{n \times n} be symmetric, and define \tilde{A} := A + E. Then, for any 1 \le i \le n,
|\tilde{\lambda}_i - \lambda_i| \le ||E|| and |\tilde{\sigma}_i - \sigma_i| \le ||E||,
where \lambda_i, \tilde{\lambda}_i are the ith eigenvalues of A and \tilde{A}, and \sigma_i, \tilde{\sigma}_i are the corresponding singular values.

H Notation

This section summarizes the key notations used throughout the paper. Let A, E be symmetric n \times n matrices, and define the perturbed matrix \tilde{A} := A + E. Let f be an entire function, and let s \in \mathbb{N}.

| Symbol                                                   | Definition                                                                                              |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| n                                                        | Dimension of A, \tilde{A}                                                                               |
| p                                                        | Target rank parameter                                                                                   |
| A_p                                                      | Best rank-p approximation of A                                                                          |
| \tilde{A}_p                                              | Best rank-p approximation of \tilde{A}                                                                  |
| \lambda_1 \ge \cdots \ge \lambda_n                       | Eigenvalues of A in descending order                                                                    |
| \tilde{\lambda}_1 \ge \cdots \ge \tilde{\lambda}_n       | Eigenvalues of \tilde{A} in descending order                                                            |
| \sigma_1 \ge \cdots \ge \sigma_n                         | Singular values of A in descending order                                                                |
| \delta_i for i \in [n-1]                                 | ith eigengap: \delta_i := \lambda_i - \lambda_{i+1}                                                     |
| u_i for i \in [n]                                        | Eigenvector of A corresponding to \lambda_i                                                             |
| \tilde{u}_i for i \in [n]                                | Eigenvector of \tilde{A} corresponding to \tilde{\lambda}_i                                             |
| \text{sr}(A)                                             | Stable rank: \text{sr}(A) := ||A||_F^2/||A||^2 (p. 22)                                                  |
| Halving distance r                                       | Smallest integer such that \lambda_{p/2} \ge \lambda_{r+1} (p. 3, Thm. 2.2)                             |
| Interaction term x                                       | x := \max_{1 \le i,j \le r} |u_i^\top E u_j| (p. 3, Thm. 2.2)                                          |
| f_p(A)                                                   | f_p(A) := \sum_{i=1}^p f(\lambda_i) u_i u_i^\top (p. 4, Thm. 2.3)                                       |
| f_p(\tilde{A})                                           | f_p(\tilde{A}) := \sum_{i=1}^p f(\tilde{\lambda}_i) \tilde{u}_i \tilde{u}_i^\top (p. 4, Thm. 2.3)       |
| \Gamma                                                   | Contour enclosing {\lambda_1, \ldots, \lambda_p} (p. 5)                                                 |
| F(f)                                                     | \frac{1}{2\pi} \int_\Gamma ||f(z)[(zI - \tilde{A})^{-1} - (zI - A)^{-1}]||\, |dz| (p. 5, Eq. (2))       |
| F_s(f)                                                   | \frac{1}{2\pi} \int_\Gamma ||f(z)(zI - A)^{-1}[E(zI - A)^{-1}]^s||\, |dz| (p. 6)                        |
| F_1(f)                                                   | \frac{1}{2\pi} \int_\Gamma ||f(z)(zI - A)^{-1}E(zI - A)^{-1}||\, |dz| (p. 6, Lem. 3.1)                  |
| F(z)                                                     | \frac{1}{2\pi} \int_\Gamma ||z[(zI - \tilde{A})^{-1} - (zI - A)^{-1}]||\, |dz| (p. 6)                   |
| F_1(z)                                                   | \frac{1}{2\pi} \int_\Gamma ||z(zI - A)^{-1}E(zI - A)^{-1}||\, |dz| (p. 6)                                |
| ||\cdot||                                                | Spectral norm                                                                                           |
| ||\cdot||_F                                              | Frobenius norm                                                                                          |
| EYM bound                                                | Eckart–Young–Mirsky bound                                                                               |
| M–V bound                                                | Mangoubi–Vishnoi bound                                                                                  |
| PSD                                                      | Positive semidefinite                                                                                   |