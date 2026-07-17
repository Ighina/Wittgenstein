
## Abstract


High-Dimensional Calibration from Swap Regret

Maxwell Fishelson*
maxfish@mit.edu

Noah Golowich†
nzg@mit.edu

Mehryar Mohri‡
mohri@google.com

Jon Schneider§
jschnei@google.com

Abstract

We study the online calibration of multi-dimensional forecasts over an arbitrary convex set $\mathcal{P} \subset \mathbb{R}^d$ relative to an arbitrary norm $\|\cdot\|$. We connect this with the problem of external regret minimization for online linear optimization, showing that if it is possible to guarantee $O(\sqrt{\rho T})$ worst-case regret after $T$ rounds when actions are drawn from $\mathcal{P}$ and losses are drawn from the dual $\|\cdot\|_*$ unit norm ball, then it is also possible to obtain $\epsilon$-calibrated forecasts after $T = \exp(O(\rho/\epsilon^2))$ rounds. When $\mathcal{P}$ is the $d$-dimensional simplex and $\|\cdot\|$ is the $\ell_1$-norm, the existence of $O(\sqrt{T \log d})$-regret algorithms for learning with experts implies that it is possible to obtain $\epsilon$-calibrated forecasts after $T = \exp(O(\log d/\epsilon)) = d^{O(1/\epsilon)}$ rounds, significantly improving upon the recent result of [Pen25].

Interestingly, our algorithm obtains this guarantee without requiring access to any online linear optimization subroutine or knowledge of the optimal rate $\rho$—in fact, our algorithm is identical for every setting of $\mathcal{P}$ and $\|\cdot\|$. Instead, we show that the optimal regularizer for the above OLO problem can be used to upper bound the above calibration error by a swap regret, which we then minimize by running the recent TreeSwap algorithm ([DDFG24, PR24]) with Follow-The-Leader as a subroutine. The resulting algorithm is highly efficient, running in time O(L · log T) per round, and plays a distribution over simple averages of past observations in each round, requiring only constant memory per tree node.

Finally, we prove that any online calibration algorithm that guarantees $\epsilon T$ $\ell_1$-calibration error over the $d$-dimensional simplex requires $T \geq \exp(\Omega(1/\epsilon))$ (assuming $d \geq \text{poly}(1/\epsilon)$). This strengthens the corresponding $d^{\Omega(\log 1/\epsilon)}$ lower bound of [Pen25], and shows that an exponential dependence on $1/\epsilon$ is necessary. Moreover, our lower bound is tight up to constant factors in the exponent, matching our upper bound.


## Introduction


1 Introduction

Consider the problem faced by a forecaster who must report probabilistic predictions for a sequence of events (e.g. whether it will rain or not tomorrow). One of the most common methods to evaluate the quality of such a forecaster is to verify whether they are calibrated: for example, does it indeed rain with probability 40% on days where the forecaster makes this prediction? In addition to calibration being a natural property to expect from predictions, several applications across machine learning, fairness, and game theory require the ability to produce online calibrated predictions [ZME20, GPSW17, HJKRR18, FV97].

When events have binary outcomes, calibration can be quantified by the notion of expected calibration error, which measures the expected distance between a prediction made by a forecaster and the actual empirical probability of the outcome on the days where they made that prediction. In a seminal result by Foster and Vohra [FV98], it was proved that it is possible for an online forecaster to efficiently

*MIT.

†MIT.

‡Courant Institute of Mathematical Sciences, New York.

§Independent Researcher.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

guarantee a sublinear calibration error of O(T^{2/3}) against any adversarial sequence of T binary events. Equivalently, this can be interpreted as requiring at most O(\epsilon^{-3}) rounds of forecasting to guarantee an \epsilon per-round calibration error on average.

However, many applications require forecasting sequences of multi-dimensional outcomes. The previous definition of calibration error easily extends to the multi-dimensional setting where predictions and outcomes belong to a d-dimensional convex set \mathcal{P} \subset \mathbb{R}^d. Specifically, if a forecaster makes a sequence of predictions p_1, p_2, \ldots, p_T \in \mathcal{P} for the outcomes y_1, y_2, \ldots, y_T \in \mathcal{P}, their \|\cdot\|-calibration error (for any norm \|\cdot\| over \mathbb{R}^d) is given by
\[
\text{Cal}_T^{\|\cdot\|} = \sum_{t=1}^T \|p_t - \nu_{p_t}\|
\]
where \nu_{p_t} is the average of the outcomes y_t on rounds where the learner predicted p_t.

The algorithm of Foster and Vohra extends to the multidimensional calibration setting, but at the cost of producing bounds that decay exponentially in the dimension d. In particular, their algorithm only guarantees that the forecaster achieves an average calibration error of \epsilon after (1/\epsilon)^{\Omega(d)} rounds. As demonstrated by [ZKL23], until recently, no known algorithm achieved a sub-exponential dependence on d in any non-trivial instance of multi-dimensional calibration, despite extensive efforts in the learning theory community.

In 2025, [Pen25] presented a new algorithm for high-dimensional calibration, demonstrating that it is possible to obtain \ell_1-calibration rates of \epsilon T in d^{O(1/\epsilon^2)} rounds for predictions over the d-dimensional simplex (i.e., multi-class calibration). In particular, this is the first known algorithm achieving polynomial calibration rates in d for fixed constant \epsilon. [Pen25] complements this with a lower bound, showing that in the worst case d^{\Omega(\log(1/\epsilon))} rounds are necessary to obtain this rate (implying that a fully polynomial bound \text{poly}(d, 1/\epsilon) is impossible).

## 1.1 Our results

Although the algorithm of [Pen25] is simple to describe, its analysis is fairly nuanced and tailored to \ell_1-calibration over the simplex (e.g., by analyzing the KL divergence between predictions and distributions of historical outcomes). We present a very similar algorithm (TreeCal) for multidimensional calibration over an arbitrary convex set \mathcal{P} \subset \mathbb{R}^d, but with a simple, unified analysis that provides simultaneous guarantees for calibration with respect to any norm \|\cdot\|. In particular, we prove the following theorem.

**Theorem 1.1** (Informal restatement of Corollary C.5). Fix any convex set \mathcal{P} (with no restrictions on symmetry or other geometric properties) and any norm \|\cdot\| (including non-standard norms). Assume there exists a function R : \mathcal{P} \to \mathbb{R} that is 1-strongly-convex with respect to \|\cdot\| and has range (\max_{x \in \mathcal{P}} R(x) - \min_{x \in \mathcal{P}} R(x)) at most \rho. Then TreeCal guarantees that the calibration error of its predictions is bounded by \text{Cal}_T^{\|\cdot\|} \leq \epsilon T for T \geq (\text{diam}_{\|\cdot\|}(\mathcal{P})/\epsilon)^{O(\rho/\epsilon^2)} for all possible choices of \mathcal{P} and \|\cdot\|.

Interestingly, the function R(p) and parameter \rho appearing in the statement of Theorem 1.1 have an independent learning-theoretic interpretation: if we consider the online linear optimization problem where a learner plays actions in \mathcal{P} and the adversary plays linear losses that are unit bounded in the dual norm \|\cdot\|_*, then it is possible for the learner to guarantee a regret bound of at most O(\sqrt{\rho T}) by playing Follow-The-Regularized-Leader (FTRL) with R(p) as a regularizer. In fact, since universality results for mirror descent guarantee that some instantiation of FTRL achieves near-optimal rates for online linear optimization (as long as the action and loss sets are centrally convex) [SST11, GSJ24], this allows us to relate the performance of Theorem 3.1 directly to what rates are possible in online linear optimization.

**Corollary 1.2** (Informal restatement of Corollary C.6). Let \mathcal{P} \subseteq \mathbb{R}^d be a centrally symmetric convex set, and let \mathcal{L} = \{y \in \mathbb{R}^d \mid \|y\|_* \leq 1\} for some norm \|\cdot\|. Then if there exists an algorithm for online linear optimization with action set \mathcal{P} and loss set \mathcal{L} that incurs regret at most O(\sqrt{\rho T}), TreeCal guarantees that the calibration error of its predictions is bounded by \text{Cal}_T^{\|\cdot\|} \leq \epsilon T for T \geq (\text{diam}_{\|\cdot\|}(\mathcal{P})/\epsilon)^{O(\rho/\epsilon^2)}.

Theorem 1.1 and its corollary allow us to immediately recover several existing and novel bounds on calibration error in a variety of settings:

• When ℘ is the d-simplex Δ_d and ||·|| is the ℓ1-norm, the existence of the negative entropy regularizer R(x) = ∑_{i=1}^d x_i log x_i (which is 1-strongly convex w.r.t. the ℓ1 norm with range ρ = log d) implies that the ℓ1 calibration error of TreeCal is at most (1/ε)^{O(log d/ε^2)} = d^{O(1/ε^2)}. This recovers the result of [Pen25].

• When ℘ is the ℓ2 ball and ||·|| is the ℓ2 norm, the Euclidean regularizer (R(x) = ||x||^2) implies a calibration bound of (1/ε)^{O(1/ε^2)} (notably, this bound is completely independent of d, unlike all prior work which had at least logarithmic dependence on dimension).

It should be emphasized here that running TreeCal does not require any online linear optimization subroutine. While the theoretical analysis uses regularizers R(x) and ranges ρ, the algorithm itself has no functional dependence on any specific ||·|| and achieves ||·||-calibration at the above rate (Theorem 1.1) for all ||·|| simultaneously, with parameter choices that can be set adaptively. The TreeCal algorithm represents a novel approach that, while sharing some structural similarities with [Pen25], introduces several key algorithmic innovations. Both algorithms initialize a tree of sub-forecasters and at each round play a uniform combination of some subset of them (see Figure 1), but TreeCal's design is fundamentally different in its treatment of regularization and its ability to handle arbitrary norms.^5
^5 Unlike [Pen25], which requires explicit regularization by mixing predictions with the uniform distribution, TreeCal achieves calibration without any such mixing, representing a significant algorithmic simplification.

The novelty in our analysis stems from the observation that TreeCal is simply a specific instantiation of our novel TreeSwap swap regret minimization algorithm (which generalizes ideas from [BM07]) and can be analyzed directly in this way. While [DDFG24, PR24] studied related tree-based approaches, our TreeSwap algorithm introduces several key innovations that enable the improved calibration bounds. In particular, our analysis consists of the following steps:

1. First, minimizing calibration error can be reduced to minimizing swap regret, generalizing an idea of [LSS25, FKO^+25]. That is, it is possible to assign the learner loss functions ℓ_t : ℘ → ℝ at each round such that their calibration error is upper bounded by the gap between the total loss they received and the minimal loss they could have received after applying an arbitrary “swap function” π : ℘ → ℘ to their predictions.

   In fact, any strongly convex function R (w.r.t. the norm ||·||) gives rise to one such reduction, by setting the loss function ℓ_t(p) to equal the Bregman divergence D_R(y_t∥p).

2. Second, the TreeSwap algorithm of [DDFG24, PR24] provides a general recipe for converting external regret minimization algorithms into swap regret minimization algorithms. We obtain TreeCal by plugging in the Follow-The-Leader algorithm (the learning algorithm which simply always best responds to the current history) into TreeSwap.

3. Instead of analyzing the swap regret bound of TreeSwap with Follow-The-Leader (which may not have a good enough external regret bound, as discussed in Section 3.3), we instead analyze the swap regret of TreeSwap with Be-The-Leader (the fictitious algorithm that best responds to the current history, including the current round). Though it is not possible to actually implement Be-The-Leader due to its clairvoyance, we use it as a tool for analysis. We then relate the calibration error of TreeSwap with Be-The-Leader to that of TreeSwap with Follow-The-Leader using the fact that Be-The-Leader and Follow-The-Leader make similar predictions.

In the above step 1, we will choose R to be ||·||-norm 1-strongly convex, which guarantees that D_R(y∥p) ≥ ||y − p||^2. Going through the analysis, this actually leads to the stronger guarantee that TreeCal minimizes squared-norm calibration error.

Theorem 1.3 (Informal restatement of Theorem 3.1). Fix a convex set ℘ and a norm ||·||. Assume there exists a function R : ℘ → ℝ that is 1-strongly-convex with respect to ||·|| and has range (max_{x∈℘} R(x) − min_{x∈℘} R(x)) at most ρ. Then TreeCal guarantees that the calibration error of its predictions is bounded by Cal_T^{||·||^2} ≤ ε T for T ≥ (diam_{||·||}(℘)/√ε)^{O(ρ/ε)}.

Note here we have only singly-exponential dependence on 1/ε. We arrive at Theorem 1.1 as a corollary of this result by simply applying Cauchy–Schwarz. Finally, we strengthen the lower bound of [Pen25] by showing an exponential dependence on 1/ε is necessary.

Theorem 1.4 (Informal restatement of Theorem 4.3). There is a sufficiently small constant c > 0 so that the following holds. Fix any ε > 0, d ∈ ℕ. Then for any T ≤ exp(c · min{d^{1/14}, ε^{-1/6}}), there is an oblivious adversary producing a sequence of outcomes so that any learning algorithm must incur ℓ1-calibration error Cal_T^{||·||_1} ≥ ε · T.

Unlike the lower bound of [Pen25], this lower bound requires no specialized construction. Instead, it follows from the original observation of [FV98] that any algorithm for online calibration can be used to construct an algorithm for swap regret minimization by simply best responding to a sequence of calibrated predictions of the adversary’s losses. The existing lower bound for swap regret in [DFG+24] then immediately precludes the existence of sufficiently strong calibration bounds (e.g., of the form d^{O(\log 1/\epsilon)}, which was still allowed by the work of [Pen25]).

Using a similar technique, in Theorem D.2, we show a similar lower bound for \ell_2 calibration, namely that \exp(\Omega(\min\{d^{1/14}, \epsilon^{-1/7}\})) time steps are needed to achieve \ell_2 calibration error at most \epsilon \cdot T. For d \ge \epsilon^{-2}, this bound is tight up to a polynomial in the exponent.

We discuss additional related work in the appendix.


## Preliminaries


2 Setup

For a positive integer n, we let [0 : n - 1] denote the sequence 0, 1, \ldots, n - 1, and [n] denote the sequence 1, 2, \ldots, n. We say a convex set S \subseteq \mathbb{R}^d is centrally symmetric if s \in S \Leftrightarrow -s \in S for all s \in \mathbb{R}^d. A norm \|\cdot\| is a function corresponding to a convex, bounded, centrally symmetric set S of the form \|s\| = \inf\{c \in \mathbb{R}_{\ge 0} \mid s \in cS\}. The corresponding dual norm is defined \|v\|_* = \sup\{\langle s, v \rangle \mid \|s\| \le 1\}.

2.1 Calibration

We consider the following setting of multi-dimensional calibration. Positive integers d \in \mathbb{N} representing the number of dimensions and T \in \mathbb{N} representing the number of rounds are given. We let \mathcal{P} \subset \mathbb{R}^d denote a bounded convex subset of \mathbb{R}^d. An adversary and a learning algorithm interact for a total of T time steps; at each time step t \in [T]:

• The learning algorithm chooses a distribution<sup>6</sup> \mathbf{x}_t \in \Delta(\mathcal{P}) with finite support.
• The adversary observes \mathbf{x}_t and chooses an outcome y_t \in \mathcal{P}.

In order for the learner to be calibrated, we would like the average outcome conditional on the learner making a specific prediction p to be “close” to p. We formalize this as follows. For a point p \in \mathcal{P}, we define \nu_p to be the average outcome conditioned on the learner predicting p, that is:

\nu_p := \frac{\sum_{t=1}^T \mathbf{x}_t(p) \cdot y_t}{\sum_{t=1}^T \mathbf{x}_t(p)}.
(1)

Fix a distance measure D : \mathcal{P} \times \mathcal{P} \to \mathbb{R}_{\ge 0}, namely an arbitrary non-negative valued function on \mathcal{P} \times \mathcal{P}. Given a distance measure D, we define the D-calibration error as follows:

\text{Cal}_T^D(\mathbf{x}_{1:T}, y_{1:T}) := \sum_{p \in \mathcal{P}} \left(\frac{1}{T}\sum_{t=1}^T \mathbf{x}_t(p)\right) \cdot D(\nu_p, p).

In the event that D(p, q) = \|p - q\|, we will write \text{Cal}_T^{\|\cdot\|}(\mathbf{x}_{1:T}, y_{1:T}) = \text{Cal}_T^D(\mathbf{x}_{1:T}, y_{1:T}), and we define \text{Cal}_T^{\|\cdot\|^2}(\mathbf{x}_{1:T}, y_{1:T}) analogously.

2.2 Regret minimization

For a sequence of actions p_1, \cdots, p_T \in \mathcal{P} and loss functions \ell_1, \cdots, \ell_T : \mathcal{P} \to \mathbb{R}, we define

\text{ExtReg}_T(p_{1:T}, \ell_{1:T}) := \sup_{p^* \in \mathcal{P}} \sum_{t=1}^T \left(\ell_t(p_t) - \ell_t(p^*)\right).

<sup>6</sup> Some authors refer to this setting as “pseudo-calibration” or “distributional calibration”, and reserve the term “calibration” for the setting where the learner is required to randomly select a pure forecast p_t \in \mathcal{P} each round instead of a distribution. In Appendix E we describe how to extend our results to this pure-strategy setting of calibration.

For a sequence of distributions $\mathbf{x}_1, \cdots, \mathbf{x}_T \in \Delta(\mathcal{P})$ and loss functions $\ell_1, \cdots, \ell_T : \mathcal{P} \to \mathbb{R}$, we define

$$\text{FullSwapReg}_T(\mathbf{x}_{1:T}, \ell_{1:T}) := \sup_{\pi: \mathcal{P} \to \mathcal{P}} \sum_{t=1}^T \sum_{p \in \mathcal{P}} \mathbf{x}_t(p) \cdot (\ell_t(p) - \ell_t(\pi(p))). \tag{2}$$

Here, we adopt the convention of [FKO⁺25], referring to the latter quantity as Full Swap Regret to emphasize that we consider all swap transformations $\pi : \mathcal{P} \to \mathcal{P}$ (instead of e.g. just linear transformations $\pi$).

Throughout, we consider the performance of regret minimizing algorithms. These algorithms sequentially map loss functions $\ell_1, \cdots, \ell_T$ to actions $p_1, \cdots, p_T$ or action distributions $\mathbf{x}_1, \cdots, \mathbf{x}_T$ with the goal of minimizing the above quantities. We consider the performance of these algorithms on adversarially selected loss functions from a set $\mathcal{L}$. Abusing notation slightly, for an external regret minimizing algorithm $\text{Alg} : \mathcal{L}^T \to \mathcal{P}^T$, we define

$$\text{ExtReg}_T(\text{Alg}) := \sup_{\ell_{1:T} \in \mathcal{L}^T} \text{ExtReg}_T(\text{Alg}(\ell_{1:T}), \ell_{1:T}) \tag{3}$$

and for a full swap regret minimizing algorithm $\text{Alg} : \mathcal{L}^T \to \Delta(\mathcal{P})^T$, we define

$$\text{FullSwapReg}_T(\text{Alg}) := \sup_{\ell_{1:T} \in \mathcal{L}^T} \text{FullSwapReg}_T(\text{Alg}(\ell_{1:T}), \ell_{1:T}).$$

We will denote the $t$th action played by Alg on a sequence of losses $\ell_{1:T}$ by $\text{Alg}_t(\ell_{1:T})$. One important subclass of external regret minimization problems is the setting of online linear optimization (OLO), where all loss functions in $\ell$ are linear. Here we slightly abuse notation and identify $\mathcal{L}$ with a subset of $\mathbb{R}^d$ (with the understanding that an element $\ell \in \mathcal{L}$ refers to the linear loss function $\ell(p) = \langle p, \ell \rangle$). Although we will never actually employ any OLO algorithms themselves, the calibration bounds we obtain will be closely related to optimal regret bounds for instances of OLO (we discuss this further in Section 2.4).

## 2.3 From swap regret to calibration

As noted in [LSS25, FKO⁺25], calibration with a distance measure $D$ that corresponds to a Bregman divergence can be written as a full swap regret with loss functions given by the associated proper scoring rule. Given a convex function $R : \mathcal{P} \to \mathbb{R}$, the Bregman divergence associated to $R$, $D_R : \mathcal{P} \times \mathcal{P} \to \mathbb{R}_{\geq 0}$, is defined as⁷

$$D_R(y \| p) := R(p) - R(y) - \langle \nabla R(y), p - y \rangle.$$

Geometrically, this divergence is defined by taking the hyperplane tangent to $R$ at $p$ and computing the difference in height between $R$ and the hyperplane at $y$ (see Figure 2).

When viewed as a loss function in $p$, the Bregman divergence $D_R(y \| p)$ also has the property that it is a proper scoring rule. This refers to the fact that if $y$ is drawn from some distribution $\mathbf{y} \in \Delta(\mathcal{P})$, the optimal response $p$ (to minimize the expected loss $D_R(y \| p)$) is simply the expectation $\bar{y} = \mathbb{E}_{y \sim \mathbf{y}}[y]$. In particular, we have the following lemma.

Lemma 2.1. For any $\mathbf{y} \in \Delta(\mathcal{P})$ and convex function $R : \mathcal{P} \to \mathbb{R}$, let $\bar{y} = \mathbb{E}_{y \sim \mathbf{y}}[y]$, and $\overline{R(y)} = \mathbb{E}_{y \sim \mathbf{y}}[R(y)]$. For all $p \in \mathcal{P}$, $\mathbb{E}_{y \sim \mathbf{y}}[D_R(y \| p)] = D_R(\bar{y} \| p) + \overline{R(y)} - R(\bar{y})$. In particular, $\ell(p) = \mathbb{E}_{y \sim \mathbf{y}}[D_R(y \| p)]$ is minimized at $p = \bar{y}$ at a value of $\overline{R(y)} - R(\bar{y})$ (Figure 3).

This implies the following connection between full swap regret and calibration.

Lemma 2.2. Fix any convex function $R : \mathcal{P} \to \mathbb{R}$. For any sequence of distributions $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T \in \Delta(\mathcal{P})$ and outcomes $y_1, y_2, \ldots, y_T \in \mathcal{P}$, define the sequence of loss functions $\ell_1, \ell_2, \ldots, \ell_T$ via $\ell_t(p) = D_R(y_t \| p)$. Then,

$$\text{FullSwapReg}_T(\mathbf{x}_{1:T}, \ell_{1:T}) = \text{Cal}_T^{D_R}(\mathbf{x}_{1:T}, y_{1:T}).$$

The proofs of Lemmas 2.1 and 2.2 may be found in Appendix B.

⁷In the event that $R$ is not differentiable, we can replace the $\nabla R(p)$ term with any element of the sub-gradient at $p$. When $\mathcal{P}$ is not open and $p$ is on the boundary, the $\nabla R(p)$ term represents the inward directional gradient.

2.4 Rates and regularization

In order to reduce our general calibration problem to a swap regret minimization problem (via Lemma 2.2), we will need to construct a convex function R whose Bregman divergence upper bounds our distance measure. It turns out that the optimal choice of such a function is closely related to the design of optimal regularizers for online linear optimization. In this section, we describe this functional optimization problem and detail this connection.

We say that a convex function R : 𝒫 → ℝ is α-strongly convex with respect to a given norm ‖·‖ if for any points y, p ∈ 𝒫 it is the case that
R(y) ≤ R(p) + ⟨∇R(p), y − p⟩ + α‖y − p‖².
Equivalently, the Bregman divergence must satisfy D_R(y ∥ p) ≥ α‖y − p‖². Thus, ‖·‖²-calibration error is bounded by D_R-calibration error if R is ‖·‖-norm 1-strongly convex.

Our later analysis will need not only R to be strongly convex with respect to our norm, but for the Bregman divergence to have a small maximal value. Motivated by this, we will say that a convex function R : 𝒫 → ℝ has rate ρ with respect to a given norm ‖·‖ if: (1) R is 1-strongly convex with respect to ‖·‖, and (2) the range of the Bregman divergence is at most ρ, i.e., max_{y,p∈𝒫} D_R(y ∥ p) ≤ ρ. We define Rate(𝒫, ‖·‖) to be the infimum of the rates of all 1-strongly convex functions R : 𝒫 → ℝ.

As mentioned earlier, we call this quantity a “rate” due to its connection with the optimal regret rates for online linear optimization. For a learning algorithm Alg : 𝓛^T → 𝒫^T, we defined (in (3)) ExtReg_T(Alg) to be the worst-case regret against any sequence ℓ_{1:T} of T losses. It is known that for any fixed action set and loss set, the optimal worst-case regret bound is of the form √Rate_OLO(𝒫, 𝓛) · √T + o(√T), for some constant Rate_OLO(𝒫, 𝓛). Formally, we define
Rate_OLO(𝒫, 𝓛) = limsup_{T→∞} inf_{Alg} (1/T) · ExtReg_T(Alg)².

One important class of learning algorithms for online linear optimization is the class of Follow-The-Regularized-Leader (FTRL) algorithms. Each algorithm in this class is specified by a convex “regularizer” function R : 𝒫 → ℝ, and at round t selects the action
p_t = argmin_{p∈𝒫} ∑_{s=1}^{t-1} ⟨p, ℓ_s⟩ + R(p).
The work of [SST11] and [GSJ24] shows that there always exists some instantiation of FTRL which achieves (up to a universal constant factor) the optimal regret rate of √Rate_OLO(𝒫, 𝓛) · √T + o(√T) defined above. Moreover, the optimal regularizer for this instance can be constructed by solving a similar functional optimization problem over strongly convex regularizers R, as described in the following theorem.

Theorem 2.3. Let 𝒫 and 𝓛 be centrally symmetric convex sets. Then, if the function R : 𝒫 → ℝ is 1-strongly convex with respect to the norm ‖·‖_{𝓛*} and has range ρ (i.e., max_{p∈𝒫} R(p) − min_{p∈𝒫} R(p) = ρ), then Rate_OLO(𝒫, 𝓛) ≤ ρ. Conversely, there exists a function R : 𝒫 → ℝ that is 1-strongly convex with respect to ‖·‖_{𝓛*} and has range O(Rate_OLO(𝒫, 𝓛)).

Proof. The first result (that Rate_OLO(𝒫, 𝓛) ≤ ρ) follows from the standard analysis of FTRL — see e.g. Theorem 5.2 in [H^+16]. The converse result follows from Theorem 2 of [GSJ24]. □

Theorem 2.3 allows us to relate the quantity Rate(𝒫, ‖·‖) to the quantity Rate_OLO(𝒫, 𝓛) (where 𝓛 is chosen to be the unit dual norm ball). Note that there is a slight difference in the two functional optimization problems defined above — the one for Rate(𝒫, ‖·‖) asks us to bound the range of the Bregman divergence of R, while the one for Rate_OLO(𝒫, 𝓛) asks us to bound the range of R itself. While these two quantities do not directly bound each other (the negative entropy function R(p) = ∑ p_i log p_i has bounded range over the simplex but unbounded Bregman divergence), we can nonetheless show that optimal solutions to one problem can be used to construct optimal solutions to the other problem of similar quality.

Lemma 2.4. If the action set 𝒫 is centrally symmetric and 𝓛 = {y ∈ ℝ^d | ‖y‖_* ≤ 1} (i.e., the unit ball in the dual norm to ‖·‖), then Rate_OLO(𝒫, 𝓛) = Θ(Rate(𝒫, ‖·‖)).


## Method


3 Main result

We now describe our main algorithm for calibration, TreeCal (Algorithm 1). As we will see, it is equivalent to the TreeSwap algorithm for Full Swap Regret minimization ([DDFG24, PR24]; Algorithm 2), where the loss functions are given by appropriate Bregman divergences as determined by

Lemma 2.2. Moreover, TreeCal is effectively the same as the main algorithm of [Pen25]. However, the perspective that TreeCal can be viewed as a particular instance of TreeSwap (Lemma 3.2) is novel to this work, and it enables us to tackle a much more general set of calibration problems (Theorem 3.1). We first describe the TreeCal and TreeSwap algorithms, then state Theorem 3.1 which establishes our main upper bound for TreeCal, and finally discuss the proof of Theorem 3.1, which uses the TreeSwap algorithm as a tool in the analysis.

## 3.1 Algorithm description

Given some number of rounds $T \in \mathbb{N}$, TreeCal and TreeSwap sequentially produce distributions $\mathbf{x}_1, \cdots, \mathbf{x}_T \in \Delta(\mathcal{P})$. TreeCal receives from the adversary an outcome sequence $y_1, \cdots, y_T \in \mathcal{P}$ whereas TreeSwap receives loss functions $\ell_1, \cdots, \ell_T : \mathcal{P} \to \mathbb{R}$.

To describe how the algorithms use the adversary’s actions to produce the distributions $\mathbf{x}_t$, we need some additional notation. The algorithms take as input parameters $H, L \in \mathbb{N}$ satisfying $H \geq 2$ and $H^{L-1} \leq T \leq H^L$. We index time steps $t \in [T]$ via base-$H$ $L$-tuples: in particular, for $t \in [T]$, we let $t_1, \ldots, t_L \in [0 : H - 1]$ be the base-$H$ representation of $t - 1$; we will write $t - 1 = (t_1 t_2 \cdots t_L)$. For all $0 \leq l \leq L$, for all $k \in [0 : H - 1]^l$, let $\Gamma_k^{(l)} \subset [T]$ represent the interval of times $t$ with prefix $k$. That is, $t \in \Gamma_k^{(l)}$ iff $t_i = k_i$ for all $i \in [1 : l]$. These intervals may be arranged to form an $H$-ary depth-$L$ tree, where the children of $\Gamma_k^{(l)}$ are $\Gamma_{k0}^{(l+1)}, \Gamma_{k1}^{(l+1)}, \cdots, \Gamma_{k,H-1}^{(l+1)}$.^8

Both TreeCal and TreeSwap operate by assigning an action $p_k^{(l)}$ to each node $\Gamma_k^{(l)}$ of the tree, except the root. At time $t$, both algorithms return the uniform distribution over the actions on the root-to-leaf-$t$ path, namely $\mathbf{x}_t := \mathrm{Unif}\left(\left\{p_{t_1}^{(1)}, p_{t_1 t_2}^{(2)}, \cdots, p_{t_1 t_2 \cdots t_L}^{(L)}\right\}\right)$ (see Figure 1). The algorithms differ in how the actions $p_k^{(l)}$ are chosen:

| $p_0^{(1)}$ |   | $p_1^{(1)}$ |   |   |   |
| ------------ | - | ------------ | - | - | - |

| $p_{00}^{(2)}$ | $p_{01}^{(2)}$ | $p_{02}^{(2)}$ | $p_{10}^{(2)}$ | $p_{11}^{(2)}$ |   |   |   |   |   |   |   |   |   |
| --------------- | --------------- | --------------- | --------------- | --------------- | - | - | - | - | - | - | - | - | - |

| $p_{000}^{(3)}$ | $p_{001}^{(3)}$ | $p_{002}^{(3)}$ | $p_{010}^{(3)}$ | $p_{011}^{(3)}$ | $p_{012}^{(3)}$ | $p_{020}^{(3)}$ | $p_{021}^{(3)}$ | $p_{022}^{(3)}$ | $p_{100}^{(3)}$ | $p_{101}^{(3)}$ | $p_{102}^{(3)}$ | $p_{110}^{(3)}$ | $p_{111}^{(3)}$ |   |   |   |   |   |   |   |   |   |   |   |   |   |
| ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | - | - | - | - | - | - | - | - | - | - | - | - | - |

Figure 1: Visualization of the state of TreeCal/TreeSwap at time step $t$ (about half-way through the algorithm). For $H = 3$, we depict the intervals $\Gamma$ of the first three non-root levels of the tree ($l = 1, 2, 3$). Each rectangular node represents an interval, with sibling nodes separated by red lines. We represent the specific time step $t$ via the vertical dashed green line. The yellow intervals it intersects at each level correspond to the nodes on the root-to-leaf-$t$ path. Accordingly, $\mathbf{x}_t$ will be the uniform distribution over the labels $p$ of these yellow intervals. We see that the algorithm has committed to the labels of all intervals that started at or before time $t$, and has yet to label the future intervals.

• TreeCal (Algorithm 1) assigns actions to nodes as follows. For all $1 \leq l \leq L$, $k \in [0 : H - 1]^{l-1}$, $h \in [0 : H - 1]$, at the start of $\Gamma_{kh}^{(l)}$, TreeCal sets $p_{kh}^{(l)}$ to be the average over all $y_t$ that have been observed thus far in the parent interval $\Gamma_k^{(l-1)}$. That is,
$$
p_{kh}^{(l)} = \frac{1}{h H^{L-l}} \sum_{i=0}^{h-1} \sum_{t \in \Gamma_{ki}^{(l)}} y_t
$$
(4)

• The more general TreeSwap algorithm (Algorithm 2) also takes as a parameter an external regret-minimizing algorithm Alg, which operates with horizon of length $H$: we denote the resulting algorithm by TreeSwap.Alg. TreeSwap.Alg associates each internal node of the tree, $\Gamma_k^{(l-1)}$ (with $1 \leq l \leq L$), with an instance Alg, denoted $\mathrm{Alg}_k^{(l-1)}$. The subroutine $\mathrm{Alg}_k^{(l-1)}$ is responsible for choosing the actions $p_{k0}^{(l)}, p_{k1}^{(l)}, \cdots, p_{k(H-1)}^{(l)}$. It does so by responding to the average losses over each of its child intervals. In particular: at the end of

^8 We ignore the truncated branches that exist if $T < H^L$.


![Figure 22](image_22.jpg)
<!-- Caption: Method, index 22 -->


For each child interval Γ_{kh}^{(l)}, we pass Alg_k^{(l-1)} the average loss over that interval. Alg_k^{(l-1)} then outputs the action p_{k(h+1)}^{(l)} assigned to the next child interval.

3.2 Main result

Theorem 3.1 upper bounds the calibration error of TreeCal with respect to the squared norm ||·||^2.

Theorem 3.1 (Main theorem). Let P ⊂ R^d be a bounded convex set and ||·|| be an arbitrary norm. Then, TreeCal (Algorithm 1) guarantees that for an arbitrary sequence of outcomes y_1, …, y_T ∈ P, the ||·||^2 calibration error of its predictions x_1, …, x_T ∈ Δ(P) is bounded as follows:
Cal_T^{||·||^2}(x_{1:T}, y_{1:T}) ≤ ε T for T ≥ (diam(P)/√ε)^{O(Rate(P, ||·||)/ε)}.

It is straightforward to derive from Theorem 3.1 via an application of Jensen’s inequality an upper bound on the calibration error of TreeCal with respect to the (non-squared) norm ||·||, as stated in Theorem 1.1; see Corollary C.5. In Appendix E, we additionally consider a variant of TreeCal which plays pure actions in P (i.e., not distributions) by sampling from the distributions x_t for each t ∈ [T]. We show that the pure calibration error of this variant can be bounded by a similar quantity to that in Theorem 3.1.

3.3 Outline of the proof of Theorem 3.1

Step 1: Reduction from calibration error to swap regret. Let us choose a convex function R : P → R given P, ||·|| as described in Section 2.4. The first step in the proof of Theorem 3.1 is to reduce the problem of minimizing (squared-norm) calibration error to that of minimizing full swap regret for an appropriate sequence of loss functions. In particular, for any sequence x_1, …, x_T ∈ Δ(P) and y_1, …, y_T ∈ P, we have
Cal_T^{||·||^2}(x_{1:T}, y_{1:T}) ≤ Cal_T^{D_R}(x_{1:T}, y_{1:T}) = FullSwapReg_R(x_{1:T}, ℓ_{1:T}), (5)
where ℓ_t : P → R is the loss function given by ℓ_t(p) := D_R(y_t || p); the inequality uses strong convexity of R, and the subsequent equality uses Lemma 2.2.

Step 2: Equivalence with TreeSwap. Thus, it suffices to find an algorithm which minimizes the full swap regret quantity on the right-hand side of (5). Fortunately, the TreeSwap algorithm is known to do exactly this! (See Theorem C.1, from [DDFG24], for a formal statement for the swap regret bound of TreeSwap.) In order to apply the swap regret bound of Theorem C.1, we need to ensure that the TreeCal algorithm is an instantiation of TreeSwap.Alg for an appropriate choice of (a) the loss functions fed as input to TreeSwap and (b) the Alg subroutine. The loss functions have already been defined: given a sequence y_1, …, y_T, recall that we chose ℓ_t(p) := D_R(y_t || p). Moreover, we let the Alg subroutine be given by Follow-the-Leader (FTL), which simply chooses an action at each step minimizing the sum of losses up to the previous time step. The following lemma shows that TreeSwap with the losses ℓ_t and the FTL subroutine produces the same action distributions as TreeCal:
Lemma 3.2. Let P ⊂ R^d be a bounded convex set and let R : P → R be a convex function. For a sequence of loss functions ℓ_1, …, ℓ_H : P → R, define FTL_h(ℓ_{1:H}) = argmin_{p∈P} ∑_{s=1}^{h-1} ℓ_s(p). For all sequences of outcomes y_{1:T} ∈ P^T, the action distributions x_t produced by TreeCal on y_{1:T} equal those produced by TreeSwap.FTL on loss functions ℓ_t(p) = D_R(y_t || p) for all t.
The proof of Lemma 3.2 (given in full in the appendix) is a straightforward consequence of the fact that the Bregman divergence is a proper scoring rule: the action p ∈ P minimizing an average of Bregman divergences D_R(y || p) is simply the average of the constituent points y (Lemma 2.1).

Step 3: Applying the swap regret bound of TreeSwap to FTL. Finally, we want to apply the main result of [DDFG24] (restated as Theorem C.1) to bound the full swap regret for the iterates x_{1:T} produced by TreeSwap.Alg, for an appropriate choice of Alg. The most natural way to do so would be to try to directly apply this result in the case when Alg = FTL (which corresponds to how we actually implement TreeSwap). However, applying this theorem requires an external regret bound on FTL for an arbitrary sequence of losses. While FTL is known to possess strong external regret

Instead, the main idea is to consider the “Be-The-Leader” algorithm BTL, which is the same as FTL but where actions are shifted ahead in time by 1 time step: in particular, the action chosen by BTL at time step $h$ given a sequence $\ell_1, \ell_2, \ldots, \ell_H : \mathcal{P} \to \mathbb{R}$ is $\text{BTL}_h(\ell_{1:H}) = \text{FTL}_{h+1}(\ell_{1:H}) = \operatorname{argmin}_{p \in \mathcal{P}} \sum_{s=1}^h \ell_s(p)$. BTL can be efficiently implemented in practice by using a simple lookahead technique that observes ℓ_h before choosing the action at time step h. Since its regret is always non-positive (i.e., ExtReg_H(BTL) ≤ 0 for any H), if we apply Theorem C.1 to the algorithm TreeSwap.BTL, we get that FullSwapReg_T(TreeSwap.BTL) ≤ ε · T as long as T ≥ H^(O(ρ/ε)) for any choice of H (the arity parameter H used in TreeSwap). This implementability is a key advantage of our approach over prior work. Using (5), this implies that the calibration error of the iterates produced by TreeSwap.BTL can also be bounded above by $\epsilon \cdot T$.

Of course, this result on its own is uninteresting (since BTL is unimplementable, as mentioned above). However, the key insight is that we can show that the actions chosen by TreeSwap.BTL are close to (as measured by the norm $\|\cdot\|$) those chosen by TreeSwap.FTL, which in turn is equivalent to TreeCal (Lemma 3.2). This closeness is an immediate consequence of the fact that the actions chosen by FTL for our loss functions $D_R(y_1\mid\cdot), D_R(y_2\mid\cdot), \ldots$ are simply the empirical average of all actions $y_1, y_2, \ldots \in \mathcal{P}$ of the adversary up to the previous time step.^9 In turn, we can use this closeness to show that the calibration error of TreeSwap.FTL is close to that of TreeSwap.BTL. This latter part of the argument becomes slightly tricky due to the possibility that different nodes of the tree might output the same action $p \in \mathcal{P}$; accordingly, we need to work with a labeled variant of the action set and bound the swap regret over this labeled variant; see Appendix C for further details.

## 4 Lower bound

To prove our calibration lower bound, we make use of the following swap regret lower bound.

Theorem 4.1 (Theorem 4.1 of [DFG^+24]). There is a sufficiently small constant $c_{4.1} > 0$ so that the following holds. Fix any $\epsilon > 0$. For any $d \in \mathbb{N}$, there is a subset $\mathcal{X} \subset [-1, 1]^d$ so that the following holds for any $T \leq \exp(c_{4.1} \min\{d^{1/14}, \epsilon^{-1/6}\})$. There is an oblivious adversary producing a sequence $v_1, \ldots, v_T$ with $\|v_t\|_1 \leq 1$ and $\|v_t\|_\infty \leq \max\{d^{-13/14}, \epsilon^{13/6}\}$ for all $t$, which satisfies the following property. For linear loss functions $\ell(x, v) = \langle v, x \rangle$ for vectors $v \in \mathbb{R}^d$ and $x \in \mathbb{R}^d$, any learning algorithm producing $\mathbf{x}_1, \ldots, \mathbf{x}_T \in \Delta(\mathcal{X})$,

$$\text{FullSwapReg}_T(\mathbf{x}_{1:T}, \ell(\cdot, v_{1:T})) = \sup_{\pi: \mathcal{X} \to \mathcal{X}} \sum_{t=1}^T \sum_{p \in \mathcal{X}} \mathbf{x}_t(p) \cdot (\langle v_t, p \rangle - \langle v_t, \pi(p) \rangle) \geq \epsilon \cdot T.$$

We leverage the classic reduction from swap-regret minimization to calibration [FV98]: by producing calibrated predictions of the upcoming loss and best-responding to it, we can effectively minimize swap regret. This is formalized in the following lemma, proved in Appendix D.

Lemma 4.2. Fix a set $\mathcal{P} \subset \mathbb{R}^d$, a norm $\|\cdot\|$, and write $D(p, p') := \|p - p'\|$. Suppose that, for some $\epsilon > 0, T \in \mathbb{N}$, there is an algorithm which chooses $\mathbf{x}_1, \ldots, \mathbf{x}_T \in \Delta(\mathcal{P})$ and which ensures that for every oblivious adversary choosing $y_1, \ldots, y_T \in \mathcal{P}$, we have $\text{Cal}^D(\mathbf{x}_{1:T}, y_{1:T}) \leq \epsilon \cdot T$. Then for every set $\mathcal{P}' \subset \mathbb{R}^d$, there is an algorithm which chooses $\mathbf{x}'_1, \ldots, \mathbf{x}'_T \in \Delta(\mathcal{P}')$ and which ensures that for every oblivious adversary choosing $y_1, \ldots, y_T \in \mathcal{P}$, we have

$$\text{FullSwapReg}_T(\mathbf{x}'_{1:T}, \ell(\cdot, y_{1:T})) \leq \epsilon \cdot T \cdot \text{diam}_{\|\cdot\|}(\mathcal{P}').$$

Combining these two ideas, we demonstrate that an algorithm producing $\epsilon$-calibrated predictions of outcomes on the simplex in $T \leq \exp(\text{poly}(1/\epsilon))$ rounds could be used in Lemma 4.2 to achieve a swap regret algorithm contradicting Theorem 4.1. This gives the following (proved in Appendix D).

^9 An observant reader might note that this same argument also lets us provide bounds on the regret of FTL for these losses. One subtlety in the analysis is that we obtain better calibration bounds by bounding the distance between the predictions of FTL and BTL in the $\|\cdot\|$ norm rather than in the losses $D_R(y_t\mid\cdot)$, and so it is important that we directly analyze TreeSwap.BTL instead of TreeSwap.FTL (the latter causes us to pick up an extra factor related to the smoothness of $R$).

Theorem 4.3. There is a sufficiently small constant c > 0 so that the following holds. Write D(p, p′) = ||p − p′||1, and fix any ε > 0, d ∈ N. Then for any T ≤ exp(c · min{d^{1/7}, ε^{−1/6}}), there is an oblivious adversary producing a sequence y1, …, yT ∈ Δ^d so that for any learning algorithm producing x1, …, xT ∈ Δ(Δ^d), Cal_T^D(x_{1:T}, y_{1:T}) ≥ ε · T.

In Theorem D.2 (see Appendix D.2), we show a similar lower bound for ℓ2 calibration over the unit ℓ2 ball.


## References


References

[ACRS25] Eshwar Ram Arunachaleswaran, Natalie Collina, Aaron Roth, and Mirah Shi. An elementary predictor obtaining distance to calibration. In Proceedings of the 2025 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 1366–1370. SIAM, 2025.

[AM11] Jacob Abernethy and Shie Mannor. Does an efficient calibrated forecasting strategy exist? In Proceedings of the 24th Annual Conference on Learning Theory, pages 809–812. JMLR Workshop and Conference Proceedings, 2011.

[BGHN23] Jarosław Błasiok, Parikshit Gopalan, Lunjia Hu, and Preetum Nakkiran. A unifying theory of distance from calibration. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing, pages 1727–1740, 2023.

[BM07] Avrim Blum and Yishay Mansour. From external to internal regret. Journal of Machine Learning Research, 8(6), 2007.

[Daw82] A. Philip Dawid. The well-calibrated Bayesian. Journal of the American Statistical Association, 77(379):605–610, 1982.

[DDF+24] Yuval Dagan, Constantinos Daskalakis, Maxwell Fishelson, Noah Golowich, Robert Kleinberg, and Princewill Okoroafor. Breaking the t^{2/3} barrier for sequential calibration. arXiv preprint arXiv:2406.13668, 2024.

[DDFG24] Yuval Dagan, Constantinos Daskalakis, Maxwell Fishelson, and Noah Golowich. From external to swap regret 2.0: An efficient reduction for large action spaces. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing, pages 1216–1222, 2024.

[DFG+24] Constantinos Daskalakis, Gabriele Farina, Noah Golowich, Tuomas Sandholm, and Brian Hu Zhang. A lower bound on swap regret in extensive-form games. arXiv preprint arXiv:2406.13116, 2024.

[FH18] Dean P. Foster and Sergiu Hart. Smooth calibration, leaky forecasts, finite recall, and Nash dynamics. Games and Economic Behavior, 109:271–293, 2018.

[FKO+25] Maxwell Fishelson, Robert Kleinberg, Princewill Okoroafor, Renato Paes Leme, Jon Schneider, and Yifeng Teng. Full swap regret and discretized calibration. arXiv preprint arXiv:2502.09332, 2025.

[FL99] Drew Fudenberg and David K. Levine. An easier way to calibrate. Games and Economic Behavior, 29(1–2):131–137, 1999.

[Fos99] Dean P. Foster. A proof of calibration via Blackwell’s approachability theorem. Games and Economic Behavior, 29(1–2):73–78, 1999.

[FV97] Dean P. Foster and Rakesh V. Vohra. Calibrated learning and correlated equilibrium. Games and Economic Behavior, 21(1–2):40–55, 1997.

[FV98] Dean P. Foster and Rakesh V. Vohra. Asymptotic calibration. Biometrika, 85(2):379–390, 1998.

[GJRR24] Sumegha Garg, Christopher Jung, Omer Reingold, and Aaron Roth. Oracle efficient online multicalibration and omniprediction. In Proceedings of the 2024 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 2725–2792. SIAM, 2024.

[GPSW17] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321–1330. PMLR, 06–11 Aug 2017.

[GSJ24] Khashayar Gatmiry, Jon Schneider, and Stefanie Jegelka. Computing optimal regularizers for online linear optimization. arXiv preprint arXiv:2410.17336, 2024.

[H+16] Elad Hazan et al. Introduction to online convex optimization. Foundations and Trends® in Optimization, 2(3-4):157–325, 2016.

[Har22] Sergiu Hart. Calibrated forecasts: The minimax proof. arXiv preprint arXiv:2209.05863, 2022.

[HJKRR18] Úrsula Hébert-Johnson, Michael P. Kim, Omer Reingold, and Guy N. Rothblum. Multicalibration: Calibration for the (computationally identifiable) masses. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 1939–1948. PMLR, 10–15 Jul 2018.

[HK12] Elad Hazan and Sham M. Kakade. (Weak) calibration is computationally hard. In Conference on Learning Theory, pages 3.1–3.24. JMLR Workshop and Conference Proceedings, 2012.

[HW24] Lunjia Hu and Yifan Wu. Predict to minimize swap regret for all payoff-bounded tasks. In 2024 IEEE 65th Annual Symposium on Foundations of Computer Science (FOCS), pages 244–263. IEEE, 2024.

[KF08] Sham M. Kakade and Dean P. Foster. Deterministic calibration and Nash equilibrium. Journal of Computer and System Sciences, 74(1):115–130, 2008.

[KLST23] Bobby Kleinberg, Renato Paes Leme, Jon Schneider, and Yifeng Teng. U-calibration: Forecasting for an unknown agent. In The Thirty-Sixth Annual Conference on Learning Theory, pages 5143–5145. PMLR, 2023.

[LSS24] Haipeng Luo, Spandan Senapati, and Vatsal Sharan. Optimal multiclass U-calibration error and beyond. arXiv preprint arXiv:2405.19374, 2024.

[LSS25] Haipeng Luo, Spandan Senapati, and Vatsal Sharan. Simultaneous swap regret minimization via KL-calibration. arXiv preprint arXiv:2502.16387, 2025.

[MS10] Shie Mannor and Gilles Stoltz. A geometric proof of calibration. Mathematics of Operations Research, 35(4):721–727, 2010.

[MSA07] Shie Mannor, Jeff S. Shamma, and Gürdal Arslan. Online calibrated forecasts: Memory efficiency versus universality for learning in games. Machine Learning, 67:77–115, 2007.

[NRRX23] Georgy Noarov, Ramya Ramalingam, Aaron Roth, and Stephan Xie. High-dimensional prediction for sequential decision making. arXiv preprint arXiv:2310.17651, 2023.

[Pen25] Binghui Peng. High-dimensional online calibration in polynomial time. arXiv preprint arXiv:2504.09096, 2025.

[PR24] Binghui Peng and Aviad Rubinstein. Fast swap regret minimization and applications to approximate correlated equilibria. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing, pages 1223–1234, 2024.

[QV21] Mingda Qiao and Gregory Valiant. Stronger calibration lower bounds via sidestepping. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing, pages 456–466, 2021.

[QZ24] Mingda Qiao and Letian Zheng. On the distance from calibration in sequential prediction. In The Thirty-Seventh Annual Conference on Learning Theory, pages 4307–4357. PMLR, 2024.

[RS24] Aaron Roth and Mirah Shi. Forecasting for swap regret for all downstream agents. In Proceedings of the 25th ACM Conference on Economics and Computation, pages 466–488, 2024.

[RST15] Alexander Rakhlin, Karthik Sridharan, and Ambuj Tewari. Sequential complexities and uniform martingale laws of large numbers. Probability Theory and Related Fields, 161(1-2):111–153, 2015.

[SS11] Shai Shalev-Shwartz. Online learning and online convex optimization. Foundations and Trends in Machine Learning, 4(2):107–194, 2011.

[SST11] Nati Srebro, Karthik Sridharan, and Ambuj Tewari. On the universality of online mirror descent. *Advances in Neural Information Processing Systems*, 24, 2011.

[ZME20] Shengjia Zhao, Tengyu Ma, and Stefano Ermon. Individual calibration with randomized forecasting. In Hal Daumé III and Aarti Singh, editors, *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pages 11387–11397. PMLR, 13–18 Jul 2020.


## Appendix


A Additional Related Work

There is a large range of other existing work on online (sequential) calibration [Daw82, FV97, FV98, QV21, DDF^+24, Har22, Fos99, FL99, KF08, MSA07, MS10, AM11, HK12, FH18, LSS24, NRRX23, KLST23, GJRR24, QZ24, ACRS25]. We briefly survey some of these areas below.

Binary outcomes. For binary outcomes (i.e., one-dimensional calibration), classical results of [FV97, Fos99, BM07, AM11] demonstrate that it is possible to efficiently guarantee $$O(T^{2/3})$$ $$\ell_1$$-calibration. The optimal possible rates for $$\ell_1$$-calibration remain a major unsolved problem in online learning. Recently [QV21] improved over the naive lower bound of $$\Omega(\sqrt{T})$$ by demonstrating a lower bound of $$\Omega(T^{0.528})$$; this was further improved to $$\Omega(T^{0.543})$$ by [DDF^+24], who also improved on the upper bound, demonstrating the existence of an algorithm with $$O(T^{2/3-\epsilon})$$ calibration for some constant $$\epsilon > 0$$.

Calibration and swap regret. The connection between calibration and swap regret has been acknowledged since the earliest works on swap regret. For example, the earliest algorithms for minimizing swap regret worked by best responding to online calibrated predictions [FV97] (later algorithms for swap regret minimization, such as [BM07] and [DDF^+24], obtain better swap regret bounds by side-stepping the need to generate calibrated predictions). In the other direction, several works minimize calibration by relating it to a swap regret that can then be minimized [FKO^+25, LSS25, AM11, Fos99].

Other forms of calibration. Due to the difficulty of minimizing (high-dimensional) calibration, there has been a line of work on designing forecasting algorithms that minimize weaker forms of calibration that recover some of the important guarantees of calibration (e.g., trustworthiness by a decision-maker). These include distance from calibration [BGHN23, QZ24, ACRS25], omni-prediction error / U-calibration [KLST23, LSS24, GJRR24], calibration conditioned on downstream outcomes [NRRX23], and prediction for downstream swap regret [RS24, HW24]. Other work focuses on minimizing notions of calibration designed to lead to specific classes of equilibria, e.g., weak calibration [HK12], deterministic calibration [KF08], and smooth calibration [FH18].

B Proofs of preliminary results

Figure 2: Geometric depiction of the Bregman divergence from $$p$$ to $$y$$.

Proof of Lemma 2.1.

$$\mathbb{E}_{y\sim\mathbf{y}}[D_R(y||p)] = \mathbb{E}_{y\sim\mathbf{y}}[R(y) - R(p) - \langle\nabla R(p), y - p\rangle]$$
$$= \overline{R}(\bar{y}) - R(p) - \langle\nabla R(p), \bar{y} - p\rangle$$
$$= D_R(\bar{y}||p) + \overline{R}(\bar{y}) - R(\bar{y})$$

See Figure 3 for a visual proof. $$\square$$


![Figure 35](image_35.jpg)
<!-- Caption: Appendix, index 35 -->


Figure 3: [Proof of Lemma 2.1] The average Bregman divergence (orange + purple) decomposes into the Jensen error (orange) and the Bregman divergence to the mean (purple). For example, when $$R(p) = \|p\|_2^2,$$ $$D_R(y|p) = \|y - p\|_2^2,$$ and we recover the bias–variance decomposition.

Proof of Lemma 2.2. Fix any $$p \in \mathcal{P},$$ and consider the quantity $$\max_{p^* \in \mathcal{P}} \sum_t x_t(p)\big(D_R(y_t|p) - D_R(y_t|p^*)\big).$$ By considering the distribution $$y$$ that has weight $$x_t(p)/\sum_i x_i(p)$$ on $$y_t,$$ Lemma 2.1 implies that this quantity is maximized when $$p^* = \nu_p = \big(\sum_t x_t(p)y_t\big)\big/\big(\sum_t x_t(p)\big).$$ At this optimal value of $$p^*,$$ this quantity can be rewritten as:
$$\sum_t x_t(p)\big(D_R(y_t|p) - D_R(y_t|\nu_p)\big)$$
$$= \sum_t x_t(p)\big[(R(y_t) - R(p) - \langle\nabla R(p), y_t - p\rangle) - (R(y_t) - R(\nu_p) - \langle\nabla R(\nu_p), y_t - \nu_p\rangle)\big]$$
$$= \sum_t x_t(p)\big[(R(\nu_p) - R(p) - \langle\nabla R(p), \nu_p - p\rangle) + \langle\nabla R(\nu_p) - \nabla R(p), y_t - \nu_p\rangle\big]$$
$$= \sum_t x_t(p) D_R(\nu_p|p) + \left\langle\nabla R(\nu_p) - \nabla R(p), \sum_t x_t(p)(y_t - \nu_p)\right\rangle$$
$$= \sum_t x_t(p) D_R(\nu_p|p).$$
(Here the last term vanishes since $$\sum_t x_t(p)y_t = \sum_t x_t(p)\nu_p$$). We therefore have that:


![Figure 38](image_38.jpg)
<!-- Caption: Appendix, index 38 -->


FullSwapReg_T(\mathbf{x}_{1:T}, \ell_{1:T}) = \sup_{\pi:\mathcal{P}\to\mathcal{P}} \sum_{t=1}^T \sum_{p\in\mathcal{P}} \mathbf{x}_t(p)\,(\ell_t(p) - \ell_t(\pi(p)))
= \sum_{p\in\mathcal{P}} \max_{p^*\in\mathcal{P}} \sum_{t=1}^T \mathbf{x}_t(p)\,(\ell_t(p) - \ell_t(p^*))
= \sum_{p\in\mathcal{P}} \max_{p^*\in\mathcal{P}} \sum_{t=1}^T \mathbf{x}_t(p)\,(D_R(y_t\,|\,p) - D_R(y_t\,|\,p^*))
= \sum_{p\in\mathcal{P}} \sum_{t=1}^T \mathbf{x}_t(p)\, D_R(\nu_p\,|\,p)
= \mathrm{Cal}_T^{D_R}(\mathbf{x}_{1:T}, y_{1:T}).
□

Proof of Lemma 2.4. Note that if we define \mathcal{L} = \{y \in \mathbb{R}^d \mid \|y\|_* \le 1\} to be the unit dual-norm ball for some norm \|\cdot\|, then by duality the norm \|\cdot\|_{\mathcal{L}^*} corresponding to \mathcal{L}^* is simply the original norm \|\cdot\|. It therefore suffices to show that given a 1-strongly convex function R with bounded range \rho, it is possible to construct a 1-strongly convex function R' with bounded Bregman divergence O(\rho) (and vice versa).

Assume R is 1-strongly convex and satisfies \max_{p\in\mathcal{P}} R(p) - \min_{p\in\mathcal{P}} R(p) = \rho. Define R'(p) = 4\,R\!\left(\frac{p}{2}\right) (since \mathcal{P} is centrally symmetric, p/2 \in \mathcal{P}). If R is 1-strongly convex, then R(p/2) is 1/4-strongly convex, and so R' is also 1-strongly convex. We claim the maximum Bregman divergence of R' is at most O(\rho). To show this, first note that for any z_1, z_2 \in \mathcal{P}, \langle \nabla R(\tfrac{z_1}{2}), z_2 \rangle \le 2\rho. Indeed, since R is convex and has range bounded by \rho, we have \rho \ge R(x) - R(\tfrac{z_1}{2}) \ge \langle \nabla R(\tfrac{z_1}{2}), x - \tfrac{z_1}{2} \rangle for all x \in \mathcal{P}. Setting x = \tfrac{z_1 + z_2}{2} then gives \langle \nabla R(\tfrac{z_1}{2}), z_2 \rangle \le 2\rho. Now, note that
\max_{y,p\in\mathcal{P}} D_{R'}(y\,|\,p) = R'(y) - R'(p) - \langle \nabla R'(p), y - p \rangle
= 4\,R\!\left(\tfrac{y}{2}\right) - 4\,R\!\left(\tfrac{p}{2}\right) - 2\left\langle \nabla R\!\left(\tfrac{p}{2}\right), y - p \right\rangle
\le 4\left| R\!\left(\tfrac{y}{2}\right) - R\!\left(\tfrac{p}{2}\right) \right| + 2\left( \langle \nabla R\!\left(\tfrac{p}{2}\right), y \rangle - \langle \nabla R\!\left(\tfrac{p}{2}\right), p \rangle \right)
\le 4\rho + 4\rho = 8\rho.

Conversely, if R is 1-strongly convex and satisfies \max_{y,p\in\mathcal{P}} D_R(y\,|\,p) \le \rho, define R'(p) = R(p) - \langle \nabla R(0), p \rangle - R(0) (i.e., subtract a linear function to make 0 a minimizer of R'). Since R and R' differ by a linear function, R' is also 1-strongly convex. Moreover, D_R(y\,|\,0) = R(y) - R(0) - \langle \nabla R(0), y \rangle = R'(y); since D_R is bounded in range by \rho, it follows that so is R'.
□

C Proof of Theorem 3.1

In this section, we prove Theorem 3.1. First, in Appendix C.1, we introduce a slightly stronger notion of calibration error and swap regret to deal with a technicality in the proof. We then give the proof of Theorem 3.1.

C.1 Labeled calibration and swap regret

Intuition. Recall that the TreeCal algorithm labels each interval \Gamma_k^{(l)} of the tree with some action p_k^{(l)} \in \mathcal{P}. At each time step t, the algorithm outputs the uniform distribution over all p_k^{(l)}.

Algorithm 1 TreeCal(𝒫, T, H, L)
Require: Action set 𝒫 ⊂ ℝ^d, time horizon T, parameters H, L with T ≤ H^L.
1: for 1 ≤ t ≤ T do
2:     Write the base-H representation of t − 1 as (h₁ … h_L), for h₁, …, h_L ∈ [0 : H − 1].
3:     for 1 ≤ l ≤ L do
4:         Write k := (h₁ … h_{l−1}) ∈ [0 : H − 1]^{l−1}.
5:         if h_{l+1} = ⋯ = h_L = 0 or l = L then
6:             If h_l > 0, define ν_{k,h_l−1}^{(l)} := (1/H^{L−l}) · ∑_{s ∈ Γ_{k,h_l−1}^{(l)}} y_s.
7:             Define p_{k,h_l}^{(l)} := (1/h_l) ∑_{i=0}^{h_l−1} ν_{k,i}^{(l)} if h_l > 0, otherwise choose arbitrary p_{k,h_l}^{(l)} ∈ 𝒫.
8:         end if
9:     end for
10:    Output the uniform mixture x_t := Unif({p_{h₁}^{(1)}, …, p_{h₁⋯h_L}^{(L)}}), and observe y_t.
11: end for

with Γ_k^{(l)} ∋ t. When evaluating the calibration error, suppose that the actions p_k^{(l)} are all distinct, for l ∈ [L], k ∈ [0 : H − 1]^l (as we discuss below, this case is in some sense the “worst case”). In this event, each action p_k^{(l)} is compared to the average outcome over the interval Γ_k^{(l)}: ȳ_k^{(l)} = (1/|Γ_k^{(l)}|) ∑_{t ∈ Γ_k^{(l)}} y_t. Formally, this would give
Cal_T^D(x_{1:T}, y_{1:T}) = ∑_{l=1}^L (H^{L−l}/L) ∑_{k ∈ [H]^l} D(ȳ_k^{(l)}, p_k^{(l)}). (6)
as each level l action is selected with 1/L mass for H^{L−l} rounds.

If it happened that two distinct intervals Γ_{k₁}^{(l₁)}, Γ_{k₂}^{(l₂)} were assigned the same action p = p_{k₁}^{(l₁)} = p_{k₂}^{(l₂)}, then the calibration error would be at most the quantity on the right-hand side of (6) (by Jensen’s inequality). In particular, rather than having to compare p to two potentially distinct quantities D(ȳ_{k₁}^{(l₁)}, p), D(ȳ_{k₂}^{(l₂)}, p), the mass placed on p would be categorized under the same forecast and we would only compare p to an appropriately weighted average of ȳ_{k₁}^{(l₁)} and ȳ_{k₂}^{(l₂)}.

For technical reasons, it will turn out to be necessary to upper bound the “worst case quantity” on the right-hand side of (6) (and an analogous version for swap regret), even in the event that the actions p_k^{(l)} are not all distinct. To streamline our notation, we introduce a generalization of these quantities which apply for arbitrary algorithms, which we call labeled calibration error and labeled swap regret.

Formal definitions.
Given a convex set 𝒫 ⊂ ℝ^d, we define its labeled extension to be 𝒫̄ := 𝒫 × {0,1}^*, i.e., elements of 𝒫̄ are tuples (p, σ), where σ ∈ {0,1}^* is a string that is said to label p. For a loss function ℓ : 𝒫 → ℝ, we extend its domain to 𝒫̄ in the natural way, i.e., ℓ((p, σ)) := ℓ(p) for (p, σ) ∈ 𝒫̄. Given a sequence of distributions over the labeled extension, x₁, …, x_T ∈ Δ(𝒫̄), and loss functions ℓ₁, …, ℓ_T : 𝒫 → ℝ, we define
FullSwapReg_T(x_{1:T}, ℓ_{1:T}) := sup_{π : 𝒫̄ → 𝒫̄} ∑_{t=1}^T ∑_{p ∈ 𝒫̄} x_t(p) · (ℓ_t(p) − ℓ_t(π(p))).
In words, the full swap regret of x_{1:T} with respect to ℓ_{1:T} is defined identically as in (2), except that the swap function π can now depend on the label σ. In particular, the labeled extension allows us to consider a more refined notion of swap regret where identical actions played in different rounds can be swapped (via π) to different alternatives as long as they have different labels.

In a similar manner we define the calibration error for a sequence of labeled distributions: given x₁, …, x_T ∈ Δ(𝒫̄) and y₁, …, y_T ∈ 𝒫, we define
Cal_T^D := ∑_{(p,σ) ∈ 𝒫̄} (∑_{t=1}^T x_t((p,σ))) · D(ν_{(p,σ)}, p), ν_{(p,σ)} := (∑_{t=1}^T x_t((p,σ)) · y_t)/(∑_{t=1}^T x_t((p,σ))).

The main result of [DDFG24] shows that the swap regret of TreeSwap is bounded, even when one labels the action produced at each node of the tree by the node of the tree. This labeled variant of TreeSwap is given in Algorithm 2. It functions exactly as discussed in Section 3.1, except that the distribution $\mathbf{x}_t$ output at time step $t$ is in $\Delta(\bar{\mathcal{P}})$ instead of $\Delta(\mathcal{P})$. In particular, each $p_k^{(l)} \in \mathcal{P}$ in the support of $\mathbf{x}_t$ is labeled by the tuple $k \in [0 : H - 1]^l$.^10

^10 Technically, the analysis of [DDFG24] does not analyze the labeled version, but the proof goes through as is—the only step where labeling changes any of the reasoning in the argument is in Eq. (8) of [DDFG24], where the upper bound as written in that equation holds even for the labeled version.

Theorem C.1 (TreeSwap; Theorem 3.1 of [DDFG24]). Suppose that $H, L \in \mathbb{N}$ satisfy $H \geq 2$ and $H^{L-1} \leq T \leq H^L$. For bounded convex action set $\mathcal{P} \subset \mathbb{R}^d$ and loss function set $\mathcal{L} \subset \{\ell : \mathcal{P} \to [0, b]\}$, let $\text{Alg}_H : \mathcal{L}^H \to \mathcal{P}^H$ be any algorithm. Then, the labeled TreeSwap algorithm (Algorithm 2) parametrized by $T, H, L, \mathcal{P}, \mathcal{L}, \text{Alg}_H$ outputs labeled distributions $\mathbf{x}_1, \ldots, \mathbf{x}_T \in \Delta(\bar{\mathcal{P}})$ satisfying the following: for any sequence $\ell_1, \ldots, \ell_T \in \mathcal{L}$,
$$
\text{FullSwapReg}_T(\mathbf{x}_{1:T}, \ell_{1:T}) \leq T \cdot \left(\frac{\text{ExtReg}_H(\text{Alg}_H)}{H} + \frac{b}{L}\right).
$$

Algorithm 2 TreeSwap.Alg($\mathcal{P}, \mathcal{L}, T, H, L$), labeled variant (see Appendix C.1)

Require: Action set $\mathcal{P} \subset \mathbb{R}^d$, convex loss class $\mathcal{L} \subset (\mathcal{P} \to \mathbb{R})$, no-external regret algorithm Alg, time horizon $T$, parameters $H, L$ with $T \leq H^L$.

1: For each sequence $h_1 \cdots h_{l-1} \in \bigcup_{l=1}^L [0 : H - 1]^{\,l-1}$, initialize an instance of Alg with time horizon $H$, denoted $\text{Alg}_{h_{1:l-1}}$.
2: for $1 \leq t \leq T$ do
3:     Write the base-$H$ representation of $t-1$ as $t-1 = (h_1 \cdots h_L)$, for $h_1, \ldots, h_L \in [0 : H - 1]$.
4:     for $1 \leq l \leq L$ do
5:         Write $k := (h_1 \cdots h_{l-1}) \in [0 : H - 1]^{\,l-1}$.
6:         if $h_{l+1} = \cdots = h_L = 0$ or $l = L$ then
7:             If $h_l > 0$, define $\ell_{k,h_l-1}^{(l)} := \frac{1}{H^{L-l}} \cdot \sum_{s \in \Gamma_{k,h_l-1}^{(l)}} \ell_s \in \mathcal{L}$.
8:             Define $p_{k,h_l}^{(l)} = \text{Alg}_{k,h_l}\!\left(\ell_{k,1:h_l-1}^{(l)}\right) \in \mathcal{P}$. ⊳ The $h_l$-th action of $\text{Alg}_k$ given the loss sequence $\ell_{k,1:h_l-1}^{(l)}$.
9:         end if
10:     end for
11:     Output the uniform mixture $\mathbf{x}_t := \text{Unif}(\{(p_{h_1}^{(1)}, h_1), \ldots, (p_{h_{1:L}}^{(L)}, h_{1:L})\}) \in \Delta(\bar{\mathcal{P}})$, and observe $\ell_t$. ⊳ Each action $p_k^{(l)}$ is labeled by the sequence $k$ (see Appendix C.1).
12: end for

C.2 Proof of the main theorem

First, we recall some definitions from Section 3. For all $l \in [0 : L]$, for all $k \in [H]^l$, let $\Gamma_k^{(l)}$ represent the interval of times $t$ with prefix $k$. That is, $t \in \Gamma_k^{(l)}$ iff $t_i = k_i$ for all $i \in [1 : l]$. These intervals form an $H$-ary depth-$L$ tree, where the children of $\Gamma_k^{(l)}$ are $\Gamma_{k0}^{(l+1)}, \Gamma_{k1}^{(l+1)}, \cdots, \Gamma_{k(H-1)}^{(l+1)}$. In the calibration setting where the learner receives outcomes $y_{1:T}$, let $\nu_k^{(l)} = \frac{1}{|\Gamma_k^{(l)}|} \sum_{t \in \Gamma_k^{(l)}} y_t$ (as defined on Line 6 of Algorithm 1). In the swap regret setting where the learner receives loss functions $\ell_{1:T}$, let $\ell_k^{(l)} = \frac{1}{|\Gamma_k^{(l)}|} \sum_{t \in \Gamma_k^{(l)}} \ell_t$ (as defined in Line 7 of Algorithm 2).

Finally, recall that for an online learning algorithm Alg with time horizon $H$, we define its action at time step $h \in [H]$ given losses $\ell_1, \ldots, \ell_H : \mathcal{P} \to \mathbb{R}$ by $\text{Alg}_h(\ell_1, \ldots, \ell_H)$. If $\text{Alg}_h$ only depends on the first $g$ losses, then we will write $\text{Alg}_h(\ell_1, \ldots, \ell_g)$. In the proof of Theorem 3.1 we will consider two algorithms in particular; the first, Follow-The-Leader (FTL) is defined as follows: for

For $\ell_1, \ldots, \ell_{h-1} : \mathcal{P} \to \mathbb{R}$, we have

$$\text{FTL}_h(\ell_1, \ldots, \ell_{h-1}) = \argmin_{p \in \mathcal{P}} \sum_{i=1}^{h-1} \ell_i(p).$$

The second algorithm we consider is the Be-The-Leader algorithm (BTL), which is defined as follows: for $\ell_1, \ldots, \ell_h : \mathcal{P} \to \mathbb{R}$, we have

$$\text{BTL}_h(\ell_1, \ldots, \ell_h) = \argmin_{p \in \mathcal{P}} \sum_{i=1}^{h} \ell_i(p).$$

Note that since $\text{BTL}_h(\ell_{1:h})$ depends on the unobserved loss $\ell_h$ at time step $h$, it is unimplementable. Nevertheless, it will be useful in our analysis.

Next we prove Lemma 3.2, establishing the equivalence of TreeCal and TreeSwap.FTL. In fact, we establish the stronger claim, which immediately implies Lemma 3.2.

Lemma C.2. Fix distributions $q_0, \ldots, q_h \in \Delta(\mathcal{P})$, and define $\ell_h(p) := \mathbb{E}_{y \sim q_h}[D_R(y\|p)]$. Then for each $h > 0$, $\text{FTL}_h(\ell_0, \ldots, \ell_{h-1}) = \frac{1}{h} \sum_{i=0}^{h-1} \mathbb{E}_{y \sim q_i}[y]$.

Proof. The lemma is an immediate consequence of Lemma 2.1, noting that

$$\text{FTL}_h(\ell_0, \ldots, \ell_{h-1}) = \argmin_{p \in \mathcal{P}} \sum_{i=0}^{h-1} \ell_i(p) = \argmin_{p \in \mathcal{P}} \mathbb{E}_{i \sim [0:h-1],\, y \sim q_i}[D_R(y\|p)] = \frac{1}{h} \sum_{i=0}^{h-1} \mathbb{E}_{y \sim q_i}[y].$$
(7)

□

Proof of Lemma 3.2. At time $t$, both TreeCal (Line 10 of Algorithm 1) and TreeSwap.FTL (Line 11 of Algorithm 2) select $\mathbf{x}_t = \operatorname{Unif}\left(\left\{p_{t_1}^{(1)}, p_{t_1 t_2}^{(2)}, \cdots, p_{t_1 t_2 \cdots t_L}^{(L)}\right\}\right)$. It remains to demonstrate that both algorithms assign actions $p_k^{(l)}$ to intervals $\Gamma_k^{(l)}$ identically. Fixing a choice of $l \in [L]$ and $k \in [0:H-1]^{l-1}$, this is an immediate consequence of Lemma C.2 with $q_h = \operatorname{Unif}(\{y_t : t \in \Gamma_{k,h}^{(l)}\})$ and the fact that:

• In TreeCal, $p_{k,h}^{(l)} = \frac{1}{h} \sum_{i=0}^{h-1} \nu_{k,i}^{(l)}$ with $\nu_{k,i}^{(l)} = \mathbb{E}_{t \sim \operatorname{Unif}(\Gamma_{k,i}^{(l)})}[y_t]$;

• Whereas in TreeSwap.FTL, $p_{k,h}^{(l)} = \text{FTL}_{h+1}(\ell_{k,0}^{(l)}, \ldots, \ell_{k,h-1}^{(l)})$ with $\ell_{k,i}^{(l)} = \mathbb{E}_{t \sim \operatorname{Unif}(\Gamma_{k,i}^{(l)})}[D_R(y_t\|\cdot)]$.

□

We are now ready to prove Theorem 3.1.

Proof of Theorem 3.1. Fix any convex set $\mathcal{P}$ and a norm $\|\cdot\|$, and let $R : \mathcal{P} \to \mathbb{R}$ be chosen to be 1-strongly convex which has range $\rho > 0$. Lemma 3.2 gives that the actions $\mathbf{x}_1, \ldots, \mathbf{x}_T \in \Delta(\mathcal{P})$ are identical to the actions played by TreeSwap.FTL with losses $\ell_t(p) = D_R(y_t\|p)$ (Algorithm 2; we are ignoring the labels here). Thus, from here on, it suffices to bound the calibration error of the corresponding distributions $\mathbf{x}_1, \ldots, \mathbf{x}_T$ of TreeSwap.FTL. The actions $p_{k,h}^{(l)}$ (for $l \in [L], k \in [0:H-1]^{\,l-1}, h \in [0:H-1]$) of TreeSwap.FTL satisfy $p_{k,h}^{(l)} = \text{FTL}_{h+1}(\ell_{k,0}^{(l)}, \ldots, \ell_{k,h-1}^{(l)})$.

Next, let $\tilde{p}_{k,h}^{(l)}$ denote the corresponding actions played by TreeSwap.BTL, i.e., $\tilde{p}_{k,h}^{(l)} = \text{BTL}_{h+1}(\ell_{k,0}^{(l)}, \ldots, \ell_{k,h}^{(l)})$. We let $\mathbf{x}_t \in \Delta(\bar{\mathcal{P}})$ denote the (labeled) distribution chosen by TreeSwap.FTL (Line 11 of Algorithm 2), and let $\tilde{\mathbf{x}}_t \in \Delta(\bar{\mathcal{P}})$ denote the corresponding distribution for TreeSwap.BTL. To be concrete, if $t - 1 = (h_1 \cdots h_L)$, then

$$\mathbf{x}_t = \operatorname{Unif}(\{(p_{h_1}^{(1)}, h_1), \ldots, (p_{h_1 \cdots h_L}^{(L)}, h_{1:L})\}), \quad \tilde{\mathbf{x}}_t = \operatorname{Unif}(\{(\tilde{p}_{h_1}^{(1)}, h_1), \ldots, (\tilde{p}_{h_1 \cdots h_L}^{(L)}, h_{1:L})\}),$$
(8)

We state the below claim, whose proof is deferred to the end of the section. (We remark that the primary purpose of introducing labeling is so that it is possible to establish Claim C.3.)

Claim C.3. It holds that
Cal_T^{||·||^2}(x_{1:T}, y_{1:T}) − 2 Cal_T^{||·||^2}(𝑥̃_{1:T}, y_{1:T}) ≥ (2 · diam(𝒫)^2 / H^2) · T. (9)

The fact that BTL enjoys non-positive external regret (e.g., [SS11, Lemma 2.1]) gives that for an arbitrary sequence of loss functions ℓ_t : 𝒫 → ℝ, the external regret of BTL_H satisfies ExtReg_H(BTL_H) ≤ 0. Thus, by Theorem C.1, the swap regret of (the labeled version of) TreeSwap_T applied with Alg_H = BTL_H may be bounded as follows: for any sequence of losses ℓ_1, …, ℓ_T : 𝒫 → [0, ρ],
FullSwapReg_T(𝑥̃_{1:T}, ℓ_{1:T}) ≤ T · (3ρ)/L.

Using Lemma 2.2^11 and (9), we get that for an arbitrary sequence y_1, …, y_T ∈ 𝒫,
Cal_T^{||·||^2}(x_{1:T}, y_{1:T}) ≤ 2 · Cal_T^{||·||^2}(𝑥̃_{1:T}, y_{1:T}) + (2 · diam(𝒫)^2/H^2) · T
≤ 2 · Cal_T^{D_R}(𝑥̃_{1:T}, y_{1:T}) + (2 · diam(𝒫)^2/H^2) · T
= 2 · FullSwapReg_T(𝑥̃_{1:T}, D_R(y_{1:T}[·])) + (2 · diam(𝒫)^2/H^2) · T
≤ (6ρ · T)/L + (2 · diam(𝒫)^2 · T)/H^2.

Given any desired accuracy ε > 0, choosing L = 12ρ/ε and H = diam(𝒫)/√ε gives that we can guarantee Cal_T^{||·||^2}(x_{1:T}, y_{1:T}) ≤ ε · T as long as T ≥ H^L = (diam(𝒫)/√ε)^{12ρ/ε}. □

Proof of Claim C.3. For each t ∈ [T], we can write t − 1 = h_1 h_2 ··· h_L with h_i ∈ [0 : H − 1] for all i ∈ [L], and x_t, 𝑥̃_t are as given in (8). Let us write, for (p, σ) ∈ 𝒫̃,
ν_{(p,σ)} := (∑_{t=1}^T x_t((p, σ)) · y_t) / (∑_{t=1}^T x_t((p, σ))),  𝑣̃_{(p,σ)} := (∑_{t=1}^T 𝑥̃_t((p, σ)) · y_t) / (∑_{t=1}^T 𝑥̃_t((p, σ))),
ν_σ := (∑_{p∈𝒫} ∑_{t=1}^T x_t((p, σ)) · y_t) / (∑_{p∈𝒫} ∑_{t=1}^T x_t((p, σ))). (10)

Since each p^{(l)}_{h_1···h_l} and each 𝑝̃^{(l)}_{𝑕̃_1···𝑕̃_l} is labeled by h_{1:l} in x_t and 𝑥̃_t, respectively, it holds that for each σ of the form σ = h_1 ··· h_l (for some l ∈ [L]), there are unique p, 𝑝̃ ∈ 𝒫 so that ν_σ = ν_{(p,σ)} = ν_{(𝑝̃,σ)};

^11 Technically, we need a labeled version of Lemma 2.2, where the distributions x_t are over the labeled set Δ(𝒫); it is immediate to see that the proof of Lemma 2.2 extends to the labeled case.

in particular, we have p = p_{h_1 \cdots h_l}^{(l)}, \tilde{p} = \tilde{p}_{h_1 \cdots h_l}^{(l)}. We can therefore bound

Cal_T^{\|\cdot\|^2}(\mathbf{x}_{1:T}, y_{1:T}) - 2\,Cal_T^{\|\cdot\|^2}(\tilde{\mathbf{x}}_{1:T}, y_{1:T})

= \sum_{l \in [L],\, h_{1:l} \in [0:H-1]^l} \left( \sum_{t=1}^T \mathbf{x}_t\big((p_{h_1 \cdots h_l}^{(l)}, h_1 \cdots h_l)\big) \right) \cdot \|\nu_{h_1 \cdots h_l} - p_{h_1 \cdots h_l}^{(l)}\|^2 \tag{11}

- 2 \left( \sum_{t=1}^T \tilde{\mathbf{x}}_t\big((\tilde{p}_{h_1 \cdots h_l}^{(l)}, h_1 \cdots h_l)\big) \right) \cdot \|\nu_{h_1 \cdots h_l} - \tilde{p}_{h_1 \cdots h_l}^{(l)}\|^2

= \sum_{l \in [L],\, h_{1:l} \in [0:H-1]^l} \frac{H^{L-l}}{L} \left( \|\nu_{h_1 \cdots h_l} - p_{h_1 \cdots h_l}^{(l)}\|^2 - 2\|\nu_{h_1 \cdots h_l} - \tilde{p}_{h_1 \cdots h_l}^{(l)}\|^2 \right)

\le 2 \sum_{l \in [L],\, h_{1:l} \in [0:H-1]^l} \frac{H^{L-l}}{L} \, \|p_{h_1 \cdots h_l}^{(l)} - \tilde{p}_{h_1 \cdots h_l}^{(l)}\|^2

\le \frac{2}{L} \sum_{l=1}^L \sum_{h_{1:l-1} \in [0:H-1]^{l-1}} \operatorname{diam}(\mathcal{P})^2 \cdot H^{L-l}

\le \frac{2}{L} \sum_{l=1}^L \operatorname{diam}(\mathcal{P})^2 \cdot H^{L-1}

= \frac{2T\,\operatorname{diam}(\mathcal{P})^2}{H},

where the second-to-last inequality uses that \sum_{h_l=0}^{H-1} \|p_{h_1 \cdots h_l}^{(l)} - \tilde{p}_{h_1 \cdots h_l}^{(l)}\|^2 \le \operatorname{diam}(\mathcal{P})^2 for all choices of h_1 \cdots h_{l-1} (a consequence of Lemma C.4 and Lemma C.2).

□

Lemma C.4. Fix any convex set \mathcal{P} \subset \mathbb{R}^d and a convex function R : \mathcal{P} \to \mathbb{R}. Fix a sequence y_1,\ldots,y_H \in \mathcal{P}, and set

p_h = \frac{1}{h-1} \sum_{i=1}^{h-1} y_i \ \forall h \in [H],\, h>1, \qquad \tilde{p}_h = \frac{1}{h} \sum_{i=1}^h y_i \ \forall h \in [H],

as well as p_1 \in \mathcal{P} arbitrarily. Then

\sum_{h=1}^H \|p_h - \tilde{p}_h\|^2 \le 2 \cdot \operatorname{diam}(\mathcal{P})^2.

Proof. Note that

\tilde{p}_h - p_h = \frac{y_h}{h} - \frac{1}{h(h-1)} \sum_{i=1}^{h-1} y_i,

which implies that \|\tilde{p}_h - p_h\|^2 \le \frac{\pi^2}{6} \cdot \operatorname{diam}(\mathcal{P})^2 < 2\,\operatorname{diam}(\mathcal{P})^2.

□

Applying Cauchy–Schwarz, we get the following corollary.

Corollary C.5. Let \mathcal{P} \subset \mathbb{R}^d be a bounded convex set and \|\cdot\| be an arbitrary norm. Then, TreeCal (Algorithm 1) guarantees that for an arbitrary sequence of outcomes y_1,\ldots,y_T \in \mathcal{P}, the \|\cdot\| calibration error of its predictions \mathbf{x}_1,\ldots,\mathbf{x}_T \in \Delta(\mathcal{P}) is bounded Cal_T^{\|\cdot\|}(\mathbf{x}_{1:T}, y_{1:T}) \le \epsilon T for T \ge (\operatorname{diam}_{\|\cdot\|}(\mathcal{P})/\epsilon)^{O(\operatorname{Rate}(\mathcal{P}, \|\cdot\|)/\epsilon^2)}.

Proof. Using the fact that $\sum_{p \in \mathcal{P}} \sum_{t=1}^T \mathbf{x}_t(p) = T$ together with Jensen's inequality, we have

$\displaystyle \frac{1}{T}\, \mathrm{Cal}_T^{\|\cdot\|}(\mathbf{x}_{1:T}, y_{1:T}) = \frac{1}{T} \sum_{p \in \mathcal{P}} \left( \sum_{t=1}^T \mathbf{x}_t(p) \right) \cdot \|\nu_p - p\|$

$\displaystyle \leq \sqrt{\frac{1}{T} \sum_{p \in \mathcal{P}} \left( \sum_{t=1}^T \mathbf{x}_t(p) \right) \cdot \|\nu_p - p\|^2}$

$\displaystyle = \sqrt{\frac{1}{T} \mathrm{Cal}_T^{\|\cdot\|^2}(\mathbf{x}_{1:T}, y_{1:T})} \leq \epsilon$

for $T \geq (\mathrm{diam}(\mathcal{P})/\epsilon)^{O(\mathrm{Rate}(\mathcal{P}, \|\cdot\|)/\epsilon^2)}$ by Theorem 3.1. Thus, $\mathrm{Cal}_T^{\|\cdot\|}(\mathbf{x}_{1:T}, y_{1:T}) \leq \epsilon T$ for $T \geq (\mathrm{diam}(\mathcal{P})/\epsilon)^{O(\mathrm{Rate}(\mathcal{P}, \|\cdot\|)/\epsilon^2)}$, incurring an additional factor of 2 in the exponent constant, as desired. □

Finally, for the setting of centrally symmetric $\mathcal{P}$, we can apply Lemma 2.4 to directly relate this regret bound to the optimal possible rate of an online linear optimization problem.

Corollary C.6. Let $\mathcal{P} \subset \mathbb{R}^d$ be a bounded centrally symmetric convex set and $\|\cdot\|$ be an arbitrary norm. Then, TreeCal (Algorithm 1) guarantees that for an arbitrary sequence of outcomes $y_1, \ldots, y_T \in \mathcal{P}$, the $\|\cdot\|$ calibration error of its predictions $\mathbf{x}_1, \ldots, \mathbf{x}_T \in \Delta(\mathcal{P})$ is bounded $\mathrm{Cal}_T^{\|\cdot\|}(\mathbf{x}_{1:T}, y_{1:T}) \leq \epsilon T$ for $T \geq (\mathrm{diam}_{\|\cdot\|}(\mathcal{P})/\epsilon)^{O(\mathrm{Rate}_{\mathrm{OLO}}(\mathcal{P}, \|\cdot\|)/\epsilon^2)}$.

Proof. Follows immediately by applying Lemma 2.4 to Corollary C.5. □

D Proofs for Section 4

In this section, we prove lower bounds on high-dimensional calibration that tell us that in order to achieve calibration error at most $\epsilon \cdot T$, we need to take $T \gtrsim \exp(\mathrm{poly}(1/\epsilon))$. First, in Appendix D.1, we prove a lower bound for $\ell_1$ calibration over the $d$-dimensional simplex, and then, in Appendix D.2, we prove a lower bound for $\ell_2$ calibration over the unit $d$-dimensional Euclidean ball.

D.1 Lower bound on $\ell_1$ calibration

First, we prove Theorem 4.3 which gives a lower bound on $\ell_1$ calibration over the simplex $\mathcal{P} = \Delta^d$.

Proof of Lemma 4.2. Fix an algorithm Alg which ensures that $\mathrm{Cal}_T^D(\mathbf{x}_{1:T}, y_{1:T}) \leq \epsilon \cdot T$ as in the statement of the lemma. We construct the following algorithm Alg': it simulates Alg, but whenever Alg outputs the distribution $\mathbf{x}_t \in \Delta(\mathcal{P})$, Alg' chooses instead $\mathbf{x}_t' \in \Delta(\mathcal{P}')$, defined by

$\displaystyle \mathbf{x}_t'(p') := \sum_{\substack{p \in \mathcal{P}: \\ p' = \arg\min_{q \in \mathcal{P}'} \langle q, p \rangle}} \mathbf{x}_t(p).$

To simplify notation, we define BR(p) := argmin_{q∈P'} ⟨q, p⟩. It follows that, for any oblivious adversary choosing a (random) sequence y₁, …, y_T ∈ P,

FullSwapReg_T(x'_{1:T}, ℓ(·, y_{1:T}))
= sup_{π: P' → P'} ∑_{p'∈P'} ∑_{t∈[T]} x'_t(p') · ⟨y_t, p' − π(p')⟩
= sup_{π: P' → P'} ∑_{p∈P} ∑_{t∈[T]} x_t(p) · ⟨y_t, BR(p) − π(BR(p))⟩
= sup_{π: P' → P'} ∑_{p∈P} (∑_{t∈[T]} x_t(p)) · ⟨ν_p, BR(p) − π(BR(p))⟩
= sup_{π: P' → P'} ∑_{p∈P} (∑_{t∈[T]} x_t(p)) · (⟨ν_p − p, BR(p) − π(BR(p))⟩ + ⟨p, BR(p) − π(BR(p))⟩)
≤ sup_{π: P' → P'} ∑_{p∈P} (∑_{t∈[T]} x_t(p)) · (‖ν_p − p‖ · ‖BR(p) − π(BR(p))‖_*)
≤ diam_{‖·‖_*}(P') · Cal^D_T(x_{1:T}, y_{1:T}),

where in the final inequality we have used the fact that ‖BR(p) − π(BR(p))‖_* ≤ diam_{‖·‖_*}(P'). □

For p > 0, d ∈ ℕ, write B^d_p := {x ∈ ℝ^d | ‖x‖_p ≤ 1} to denote the unit ℓ_p norm ball.

To map the lower bound Theorem 4.1 from the ‖·‖₁-norm unit ball B^d_1 to the simplex and arrive at the desired contradiction using the above lemma, we use the following.

Lemma D.1. Fix d ∈ ℕ, and write D(x, y) := ‖x − y‖₁ for x, y ∈ ℝ^d. Suppose that there is an algorithm Alg for calibration over the domain P = Δ^{2d+1} which produces x_{1:T} given the choices of an adversary y_{1:T} achieving calibration error Cal^D_T(x_{1:T}, y_{1:T}) ≤ R(T), for T ∈ ℕ. Then there is an algorithm Alg' for calibration over the domain B^d_1 which produces x'_{1:T} given y'_{1:T} achieving calibration error Cal^D_T(x'_{1:T}, y'_{1:T}) ≤ R(T).

Proof of Lemma D.1. We define a mapping φ : B^d_1 → Δ^{2d+1} as follows: for y ∈ B^d_1 ⊂ ℝ^d, we define
φ(y)_i =
{ [y_j]_+ : i = 2j − 1, j ∈ [d]
  [y_j]_- : i = 2j,     j ∈ [d]
  1 − ‖y‖_1 : i = 2d + 1. }

It is straightforward to see that φ has a left inverse ψ, defined as follows: for z ∈ Δ^{2d+1},
ψ(z)_i = z_{2i−1} − z_{2i}, i ∈ [d],
so that ψ ∘ φ(y) = y for all y ∈ B^d_1.

We define the algorithm Alg' as follows: given y'_t ∈ B^d_1 ⊂ ℝ^d, it defines y_t ∈ Δ^{2d+1} by y_t = φ(y'_t). Alg' then feeds y_t into Alg, and if we denote the distribution output by Alg at time step t by x_t, Alg' then plays the push-forward measure x'_t := ψ ∘ x_t ∈ Δ(B^d_1).

Our bound on the calibration error of Alg gives
Cal^D_T(x_{1:T}, y_{1:T}) = ∑_{p∈Δ^{2d+1}} (∑_{t=1}^T x_t(p)) · ‖ν_p − p‖_1 ≤ R(T),
where ν_p = (∑_{t=1}^T x_t(p) · y_t)/(∑_{t=1}^T x_t(p)) ∈ Δ^{2d+1}. For p' ∈ B^d_1, let us denote
ν'_{p'} := (∑_{t=1}^T x'_t(p') · y'_t)/(∑_{t=1}^T x'_t(p')) = ψ((∑_{t=1}^T x'_t(p') · y_t)/(∑_{t=1}^T x'_t(p')); by linearity of ψ.

We may now bound the calibration error of Alg' by
$$\text{Cal}^D_T(\mathbf{x}'_{1:T}, y'_{1:T}) = \sum_{p' \in \mathcal{B}^{d'}_1} \left( \sum_{t=1}^T \mathbf{x}'_t(p') \right) \cdot \|\nu'_{p'} - p'\|_1$$
$$\leq \sum_{p \in \Delta^{2d+1}} \left( \sum_{t=1}^T \mathbf{x}_t(p) \right) \cdot \|\psi(\nu_p) - \psi(p)\|_1$$
$$\leq \text{Cal}^D_T(\mathbf{x}_{1:T}, y_{1:T}).$$
□

Proof of Theorem 4.3. Suppose to the contrary that there was an algorithm Alg which bounded calibration error by $\epsilon T$ for $T \leq \exp(c \cdot \min\{d^{1/14}, \epsilon^{-1/6}\})$. Then by Lemma D.1, for $d' = \lfloor(d-1)/2\rfloor$ there is an algorithm Alg' for calibration on the domain $\mathcal{B}^{d'}_1 \subset \mathbb{R}^{d'}$ which, given $y'_{1:T}$, produces $\mathbf{x}'_{1:T}$ satisfying $\text{Cal}^D_T(\mathbf{x}'_{1:T}, y'_{1:T}) \leq \epsilon \cdot T$ for any $T \leq \exp(c \cdot \min\{d^{1/14}, \epsilon^{-1/6}\})$.

We now apply Lemma 4.2 for $\mathcal{P} = \mathcal{B}^{d'}_1 \subset \mathbb{R}^{d'}$, the norm given by the $\ell_1$ norm $\|\cdot\|_1$, and $\mathcal{P}' := [-1,1]^{d'}$. Note that $\text{diam}_{\|\cdot\|_\infty}(\mathcal{P}') = 1$. Then Lemma 4.2 ensures that there is an algorithm Alg'' which chooses $\mathbf{x}''_1, \ldots, \mathbf{x}''_T \in \Delta(\mathcal{P}')$ which ensures that for every oblivious adversary choosing $y''_1, \ldots, y''_T \in \mathcal{B}^{d'}_1$, we have $\text{FullSwapReg}_T(\mathbf{x}''_{1:T}, \ell(\cdot, y''_{1:T})) \leq \epsilon \cdot T$.

But if $T \leq \exp(c_{4.1} \cdot \min\{(d')^{1/14}, \epsilon^{-1/6}\})$, we have a contradiction to Theorem 4.1, thus completing the proof of the theorem. □

D.2 Lower bound for $\ell_2$ calibration

Next, we prove a lower bound for $\ell_2$ calibration.

Theorem D.2. There is a sufficiently small constant $c > 0$ so that the following holds. Write $D(p, p') = \|p - p'\|_2$ and fix any $\epsilon > 0, d \in \mathbb{N}$. Then for any $T \leq \exp(c \cdot \min\{d^{1/14}, \epsilon^{-1/7}\})$, there is an oblivious adversary producing a sequence $y_1, \ldots, y_T \in \mathcal{B}^d_2$ so that for any learning algorithm producing $\mathbf{x}_1, \ldots, \mathbf{x}_T \in \Delta(\mathcal{B}^d_2)$, $\text{Cal}^D_T(\mathbf{x}_{1:T}, y_{1:T}) \geq \epsilon \cdot T$.

Proof. Fix $\epsilon > 0, d \in \mathbb{N}$, and write $\tilde{\epsilon} = \epsilon^{6/7}$. We may assume without loss of generality that $d \leq \tilde{\epsilon}^{-14/6}$, so that $\min\{d^{1/14}, \tilde{\epsilon}^{-1/6}\} = \min\{d^{1/14}, \epsilon^{-1/7}\} = d^{1/14}$: if this were not the case, we simply use the adversary resulting from $\tilde{\epsilon}^{-14/6}$ dimensions and project the forecaster's predictions down into this lower-dimensional subspace, which can only decrease calibration error. Now suppose to the contrary that there was an algorithm Alg which bounded calibration error by $\epsilon T$ for $T \leq \exp(c \cdot \min\{d^{1/14}, \epsilon^{-1/7}\}) = \exp(c \cdot d^{1/14})$. Then by Lemma 4.2 with $\mathcal{P} = \mathcal{B}^d_2$ and norm $\|\cdot\| = \|\cdot\|_2$, for any subset $\mathcal{P}' \subset \mathcal{B}^d_2$ we get that there is an algorithm which chooses $\mathbf{x}'_1, \ldots, \mathbf{x}'_T \in \Delta(\mathcal{P}')$ and which ensures that for every oblivious adversary choosing $y_1, \ldots, y_T \in \mathcal{B}^d_2$, we have
$$\text{FullSwapReg}_T(\mathbf{x}'_{1:T}, (\langle \cdot, y_1 \rangle, \ldots, \langle \cdot, y_T \rangle)) \leq \epsilon \cdot T. \tag{12}$$

On the other hand, the oblivious adversary of Theorem 4.1 guarantees a subset $\mathcal{X} \subset [-1,1]^d \subset \mathbb{R}^d$ and an oblivious adversary producing a sequence $v_1, \ldots, v_T \in \mathbb{R}^d$ with $\|v_t\|_\infty \leq d^{-13/14}$ for all $t \in [T]$, so that
$$\text{FullSwapReg}_T(\mathbf{x}_{1:T}, (\langle \cdot, v_1 \rangle, \ldots, \langle \cdot, v_T \rangle)) \geq \tilde{\epsilon} \cdot T \tag{13}$$
as long as $T \leq \exp(c_{4.1} \cdot d^{1/14})$. We have $\|v_t\|_2 \leq d^{1/2-13/14} = d^{-3/7}$ for all $t$, and scaling $\mathcal{X}$ down by a factor of $1/\sqrt{d}$ (i.e., letting $\mathcal{P}' = \mathcal{X}/\sqrt{d}$) and all vectors $v_t$ up by a factor of $d^{3/7}$ (i.e., letting $v'_t = d^{3/7} \cdot v_t$) ensures that any algorithm producing $\mathbf{x}'_1, \ldots, \mathbf{x}'_T \in \mathcal{P}'$ must still have full swap regret
$$\text{FullSwapReg}_T(\mathbf{x}'_{1:T}, (\langle \cdot, v'_1 \rangle, \ldots, \langle \cdot, v'_T \rangle)) > \tilde{\epsilon} \cdot T \cdot d^{-1/14} \geq \tilde{\epsilon}^{7/6} \cdot T = \epsilon \cdot T,$$
but now ensures that $\mathcal{P}' \subset \mathcal{B}^d_2$ and that $v'_t \in \mathcal{B}^d_2$ for all $t$. By taking $c = c_{4.1}$, this contradicts (12). □

E Pure calibration and pure full swap regret

E.1 Pure calibration

In certain settings of calibration, the learner is required to randomly select a pure forecast p_t ∈ P rather than a distribution x_t ∈ Δ(P). In these settings, the above definition of calibration is instead referred to as “pseudo-calibration”. Here, we stick to calling the above calibration, as we believe it to be the more natural definition, and instead refer to this alternative setting as “pure-calibration”. The learning task changes as follows.

At each time step t ∈ [T]:
• The learning algorithm chooses a distribution x_t ∈ Δ(P).
• The adversary observes x_t and chooses an outcome y_t ∈ P.
• The learner samples p_t ∼ x_t.

We adjust the definitions of the “pure average outcome” and “pure-calibration” accordingly:
ν̇_p := (∑_{t=1}^T 1[p_t = p] · y_t) / (∑_{t=1}^T 1[p_t = p]),
PureCal_T^D(p_{1:T}, y_{1:T}) := ∑_{p∈P} (∑_{t=1}^T 1[p_t = p]) · D(ν̇_p, p).

Algorithm 3 SampleTreeCal(P, T, H, L, S)

Require: Action set P ⊂ ℝ^d, time horizon T, repetition parameter S; parameters H, L with T/S ≤ H^L.
1: Instantiate an instance TreeCal(P, T/S, H, L).
2: for 1 ≤ i ≤ T/S do
3: Let x_i ∈ Δ(P) denote the prediction of TreeCal at step i.
4: for 1 ≤ j ≤ S do
5: Sample p_{S(i−1)+j} ∼ x_i, and observe outcome y_{S(i−1)+j}.
6: end for
7: Feed the outcome ȳ_i := (1/S) ∑_{j=1}^S y_{S(i−1)+j} to TreeCal.
8: end for

To obtain a bound on the (expected) pure calibration error, we use a slight modification of TreeCal, namely SampleTreeCal (Algorithm 3). It functions identically to TreeCal except that for each time step t of TreeCal, it samples S actions from x_t on each of S contiguous time steps. (Hence, TreeCal is used with time horizon T/S.) At a high level, we will use an appropriate concentration inequality to show that the calibration upper bound of Theorem 3.1 implies a pure calibration upper bound for SampleTreeCal.

Theorem E.1 (Pure calibration error). Let P ⊂ ℝ^d be a bounded convex set and ‖·‖ be an arbitrary norm with unit dual ball L := { f ∈ ℝ^d | ‖f‖_* ≤ 1 }. Then, SampleTreeCal (Algorithm 3, with an appropriate choice of parameters H, L, S) guarantees that for an arbitrary sequence of outcomes y_1, …, y_T ∈ P, the ‖·‖^2 calibration error of its predictions x_1, …, x_T ∈ Δ(P) is bounded as follows:
E[PureCal_T^{‖·‖}(p_{1:T}, y_{1:T})] ≤ ε T, for T ≥ Rate(L, ‖·‖_*) · (diam_{‖·‖}(P)/ε)^{O(Rate(P, ‖·‖)/ε^2)}.

Proof. The proof uses Theorem 3.1 together with an appropriate concentration inequality, and closely follows that of [Pen25, Lemma 3.4].

Fix any 1 ≤ i ≤ T/S and 1 ≤ j ≤ S, and let 𝔽_{S(i−1)+j−1} denote the σ-algebra generated by y_1, …, y_{S(i−1)+j−1} and p_1, …, p_{S(i−1)+j−1}; since TreeCal is deterministic, it follows that x_1, …, x_i ∈ Δ(P) are 𝔽_{S(i−1)}-measurable. For any 1 ≤ j ≤ S, we have that, for any p ∈ supp(x_i),
E[(p − y_{S(i−1)+j}) · 1[p_{S(i−1)+j} = p] | 𝔽_{S(i−1)+j−1}] = (p − y_{S(i−1)+j}) · x_i(p).

Fixing any i ∈ [T/S], by Lemma E.2 applied to the sequence M_{S(i-1)+j} = (p - y_{S(i-1)+j}) \cdot \mathbb{1}[p_{S(i-1)+j} = p], for 1 ≤ j ≤ S (and the filtration \mathcal{F}_{S(i-1)+j}), we see that
$$\mathbb{E}\left[\left\|\sum_{j=1}^{S}(p - y_{S(i-1)+j}) \cdot \mathbb{1}[p_{S(i-1)+j} = p] - \sum_{j=1}^{S}(p - y_{S(i-1)+j}) \cdot \mathbf{x}_i(p)\right\|\right] \leq \operatorname{diam}_{\|\cdot\|}(\mathcal{P}) \cdot \sqrt{8S \cdot \operatorname{Rate}(\mathcal{L}, \|\cdot\|_*)}.$$
It follows by summing over the L values of p ∈ \mathrm{supp}(\mathbf{x}_i) that
$$\mathbb{E}\left[\sum_{p \in \mathcal{P}}\left\|\sum_{j=1}^{S}(p - y_{S(i-1)+j}) \cdot \mathbb{1}[p_{S(i-1)+j} = p] - \sum_{j=1}^{S}(p - y_{S(i-1)+j}) \cdot \mathbf{x}_i(p)\right\|\right]$$
$$\leq L \cdot \operatorname{diam}_{\|\cdot\|}(\mathcal{P}) \cdot \sqrt{8S \cdot \operatorname{Rate}(\mathcal{L}, \|\cdot\|_*)} \leq \epsilon \cdot S, \qquad (14)$$
as long as $$S \geq \frac{8 \cdot \operatorname{Rate}(\mathcal{L}, \|\cdot\|_*) \cdot \operatorname{diam}_{\|\cdot\|}(\mathcal{P})^2 \cdot L^2}{\epsilon^2}.$$

The guarantee of Corollary C.5 gives that, as long as $$T/S \geq \left(\frac{\operatorname{diam}_{\|\cdot\|}(\mathcal{P})}{\epsilon}\right)^{O(\operatorname{Rate}(\mathcal{P}, \|\cdot\|)/\epsilon^2)},$$ then
$$\operatorname{Cal}_T^{\|\cdot\|}(\mathbf{x}_{1:T/S}, \bar{y}_{1:T/S}) = \sum_{p \in \mathcal{P}}\left\|\sum_{i=1}^{T/S}\mathbf{x}_i(p) \cdot (p - \bar{y}_i)\right\|$$
$$= \sum_{p \in \mathcal{P}}\left\|\sum_{i=1}^{T/S}\frac{1}{S}\sum_{j=1}^{S}\mathbf{x}_i(p) \cdot (p - y_{S(i-1)+j})\right\| \leq \frac{\epsilon T}{S}. \qquad (15)$$

By combining Equations (14) and (15), it follows that for an arbitrary adaptive adversary who chooses a sequence y_1, …, y_T ∈ \mathcal{P},
$$\mathbb{E}\left[\operatorname{PureCal}_T^{\|\cdot\|}(p_{1:T}, y_{1:T})\right]$$
$$= \mathbb{E}\left[\sum_{p \in \mathcal{P}}\left\|\sum_{t=1}^{T}(p - y_t) \cdot \mathbb{1}[p_t = p]\right\|\right]$$
$$\leq \mathbb{E}\left[\sum_{p \in \mathcal{P}}\left\|\sum_{i=1}^{T/S}\sum_{j=1}^{S}(p - y_{S(i-1)+j}) \cdot \mathbf{x}_i(p)\right\| + \sum_{i=1}^{T/S}\sum_{p \in \mathcal{P}}\left\|\sum_{j=1}^{S}\big((p - y_{S(i-1)+j}) \cdot (\mathbb{1}[p_{S(i-1)+j} = p] - \mathbf{x}_i(p))\big)\right\|\right]$$
$$\leq 2\epsilon T.$$
The result follows by rescaling ε and our choice of L = O(\operatorname{Rate}(\mathcal{P}, \|\cdot\|)/\epsilon^2). □

As example applications of Theorem E.1:
• When \|\cdot\| is the \ell_1 norm and \mathcal{P} is the simplex, we have \operatorname{diam}_{\|\cdot\|}(\mathcal{P}) = 1, \mathcal{L} = \{f \in \mathbb{R}^d \mid \|f\|_\infty \leq 1\} satisfies \operatorname{Rate}(\mathcal{L}, \|\cdot\|_*) \leq d (as we can take the function R(x) = \|x\|_2^2), which gives that for T \geq d^{O(1/\epsilon^2)}, we can have \mathbb{E}[\operatorname{PureCal}_T^{\|\cdot\|_1}] \leq \epsilon T. This result recovers the main upper bound of [Pen25] (Theorem 1.1 therein).
• When \|\cdot\| is the \ell_2 norm and \mathcal{P} is the unit \ell_2 ball, we have \operatorname{diam}_{\|\cdot\|}(\mathcal{P}) = 1, \mathcal{L} = \{f \in \mathbb{R}^d \mid \|f\|_2 \leq 1\} satisfies \operatorname{Rate}(\mathcal{L}, \|\cdot\|_*) \leq 1 (as we can take the function R(x) = \|x\|_2^2), which gives that for T \geq \exp(O(1/\epsilon^2)), we can have \mathbb{E}[\operatorname{PureCal}_T^{\|\cdot\|_2}] \leq \epsilon T.

E.2 Sequential law of large numbers

Fix a convex set \mathcal{P} \subset \mathbb{R}^d and a norm \|\cdot\| on \mathbb{R}^d. We define
$$\mathcal{R}_n(\mathcal{P}, \|\cdot\|) := \sup_{\mathbf{p}} \mathbb{E}_{\epsilon}\left[\left\|\frac{1}{n}\sum_{i=1}^{n}\epsilon_i\,\mathbf{p}_i(\epsilon)\right\|\right].$$

where the supremum is over all sequences of mappings $\mathbf{p}_1, \ldots, \mathbf{p}_n$, where $\mathbf{p}_i : \{-1, 1\}^{i-1} \to \mathcal{P}$, and the expectation is over an i.i.d. sequence of Rademacher random variables $\epsilon = (\epsilon_1, \ldots, \epsilon_n)$, $\epsilon_i \sim \text{Unif}(\{\pm 1\})$. The below lemma (essentially contained in [RST15]) establishes a martingale law of large numbers for $\mathcal{P}$-valued martingales, in terms of geometric properties of $\mathcal{P}$ and $\|\cdot\|$.

Lemma E.2 ([RST15]). Consider a convex set $\mathcal{P} \subset \mathbb{R}^d$, a norm $\|\cdot\|$ on $\mathbb{R}^d$, and let $M_1, \ldots, M_n$ denote a sequence of random variables adapted to a filtration $(\mathcal{F}_i)_{i \in [n]}$. Let $\mathcal{L} = \{f \mid \|f\|_* \le 1\}$ be the unit ball of the dual norm $\|\cdot\|_*$. Then
$$
\mathbb{E}\left[\left\|\sum_{i=1}^n M_i - \mathbb{E}[M_i \mid \mathcal{F}_{i-1}]\right\|\right]
\le \operatorname{diam}_{\|\cdot\|}(\mathcal{P}) \cdot \sqrt{8n \cdot \operatorname{Rate}(\mathcal{L}, \|\cdot\|_*)}.
$$

Proof. By applying an appropriate translation to $\mathcal{P}$, we can assume that $\mathcal{P}$ contains the origin. We apply Theorem 2 of [RST15] with the domain $\mathcal{Z}$ equal to $\mathcal{P}$ and the function class $\mathcal{F}$ equal to the class of mappings $\{z \mapsto \langle z, f \rangle : \|f\|_* \le 1\}$ indexed by unit-dual norm linear functions on $\mathcal{Z}$. The theorem implies that
$$
\mathbb{E}\left[\frac{1}{n}\left\|\sum_{i=1}^n M_i - \mathbb{E}[M_i \mid \mathcal{F}_{i-1}]\right\|\right]
\le 2 \cdot \sup_{\mathbf{p}} \mathbb{E}_{\epsilon}\left[\sup_{\|f\|_* \le 1} \frac{1}{n}\left\langle\sum_{i=1}^n \epsilon_i \mathbf{p}_i(\epsilon), f\right\rangle\right]
= 2 \cdot \mathcal{R}_n(\mathcal{P}, \|\cdot\|).
$$
Write $\mathcal{L} = \{f \in \mathbb{R}^d : \|f\|_* \le 1\}$ to denote the unit ball for the dual norm $\|\cdot\|_*$. Proposition 16 of [RST15] gives that, if there is a function $R : \mathcal{L} \to \mathbb{R}$ which is 1-strongly convex with respect to $\|\cdot\|_*$ and which has range $\rho$, then $\mathcal{R}_n(\mathcal{P}, \|\cdot\|) \le \sqrt{\frac{2\rho}{n}} \cdot \operatorname{diam}_{\|\cdot\|}(\mathcal{P})$. In particular,
$\mathcal{R}_n(\mathcal{P}, \|\cdot\|) \le \operatorname{diam}_{\|\cdot\|}(\mathcal{P}) \cdot \sqrt{\frac{2\,\operatorname{Rate}(\mathcal{L}, \|\cdot\|_*)}{n}}$. $\square$