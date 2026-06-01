# Quantifying Agency: A Causal Information-Theoretic Taxonomy of Autonomous Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

An information-theoretic diagnostic framework designed to measure and monitor the *informational agency*—the degree to which an intelligent agent possesses an autonomous internal world-model independent of immediate environmental or supervised constraints—using the **Mapmaker Index (MI)**.

## 📌 Table of Contents
- [I. Abstract](#i-abstract)
- [II. Theoretical Framework & Mathematics](#ii-theoretical-framework--mathematics)
  - [A. Formulation of the Mapmaker Index (MI)](#a-formulation-of-the-mapmaker-index-mi)
  - [B. Prototypical Operational Regimes](#b-prototypical-operational-regimes)
- [III. System Architecture & Pipeline](#iii-system-architecture--pipeline)
- [IV. Empirical Reference Suite](#iv-empirical-reference-suite)
  - [A. Cross-Architecture Baseline Results](#a-cross-architecture-baseline-results)
  - [B. The Paradox of Noise & Null Control Sweeps](#b-the-paradox-of-noise--null-control-sweeps)
- [V. Algorithmic Implementations](#v-algorithmic-implementations)
  - [A. Non-Parametric KSG Conditional Mutual Information Estimator](#a-non-parametric-ksg-conditional-mutual-information-estimator)
  - [B. Statistical Bootstrap Diagnostics](#b-statistical-bootstrap-diagnostics)
- [VI. Discussion & AI Safety Implications](#vi-discussion--ai-safety-implications)
- [VII. Citation & References](#vii-citation--references)

---

## I. Abstract

Current methodologies in artificial intelligence prioritize performance optimization on narrow benchmarks, yet they lack formal, quantitative frameworks to measure and diagnose *informational agency*—the degree to which an agent possesses an autonomous internal world-model independent of immediate environmental or supervised constraints. This paper introduces the **Mapmaker Index (MI)**, a novel diagnostic framework rooted in conditional mutual information ($I(S_t; S_{t+1} | D)$), designed to isolate the residual autonomy of an agent's internal latent states ($S$) by conditioning on the extrinsic constraint manifold ($D$) dictating its training or environment. 

Through a systematic empirical sweep, we benchmarked three architectural paradigms: a rule-based Finite State Machine (FSM), a supervised Multi-Layer Perceptron (MLP) trained on MNIST, and a goal-directed Proximal Policy Optimization (PPO) reinforcement learning agent trained on *LunarLander-v3*. Our findings reveal a distinct **"learning-autonomy gap."** Contrary to prevailing assumptions in machine learning taxonomy, the supervised MLP failed to exhibit autonomous internal dynamics, converging to a conditional MI of **0.0450** and placing it firmly alongside the FSM ($MI_{cond} = 0.0000$) within the **Reflexive Regime**. Conversely, the PPO agent successfully crossed into the **Mapmaker Regime**, achieving a baseline conditional MI of **0.4743**. 

To verify the causal independence of this emergent world-model, we subjected the systems to an informational stress test by systematically injecting Gaussian noise into the constraint manifold. While traditional reactive control metrics predicted a catastrophic failure of internal representation, the PPO agent's conditional MI instead exhibited a counter-intuitive expansion, peaking at **0.9685** under high-noise thresholds ($\sigma = 5.0$). This **"Paradox of Noise"** provides empirical validation that the agent's internal generative dynamics decouple from extrinsic feedback when environmental predictability degrades. Ultimately, this framework provides an actionable, mathematically rigorous toolset for AI safety, alignment, and interpretability, offering a prospective "early warning diagnostic" to monitor the transition of artificial systems from reactive statistical models to self-directed generative agents.

---

## II. Theoretical Framework & Mathematics

### A. Formulation of the Mapmaker Index (MI)
The absolute information flow wrapped within the internal state transition is captured by the unconditional mutual information $I(S_t; S_{t+1})$, which measures how much information the current internal state shares with the subsequent internal state. To isolate the agent's independent world-modeling capacity, we remove the information contributed by the extrinsic constraints. We formalize the **Mapmaker Index (MI)** as the conditional mutual information:

$$MI_{cond} = I(S_t; S_{t+1} | D)$$

Expanding this definition via Shannon entropy terms:

$$MI_{cond} = H(S_t | D) - H(S_t | S_{t+1}, D)$$

Alternatively, using the joint and conditional probability distributions over the support spaces $\mathcal{S}$ and $\mathcal{D}$, the explicit integration is given by:

$$MI_{cond} = \sum_{s_t \in \mathcal{S}} \sum_{s_{t+1} \in \mathcal{S}} \sum_{d \in \mathcal{D}} p(s_t, s_{t+1}, d) \log \left( \frac{p(s_t, s_{t+1} | d)}{p(s_t | d)p(s_{t+1} | d)} \right)$$

This metric successfully operates as an informational bottleneck. By conditioning the transition on $D$, any state changes that are purely reactive or structurally dependent on the external target labels, actions, or rewards are canceled out.

### B. Prototypical Operational Regimes
Rather than asserting absolute, invariant boundaries, we leverage the endpoints of our empirical evaluations to propose a prototypical taxonomy of informational agency based on relative operational thresholds of $MI_{cond}$:

* **Reflexive Regime ($MI_{cond} < 0.10$):** Characterized by near-complete conditional independence where $p(s_{t+1} | s_t, d) \to p(s_{t+1} | d)$. The system's internal state transitions are structurally tethered to external forcing functions, leaving minimal residual autonomy.
* **Transitional Adaptive Region ($0.10 \le MI_{cond} < 0.30$):** A theoretical regime intended for systems exhibiting partial representation mapping alongside external feedback dependency. Within our narrow three-paradigm exploratory sweep, this region formed an unpopulated boundary, suggesting a potentially sharp informational phase transition between feedforward representation and autonomous temporal persistence.
* **Mapmaker Regime ($MI_{cond} \ge 0.30$):** Defined by a substantial operational divergence where $I(S_t; S_{t+1} | D)$ remains bounded away from zero. Here, internal state-to-state information flow dominates the transition probability matrix, proving the existence of an independent internal world-model.

---

## For more info in detail refer the mi.pdf file
