# micrograd-from-scratch

**Reverse-mode automatic differentiation, neural networks, and optimisation implemented in Python.**

I built this project to understand how neural-network training works beneath a framework: specifically how a computation becomes a graph, how gradients flow through it, and how those gradients update the model. Starting from Andrej Karpathy's [micrograd lesson](https://karpathy.ai/zero-to-hero.html), I extended the scalar implementation with a NumPy-backed tensor engine, numerical gradient checks, optimisation experiments, and learnable activation functions.

## Overview

- **Two autodiff engines:** a scalar engine with explicit computation graphs, and a tensor engine with manually implemented backward rules for broadcasting, reductions and 2-D matrix multiplication. Neither uses framework autograd.
- **Numerical verification:** central finite-difference checkers for scalar and tensor parameters. The included tensor checks produce maximum absolute errors of approximately $2\times10^{-9}$ on their tested inputs.
- **Optimisation from first principles:** handwritten SGD and adaptive updates, including RMSProp and Adam variants, with bias correction implemented in the random-restart script.
- **Independent experiments:** trainable tanh/ReLU mixtures for individual neurons; activation comparisons, learning trajectories, and 15 random restarts on XOR.
- **Performance investigation:** measured scalar training cost as network size grows and investigated recursion failures caused by deep computation graphs, motivating the tensor implementation.

The extensions beyond the introductory scalar implementation are the tensor engine, numerical verification, adaptive optimisation experiments, learnable activation blends, and performance investigation.

## What I Built

| Component | Implementation | What it makes explicit                                                                                                                                                    |
| --- | --- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Scalar autodiff | [`engine.py`](engine.py) | Each operation records its inputs and backward rule. Reverse topological traversal propagates gradients; shared inputs accumulate contributions from every path.          |
| Neural-network layers | [`nn.py`](nn.py) | Neurons, layers and MLPs compose the scalar operations. Hidden layers support tanh, ReLU, linear, or learned blended activations.                                         |
| Tensor autodiff | [`tensor/tensor.py`](tensor/tensor.py) | NumPy handles array arithmetic; the engine implements differentiation itself, including summing gradients over broadcast axes and backward rules for 2-D matrix products. |
| Vectorised layers | [`tensor/nn_tensor.py`](tensor/nn_tensor.py) | A layer's linear transformation becomes one matrix-multiplication node, followed by bias addition and activation, instead of many scalar graph nodes.                     |
| Numerical checks | [`num_grad.py`](num_grad.py), [`tensor/num_grad_tensor.py`](tensor/num_grad_tensor.py) | Compare backpropagated derivatives with independently computed central finite differences.                                                                                |

NumPy supplies tensor storage and numerical operations; matplotlib supplies plots. Graphviz is used for computation-graph visualisation.

## Experiments and Findings

### Can neurons learn their own activation mixture?

I extended each hidden neuron with an activation $f(z)=\alpha\tanh(z)+(1-\alpha)\operatorname{ReLU}(z)$, where $\alpha=\operatorname{sigmoid}(a)$ and $a$ is trained alongside the weights and bias. Each blended neuron therefore has one additional trainable parameter. Values of $\alpha$ near 1 favour tanh; values near 0 favour ReLU.

I compared pure activations, a fixed tanh/ReLU choice across layers, and learned mixtures on the four-point XOR problem, using squared-error training loss.

![XOR training losses for linear, tanh, ReLU, fixed mixed and learned blended activations](figs/XOR_5way.png)

**Observation:** in the illustrated run, the fixed mix and learned blend reached lower training loss than the pure activations. The blend learned different coefficients across neurons.

**Interpretation:** this demonstrates that activation coefficients can be trained through the same autodiff engine. It does not establish a reliable performance advantage: these are small training-set experiments, the blend adds parameters, and resetting the same random seed does not give the blend identical starting weights because its extra parameters consume additional random draws.

### How sensitive are the results to initialisation?

Changing the seed weakened an apparent advantage of the mixed architecture. I then examined the blended network across 15 random restarts, recording training losses and the final coefficients of its eight hidden neurons.

![Training losses across 15 random restarts and pooled activation coefficients from 120 neurons](figs/rrestart_pooled_alpha.png)

**Observation:** final losses vary substantially across initialisations. The pooled activation coefficients are spread across both sides of the midpoint, with fewer near an equal mixture in this sample.

**Interpretation:** one successful run is insufficient to establish a robust advantage. These restarts characterise the blended model; establishing superiority over tanh or ReLU would require corresponding baseline runs. The 120 neuron coefficients are descriptive observations from 15 networks, not 120 independent experimental replications. The experiments investigate optimisation on XOR, rather than generalisation to unseen data.

### What changes when the optimiser changes?

The experiments explore SGD, accumulated squared-gradient scaling (AdaGrad), exponentially averaged squared-gradient scaling (RMSProp), and Adam-style first and second moments. Their purpose is to investigate stalling, oscillation and sensitivity to activation choice.

The current code retains SGD, RMSProp and an uncorrected Adam-style update in [`demo.py`](demo.py). AdaGrad's accumulator update remains as a commented alternative inside the RMSProp routine. [`demo_random_restart.py`](demo_random_restart.py) contains both uncorrected and bias-corrected Adam variants: the restart distribution uses the former, while the neuron-trajectory plot uses the latter.

These plots are exploratory examples under particular settings, rather than a general ranking of optimisers or an isolated test of momentum's causal effect.

### Why build a tensor engine?

The scalar engine constructs Python objects and backward closures for individual arithmetic operations. In the measured small-network examples, iteration time grew approximately with parameter count; for dense layers of similar width, parameter count grows roughly quadratically with width.

The investigation also exposed a separate limit: Python's `sum()` builds a long chain of addition nodes, which can exceed the recursion limit during the engine's recursive graph traversal.

These observations motivated representing whole array operations as graph nodes. The tensor engine retains explicit differentiation rules while moving numerical array work into NumPy.

**Benchmark status:** `time_scaling.py` includes a scalar/tensor timing comparison, but its tensor branch uses a 1-D vector input. The current matrix-multiplication backward rule supports 2-D operands and gives incorrect weight gradients for that benchmark input. The earlier approximately $10^4$ speedup claim is therefore withheld pending a corrected benchmark and gradient-equivalence check. The tensor training demo uses 2-D column inputs.

<details>
<summary>Additional experiment plots</summary>

These figures record exploratory runs. Their settings were edited during development; the default scripts do not regenerate every historical figure in one command.

**Toy regression and activation comparisons**

![Training on the four-point toy regression dataset](figs/toy_lin_wins.png)

![XOR activation comparison](figs/XOR_three_way.png)

**Adaptive updates and seed sensitivity**

![Training curves illustrating stalling under the recorded settings](figs/rms_stalls_mix.png)

![Activation comparison under a different seed](figs/ada_seed43.png)

![Recorded comparison of Adam-style and adaptive-only updates](figs/adam_v_adagrad_blends.png)

**Learned activation trajectories**

![Individual neuron activation coefficients during bias-corrected Adam training](figs/neuron_trajectory.png)

</details>

## Run the code

From the repository root, create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the scalar activation demo or the random-restart experiment:

```bash
python demo.py
python demo_random_restart.py
```

Run the tensor checks and training demo:

```bash
python tensor/test_tensor.py
python tensor/grad_comparison_tensor.py
python tensor/demo_tensor.py
```

The tensor checks cover selected arithmetic and activation cases, broadcast shapes, reductions, and square/non-square 2-D matrix products. They are examples of numerical verification, not exhaustive coverage of all operations and shapes. Some comparisons print an `OK`/`FAIL` status rather than raising an assertion.

For further inspection, [`time_scaling.py`](time_scaling.py) contains the scaling and recursion experiments, and [`visualise.py`](visualise.py) / [`visualise_demo.py`](visualise_demo.py) contain graph visualisation code. Rendering graphs also requires the Graphviz system executable.

## Scope

This is an implementation and experimentation project built around small, inspectable examples. The scalar foundation follows Karpathy's micrograd lesson; the extensions explore verification, optimisation, activation learning and the move from scalar to tensor computation. The current tensor engine supports a deliberately limited operation set, with scalar-loss backpropagation and 2-D matrix-multiplication backward rules.
