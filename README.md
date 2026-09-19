# micrograd-from-scratch

Building of reverse-mode automatic differentiation engines from scratch in Python, as a
structured self-study project.
The emphasis throughout is on deriving and implementing the machinery myself rather than using 
framework autograd.

Starts from Karpathy's [micrograd](https://karpathy.ai/zero-to-hero.html) with a layer of self-directed experiments on top investigating:

(i) varying
activations for layers; tanh, linear, RELU, sigmoid as well as a mix and learnable blend of them,

(ii) different optimisers for gradient descent; SGD, AdaGrad, RMSProp, Adam.

The experiments performed and the results of changing the above are given below.

## The Neural Network
### Scalar Engine

- `engine.py`: a `Value` class implementing reverse-mode automatic differentiation;
  arithmetic operations with backward closures, topological-sort, and
  gradient accumulation.
- `nn.py`: `Neuron`/`Layer`/`MLP` built on the engine, with configurable activations
  including a learnable per-neuron tanh/ReLU blend (continuous relaxation of a discrete
  architecture choice, trained by gradient descent).
- Optimiser implemented from first principles: SGD, AdaGrad, RMSProp, Adam with
  bias correction - each motivated by the failure mode of the previous.
- `num_grad.py`: central-difference gradient checker; all operations verified to ~1e-9.

### Tensor Engine

- `tensor`: this folder contains versions of the engine and neural network adapted to tensor framework using pytorch, and includes a comparison to the scalar versions.
- `tensor/tensor.py`: a NumPy-backed autograd engine; broadcasting with correct
  backward (sum over broadcast axes via a two-step unbroadcast), matmul backward as
  vector-Jacobian products, reductions.
- `tensor/nn_tensor.py`: vectorised MLP; each layer is one matmul node rather than a
  graph of scalar operations.

### Training Loop and Data

- `demo.py`: training loop of the neural network, including the data, optimisers and fixed seed, with plots.
- `demo_random_restart.py`: like demo, but with multiple starting seeds for random restarts experiment.


## Miscellaneous

- `num_grad.py`: manually calculates loss to compare with loss.backward()
- `time_scaling.py`: measures how per-iteration training cost grows with network size on the scalar engine. It loops over architectures MLP(3, [w, w, 1]) for w in {4, 16, 32, 64, 128}, and for each one times five full training iterations (forward pass over the 4-example dataset, zero_grad, backward, SGD update), averages them, and prints the parameter count alongside the time.
- `visualise`, `visualise_demo`: visualisation of backward pass using graphs

## Experiments and Analysis

The loss-landscape example is XOR (except for initial linear data), on which the folliowing is performed; activation comparisons, multi-seed distributions of the
learned activation blend, per-neuron training trajectories, random restarts. 

Then there is a quantitative stress test of the scalar engine - time scaling linear in parameters, the
recursion-depth wall induced by Python's `sum()` building linear graph chains, and a
measured ~10^4x scalar-vs-vectorised gap on identical computations, motivating the
tensor design empirically.

## Results

### Activation Architecture

![Toy regression](figs/toy_lin_wins.png)

On the four-point toy regression set, the purely linear network converges fastest. The
target is close to linear, so nonlinearity is not effective and only slows optimisation - the simplest model winning
shows the importance of the shape of the data in the architecture of the model.

![XOR three-way](figs/XOR_three_way.png)

XOR is not linearly separable, so the linear network plateaus at chance. ReLU descends
faster early while tanh is smoother later; a per-layer mix of the two beats both pure
variants throughout, capturing each one's advantage in the regime where it holds.

### Optimisers

![Stalling](figs/rms_stalls_mix.png)

AdaGrad's accumulating denominator drives the effective learning rate
monotonically toward zero and training stalls on every architecture except the mixed
one, which retains enough gradient signal to keep descending.

![Seed sensitivity](figs/ada_seed43.png)

Repeating the comparison on a different seed with everything else held fixed, the mixed
architecture no longer holds a clear advantage, and moving from AdaGrad to
RMSProp changes little. The apparent advantage above is seed-dependent: the honest
reading is that mixing helps sometimes rather than reliably, which is why the population
analyses below use multiple seeds rather than one.

![Adam vs adaptive-only](figs/adam_v_adagrad_blends.png)

Adding momentum on the gradient itself damps the oscillation. The
adaptive-without-momentum variants swing over orders of magnitude while the Adam
variants descend smoothly, with the effect most pronounced on the mixed architecture.

### Learned per-neuron blending

![Alpha trajectories](figs/neuron_trajectory.png)

Each neuron carries a learnable blend parameter between tanh and ReLU, trained by
gradient descent alongside the weights, to test whether neurons specialise or converge on
a shared activation. Trajectories start bunched near the midpoint, fan out, cross, and
freeze as the loss bottoms out - specialisation emerges during training rather than being
fixed at initialisation.

![Restarts and pooled alphas](figs/rrestart_pooled_alpha.png)

Left image: final loss across 15 random restarts, most converging to ~1e-9 with one seed stuck
several orders of magnitude higher, which is why single-seed results are not evidence.

Right image: pooling converged blend parameters across 120 neurons gives a roughly bimodal
distribution with a slight ReLU lean and the midpoint least populated. Given the freedom,
neurons commit toward one activation rather than averaging them.

![XOR five-way](figs/XOR_5way.png)

Both the fixed per-layer mix and the learned per-neuron blend reach ~1e-5 while the pure
activations plateau three to four orders of magnitude higher. The learned blend matches
the hand-specified mix without being told which activation to use where.


### Timing
The forward pass creates roughly 2P objects, since each parameter contributes a multiply and an add, which is why operation count tracks parameter count and time tracks both.

Linear in parameters and quadratic in width are the same fact on different axes. A dense layer of width w fed by width w has w² weights, so P is quadratic in w. Confirmed in the data: width 16 to 32 doubled the width, took params from 353 to 1217 (about 3.5×), and time from 3.4 to 11.5 ms (about 3.4×).

The constant kills it rather than the growth rate; at 17k parameters it is roughly a quarter of a second per iteration. Extrapolating linearly, 1M parameters gives roughly 14 s/iter, so hours for a single training run, and anything at 100M+ is out of reach.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python demo.py
```


All code is my own implementation; external libraries are limited to NumPy and matplotlib.
