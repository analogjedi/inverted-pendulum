# Inverted Pendulum Control Lab

An interactive web simulation of the classic inverted pendulum on a cart, featuring five classical controllers and a neural network reinforcement learning controller. Runs entirely in the browser — no dependencies, no build step, no server.

![Screenshot](https://img.shields.io/badge/status-active-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue)

## Quick Start

Open `index.html` in any modern browser. That's it.

## Features

- **Six controllers** — PD, LQR, Swing-Up + LQR, MPC, and a trainable Neural Network (PPO)
- **Full nonlinear physics** — coupled equations of motion solved with RK4 at 500 Hz
- **Real-time training** — watch the RL agent learn to balance in your browser
- **Interactive** — apply forces with arrow keys/mouse, perturb with P, tune parameters with sliders
- **Save/Load** — export trained network weights to JSON, reload them later
- **Zero dependencies** — vanilla JS, HTML5 Canvas, hand-rolled linear algebra

## Controls

| Input | Action |
|-------|--------|
| ← → | Apply force to cart |
| P | Random perturbation (velocity impulse) |
| Space | Pause / Resume |
| R | Reset simulation |

## System Parameters

All parameters are tunable via sliders in the left panel:

| Parameter | Symbol | Default | Range |
|-----------|--------|---------|-------|
| Cart mass | M | 1.0 kg | 0.2–5.0 |
| Pendulum mass | m | 0.3 kg | 0.05–2.0 |
| Pendulum length | L | 0.5 m | 0.1–2.0 |
| Cart friction | d | 0.1 | 0–2.0 |

---

## Physics

The simulation uses the full nonlinear coupled equations of motion for a pendulum on a cart, where θ = 0 is upright:

```
(M + m)ẍ + mLθ̈cosθ − mLθ̇²sinθ = u − dẋ
mL²θ̈ + mLẍcosθ − mgLsinθ = 0
```

Solved for accelerations:

```
D = M + m − mcos²θ
ẍ = (u − dẋ + mLθ̇²sinθ − mgsinθcosθ) / D
θ̈ = (−ucosθ + dẋcosθ − mLθ̇²sinθcosθ + (M+m)gsinθ) / (LD)
```

Integrated with **4th-order Runge-Kutta** at dt = 0.002s (8 substeps per render frame).

### The Non-Minimum Phase Property

This system has a counter-intuitive property: pushing the cart right causes the pendulum to initially rotate *left* (negative angular acceleration). To correct a rightward tilt, you push the cart *right* — chasing the fall, like balancing a broomstick on your palm. All controllers must respect this sign convention.

---

## Controllers

### 1. PD (Proportional-Derivative)

The simplest controller. Linear state feedback with tunable gains:

```
u = Kp·θ + Kd·θ̇ + Kx·x + Kxd·ẋ
```

- **Kp** (default 150) and **Kd** (default 30) are tunable via sliders
- Position feedback gains Kx = 8.0, Kxd = 6.0 are fixed
- All terms use **positive feedback** due to the non-minimum phase dynamics
- Starts near upright (θ₀ ≈ 1°) — cannot recover from large angles
- No integral term, so some steady-state error is expected

**When to use:** Understanding the basics. Good for small perturbations but limited robustness.

### 2. LQR (Linear-Quadratic Regulator)

Optimal linear controller computed by solving the Continuous Algebraic Riccati Equation:

```
u = −[K₁x + K₂ẋ + K₃θ + K₄θ̇]
```

The gains K are computed from the **linearized** system matrices around the upright equilibrium:

```
A = [[0, 1, 0, 0],
     [0, -d/M, -mg/M, 0],
     [0, 0, 0, 1],
     [0, d/(ML), (M+m)g/(ML), 0]]

B = [0, 1/M, 0, -1/(ML)]
```

With cost weights Q = diag(10, 1, 100, 10) and R = 0.01.

The CARE is solved via ODE integration (Ṗ = AᵀP + PA − PBR⁻¹BᵀP + Q) with forward Euler, converging to steady state. Falls back to hand-tuned gains if the solver doesn't converge.

**Recomputes automatically** when you change system parameters.

**When to use:** The most reliable controller. Handles moderate perturbations and keeps the cart centered.

### 3. Swing-Up + LQR

Starts with the pendulum hanging down and swings it up to vertical, then catches it with LQR.

**Swing-up phase** uses the Åström-Furuta energy pumping method:

```
E = ½mL²θ̇² + mgLcosθ           (current energy)
E_up = mgL                        (energy at upright)
u = Ke·(E − E_up)·sign(θ̇cosθ)   (pump or brake)
```

The controller adds energy when E < E_up and removes it when E > E_up, with the sign term ensuring force is applied in the energy-increasing direction. Cart centering terms (−2x − 1.5ẋ) prevent runaway.

**Catch transition:** When |θ| < 55° (tunable), switches to LQR for stabilization.

**When to use:** The most dramatic demonstration. Starts from θ = π (hanging) and autonomously swings to upright.

### 4. MPC (Model Predictive Control)

Looks ahead N steps into the future and optimizes the control sequence:

- **Horizon:** N = 15 steps (tunable)
- **Cost:** Q_theta = 100 (angle penalty), R = 0.01 (effort penalty)
- **Optimizer:** 5 passes of gradient descent with finite-difference gradients (ε = 0.5)
- **Warm start:** Initialized with LQR output, decayed over the horizon
- **Force limit:** ±60N

Falls back to swing-up control when |θ| > 108°.

This is a simplified single-shooting MPC (not a full QP solver), but it demonstrates the predictive control concept. Computationally heavier than the other controllers.

**When to use:** Exploring predictive control. Handles nonlinear regions better than LQR since it uses the full nonlinear model for prediction.

### 5. Neural Network (PPO Reinforcement Learning)

A neural network trained from scratch in the browser using Proximal Policy Optimization. No pre-trained weights — you watch it learn.

#### Architecture

- **Actor:** 6 → 64 → 64 → 1 MLP (tanh hidden layers, linear output × maxForce)
- **Critic:** 6 → 64 → 64 → 1 MLP (tanh hidden layers, linear value output)
- **Policy:** π(a|s) = N(μ, σ²) — continuous Gaussian over force values
- **State inputs:** [x, ẋ, sin(θ), cos(θ), θ̇, previous_force / maxForce]

The sin/cos angle representation eliminates discontinuity at ±π, and the previous force input lets the network learn to smooth its own control output.

#### Training

- **Algorithm:** PPO with clipped surrogate objective
- **Advantages:** Generalized Advantage Estimation (GAE, λ = 0.95)
- **Rollout buffer:** 2048 steps, split into 64-sample mini-batches, 4 epochs per update
- **Exploration:** Learnable σ via logStd parameter, initialized to σ ≈ 7.4N for broad exploration
- **Gradient clipping:** Global norm capped at 5.0
- **All gradients are hand-rolled** — no autograd library, analytical backpropagation through tanh layers with maxForce scaling

#### Reward Function

The reward is shaped to encourage robust, quiet balancing:

| Term | Weight | Purpose |
|------|--------|---------|
| cos(θ) | 1.0 | Stay upright |
| −x² | 0.01 | Stay centered |
| exp(−10θ² − θ̇² − 0.5ẋ²) | 1.0 | Stability bonus — peaked at perfect stillness |
| timeRemaining × exp(−5·velSq) | 1.0 | Settling bonus — rewards fast recovery after perturbation |

The settling bonus activates after perturbations and rewards bringing velocities to zero within 4 seconds. Faster settling earns more reward (time-weighted payout).

No velocity penalties are used — they conflict with the aggressive corrections needed for disturbance recovery.

#### Training Perturbations

During training, random perturbations are injected mid-episode (0.5% chance per step, calibrated to ≤40N equivalent impulse). This forces the network to practice recovery, not just gentle balancing.

#### Save / Load

Trained weights can be saved to a JSON file and loaded later. The file includes actor, critic, and best-actor weights, logStd, episode count, reward history, and maxForce setting.

**When to use:** Learning how RL works on a real control problem. Train at speed 10–20, watch the reward curve climb, then stop training and test with perturbations.

---

## File Structure

| File | Purpose |
|------|---------|
| `index.html` | HTML/CSS layout, UI controls, simulation loop, rendering |
| `physics.js` | Nonlinear equations of motion, RK4 integrator |
| `pd.js` | PD controller |
| `lqr.js` | LQR controller (CARE solver) |
| `swingup.js` | Swing-Up + LQR (Åström-Furuta energy method) |
| `mpc.js` | Simplified nonlinear MPC |
| `ppo.js` | RL controller — ActorNet, CriticNet, PPO trainer, save/load |

## Acknowledgments

Inspired by a MathWorks LinkedIn post demonstrating PD, LQR, and MPC control in MATLAB. Built with [Claude Code](https://claude.ai/code).
