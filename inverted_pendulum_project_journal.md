# Inverted Pendulum Control Lab — Project Journal

## Project Overview

A multi-file interactive web simulation implementing a nonlinear inverted pendulum on a cart with five classical controllers and a neural network reinforcement learning controller. The simulation runs entirely client-side with no dependencies — pure vanilla JS, canvas rendering, and hand-rolled linear algebra.

**Entry point:** `index.html`
**Deployment:** Static files, Vercel-ready
**Origin:** Inspired by a MathWorks LinkedIn post demonstrating PD, LQR, and MPC control in MATLAB.

---

## 1. Physics Engine

### 1.1 Nonlinear Equations of Motion

The simulation uses the **full nonlinear** coupled equations, not the linearized approximation:

$$
(M + m)\ddot{x} + mL\ddot{\theta}\cos\theta - mL\dot{\theta}^2\sin\theta = u - d\dot{x}
$$

$$
mL^2\ddot{\theta} + mL\ddot{x}\cos\theta - mgL\sin\theta = 0
$$

Solved for accelerations:

$$
D = M + m - m\cos^2\theta
$$

$$
\ddot{x} = \frac{u - d\dot{x} + mL\dot{\theta}^2\sin\theta - mg\sin\theta\cos\theta}{D}
$$

$$
\ddot{\theta} = \frac{-u\cos\theta + d\dot{x}\cos\theta - mL\dot{\theta}^2\sin\theta\cos\theta + (M+m)g\sin\theta}{LD}
$$

### 1.2 State Vector

$$
X = [x, \dot{x}, \theta, \dot{\theta}]^T
$$

where θ = 0 is upright (pendulum above cart).

### 1.3 Integration

**RK4** at 500 Hz (dt = 0.002s), with 8 substeps per render frame. This gives stable integration even during aggressive control inputs.

### 1.4 System Parameters (user-tunable)

| Parameter | Symbol | Default | Range |
|-----------|--------|---------|-------|
| Cart mass | M | 1.0 kg | 0.2–5.0 |
| Pendulum mass | m | 0.3 kg | 0.05–2.0 |
| Pendulum length | L | 0.5 m | 0.1–2.0 |
| Cart friction | d | 0.1 | 0–2.0 |

### 1.5 Sign Convention (Critical)

This system is **non-minimum phase**. The key coupling term is:

$$
\ddot{\theta} \propto \frac{-u\cos\theta}{LD}
$$

**Positive force u → negative θ̈** (when near upright). This means:
- Pendulum tilts right (θ > 0) → push cart **right** (u > 0) → creates negative θ̈ → corrects tilt
- This is counter-intuitive: you "chase" the fall, like balancing a broomstick on your palm

This sign convention propagates to position feedback as well: cart drifts right → push right → tips pendulum left → gravity pulls system back left. **All feedback terms have the same sign as the error they correct.**

---

## 2. Classical Controllers

### 2.1 PD Controller

```
u = +Kp·θ + Kd·θ̇ + Kx·x + Kxd·ẋ
```

- Kp = 150, Kd = 30 (user-tunable)
- Kx = 8.0, Kxd = 6.0 (hardcoded position feedback)
- All terms are **positive feedback** due to non-minimum phase dynamics
- Starts at θ₀ = 0.02 rad (~1°) — larger angles cause saturation and divergence

**Known limitation:** PD has no integral term, so steady-state tracking is imperfect. Adequate for small perturbations but less robust than LQR.

### 2.2 LQR Controller

```
u = −KX = −[K₁x + K₂ẋ + K₃θ + K₄θ̇]
```

Gains K are computed by solving the **Continuous Algebraic Riccati Equation (CARE)** via ODE integration:

$$
\dot{P} = A^TP + PA - PBR^{-1}B^TP + Q
$$

Integrated to steady state with forward Euler (h = 0.0005, up to 40,000 iterations, convergence check at ‖dP‖ < 10⁻⁶).

**Linearized system matrices** (around θ = 0):

```
A = [[0, 1, 0, 0],
     [0, -d/M, -mg/M, 0],
     [0, 0, 0, 1],
     [0, d/(ML), (M+m)g/(ML), 0]]

B = [0, 1/M, 0, -1/(ML)]
```

**Cost weights:**
- Q = diag(10, 1, 100, 10)
- R = 0.01

Fallback to hand-tuned gains [-10, -12, -120, -30] if CARE doesn't converge (e.g., extreme parameter combinations).

**Recomputes on parameter change.** This is the most reliable controller in the system.

### 2.3 Swing-Up + LQR

Energy-based swing-up (Åström-Furuta method) with LQR catch:

**Energy computation:**
$$
E = \frac{1}{2}mL^2\dot{\theta}^2 + mgL\cos\theta
$$

- Upright: E_up = +mgL (maximum)
- Hanging: E_hang = −mgL (minimum)

**Swing-up control law:**
$$
u = K_e \cdot (E - E_{up}) \cdot \text{sign}(\dot{\theta}\cos\theta) - 2x - 1.5\dot{x}
$$

The energy pumping adds energy when E < E_up and removes it when E > E_up. The sign term ensures force is applied in the direction that increases pendulum energy.

**Mode switch:** When |θ| < catch angle (default 55°), switches to LQR.

**Cart centering:** The −2x − 1.5ẋ terms prevent the cart from running away during swing-up.

Starts at θ₀ = π + 0.05 (hanging down, slight offset to break symmetry).

### 2.4 MPC Controller

Simplified nonlinear MPC with gradient-descent optimization:

- Horizon: N = 15 steps (tunable)
- Cost: Q_theta = 100, R = 0.01 (tunable)
- Optimization: 5 passes of single-shooting with finite-difference gradients (ε = 0.5)
- Warm-started with LQR output, decayed over horizon
- Force clamped to ±60N

Falls back to swing-up controller when |θ| > 0.6π.

**Note:** This is a simplified MPC, not a full QP solver. It works but is computationally expensive per frame due to the forward simulations inside the optimizer.

---

## 3. Neural Network RL Controller — Current State

### 3.1 Architecture Evolution

The RL controller went through six major iterations.

#### Attempt 1: Gaussian Continuous REINFORCE (FAILED)
- **Network:** 4→64→64→1, linear output = force mean μ
- **Policy:** π(a|s) = N(μ_θ(s), σ²), σ annealed from 10→1.5
- **Bug:** `controllerNN()` was called on every physics substep (8/frame), drawing fresh noise each time. Only the last action was recorded. Trajectory data was inconsistent with actual forces applied.
- **Result:** Flat reward at ~-200, no learning after 2,500 episodes.

#### Attempt 2: Gaussian + Curriculum (MARGINAL)
- Fixed substep bug (compute action once, cache for all substeps)
- Added curriculum learning (start at ±0.05 rad, widen on success)
- **Bug:** Curriculum threshold was `maxSteps × 1.5 = 900`, but with exploration noise, average reward never exceeded ~250 during training. Threshold was unreachable.
- **Result:** Mastered ±2.9° window perfectly (best ~596), but curriculum never advanced. The network overfit to a trivial regime.

#### Attempt 3: Gaussian + Fixed Curriculum (MARGINAL)
- Lowered curriculum threshold to `maxSteps × 0.3`
- Added σ bump on curriculum advancement
- Relaxed gradient clipping from ±0.1 to ±0.5
- Curriculum advanced to ~30°
- **Core issue:** Gaussian policy gradient ∇log N(μ,σ²) = (a−μ)/σ² is a single noisy scalar. Insufficient signal for convergence with REINFORCE over 500+ step episodes.
- **Result:** Avg ~137, flat learning curve at 9,000+ episodes.

#### Attempt 4: Discrete Softmax, 1 Hidden Layer (FAILED)
- **Network:** 4→32→11, softmax output over 11 force levels
- Clean gradient: ∇log π = one_hot(a) − probs (updates all logits simultaneously)
- **Problem:** Single hidden layer lacks capacity for nonlinear state→action mapping.
- **Result:** No meaningful learning.

#### Attempt 5: Soft-DAC, 2 Hidden Layers (SUPERSEDED)
- **Network:** 4→64→64→21, softmax output
- **Training:** Sample discrete actions, REINFORCE gradients
- **Inference:** u = Σ pᵢ · Fᵢ (smooth expected-value output)
- **Problem:** Discrete-action REINFORCE still has high variance over long episodes. Superseded by PPO approach before thorough validation.

#### Attempt 6: PPO with Continuous Gaussian Policy (CURRENT — WORKING)
- **Algorithm:** Proximal Policy Optimization (PPO) with clipped surrogate objective
- **Actor:** 4→64→64→1, tanh hidden layers, linear output × maxForce
- **Critic:** 4→64→64→1, tanh hidden layers, linear value output
- **Policy:** π(a|s) = N(μ_θ(s), σ²) with learnable σ via logStd parameter
- **Key insight — output scaling:** μ = raw_network_output × maxForce. Without this, over-normalized inputs (÷[3,5,π,8]) produced activations of ~0.03, making the actor output ~0.03N — three orders of magnitude too small. Output scaling maps the network's natural operating range to physical force magnitudes.
- **Key insight — no over-normalization:** Raw state values [x, ẋ, θ, θ̇] fed directly to the network. Typical magnitudes (0.1–0.5) are appropriate for Xavier-initialized weights with tanh activations.
- **Result:** Avg reward ~158, best ~466, within just 206 episodes. First RL iteration to show clear, sustained learning.

### 3.2 Current RL Configuration

```javascript
// File: ppo.js

// Actor: 4→64→64→1 (Gaussian mean μ, scaled by maxForce)
// Critic: 4→64→64→1 (scalar value V(s))
// Total parameters: ~9,091 (4,545 per network + 1 logStd)

// State normalization: NONE (raw state values)
// Output scaling: μ = raw_output × maxForce (default 20N)

// PPO hyperparameters
lr = 1e-3             // learning rate (Adam-like step via gradient clipping)
gamma = 0.99          // discount factor
gaeLambda = 0.95      // GAE lambda for advantage estimation
clipRange = 0.2       // PPO clipping epsilon
entCoeff = 0.001      // entropy bonus coefficient
vfCoeff = 0.5         // value function loss coefficient
maxGradNorm = 5.0     // global gradient norm clipping
nSteps = 2048         // rollout buffer size (steps, not episodes)
batchSize = 64        // mini-batch size
nEpochs = 4           // PPO epochs per rollout

// Exploration
logStd_init = 2.0     // σ = exp(2.0) ≈ 7.4N — covers useful force range
logStd_bounds = [-3, 2]  // σ range: [0.05N, 7.4N]

// Episode configuration
maxSteps = 500        // episode length cap
maxForce = 20         // action clipping bound (N)

// Initial state distribution
x ~ U(-0.3, 0.3)
ẋ ~ U(-0.25, 0.25)
θ ~ U(-0.175, 0.175)  // ~±10°
θ̇ ~ U(-0.25, 0.25)

// Termination
|θ| > 60° OR |x| > 4.0m OR step >= 500

// Reward
r_t = 1.0 (per step alive)
```

### 3.3 PPO Gaussian Architecture — How It Works

The current approach uses **separate actor-critic networks** with a **continuous Gaussian policy**, trained with PPO's clipped surrogate objective.

#### Actor (Policy Network)

The actor maps state to a force distribution:

$$
\mu = \text{maxForce} \cdot (W_3 \cdot \tanh(W_2 \cdot \tanh(W_1 \cdot s + b_1) + b_2) + b_3)
$$

$$
\pi(a|s) = \mathcal{N}(\mu, \sigma^2), \quad \sigma = e^{\text{logStd}}
$$

**Output scaling** is the critical design choice. The network naturally operates in the [-1, 1] range due to Xavier initialization and tanh activations. Multiplying by `maxForce` maps this to physical force magnitudes. Without it, the actor output is ~0.03N — essentially zero control.

The `logStd` parameter is **state-independent** (shared across all states) and learned via gradient ascent on the PPO objective. It starts at 2.0 (σ ≈ 7.4N) for broad initial exploration and decreases as the policy becomes confident.

#### Critic (Value Network)

The critic predicts expected cumulative reward from a given state:

$$
V(s) = W_3 \cdot \tanh(W_2 \cdot \tanh(W_1 \cdot s + b_1) + b_2) + b_3
$$

No output scaling — the critic operates in reward-space directly.

#### Generalized Advantage Estimation (GAE)

Advantages are computed backward through the rollout buffer:

$$
\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

Advantages are normalized to zero mean and unit variance before the PPO update.

#### PPO Clipped Surrogate Objective

For each sample in the mini-batch:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}
$$

$$
L^\text{CLIP} = \min\left(r_t \hat{A}_t, \; \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t\right)
$$

The gradient flows through `∂log π/∂μ = (a - μ)/σ²` into `gradMu()`, which computes `∂μ/∂θ` via manual backpropagation (including the maxForce scaling factor at every layer).

#### Hand-Rolled Backpropagation

All gradients are computed analytically — no autograd library. The `gradMu()` method chains through:
1. Output layer: `∂μ/∂W3[j] = a2[j] × maxForce`
2. Layer 2 → output: `dA2[j] = W3[j] × maxForce × (1 - a2[j]²)`
3. Layer 1 → layer 2: standard tanh derivative chain rule
4. Input → layer 1: standard chain rule

The maxForce factor propagates through the entire gradient as a constant multiplier, originating from the output scaling `μ = raw × maxForce`.

#### Inference

During inference (non-training), the action is simply:
```
u = clip(μ, -maxForce, maxForce)
```
No sampling noise — the deterministic mean provides smooth control. Uses the best-reward actor snapshot if available.

### 3.4 Exploration via Learnable σ

The Gaussian standard deviation σ = exp(logStd) controls the exploration-exploitation tradeoff:

- **Initial:** logStd = 2.0 → σ ≈ 7.4N. Actions are dominated by noise, exploring the full force range.
- **During training:** PPO gradient on logStd naturally decreases σ as the policy improves. The entropy coefficient (0.001) provides a small bonus to prevent premature collapse.
- **Converged:** logStd ≈ -1 to -2 → σ ≈ 0.1–0.4N. Precise, low-noise control.
- **Clamped to [-3, 2]** to prevent numerical issues (σ too small → log-prob explosion, σ too large → no learning signal).

### 3.5 Key Lessons Learned

1. **Output scaling is critical.** With Xavier initialization and typical input magnitudes (0.1–0.5), the raw network output is O(0.3). For a pendulum requiring 10–20N forces, the output must be scaled by `maxForce`. Without scaling, the network whispers at the pendulum when it needs to shout.

2. **Don't over-normalize inputs.** Dividing by the maximum possible range (e.g., θ/π, x/3) makes inputs O(0.03) for typical operating conditions. This attenuates through each layer, producing vanishingly small outputs and gradients. Raw state values or mild normalization works better.

3. **Initial σ must cover the useful force range.** Starting with σ = 1N when the pendulum needs ±10–20N corrections means the agent barely explores. logStd = 2.0 (σ ≈ 7.4N) provides exploration across the full action range.

4. **Action caching is critical.** The physics runs 8 substeps per RL step. The RL action must be computed ONCE and held constant for all substeps. Otherwise the trajectory data is inconsistent with the actual forces applied.

5. **PPO succeeds where REINFORCE failed.** PPO's clipped surrogate, value baseline (critic), and GAE provide stable, low-variance gradient estimates that REINFORCE (even with discrete softmax) couldn't match on this problem. The actor-critic structure was essential.

6. **Reward simplicity matters.** Survival reward (+1/step) has one clear gradient: "stay upright longer." Elaborate reward shaping creates optimization complexity without proportional benefit.

7. **Network capacity matters.** Two hidden layers (64 units each) are necessary for the nonlinear state→action mapping. Single hidden layer can only create linear decision boundaries in 4D state space.

8. **Hand-rolled backprop must respect output scaling.** The maxForce factor in `μ = raw × maxForce` must propagate through the entire `gradMu()` computation. Missing it anywhere silently breaks the gradient magnitude.

---

## 4. Architecture & UI

### 4.1 Layout

Three-panel layout:
- **Left panel (280px):** Controller selection, system parameters, controller gains, action buttons
- **Center:** Canvas simulation + 3 bottom strip plots (θ, x/reward, force vs time)
- **Right panel (300px):** State vector display, energy bars (KE/PE/Total), active equations, performance metrics

### 4.2 Rendering

Canvas-based at device pixel ratio. Features:
- Grid background, ground plane, track markers
- Cart with rounded body, spinning wheels (wheel angle tracks cart position)
- Pendulum rod with glow effect, bob with radial gradient and highlight
- Force arrow (green = positive, red = negative)
- Optional trail (pendulum tip trajectory, fading dots)
- Setpoint indicator (dashed line upward from pivot)
- Camera follows cart slightly (0.3× parallax)

### 4.3 Input

- Arrow keys: apply ±15N manual force (signs inverted to match visual direction)
- Mouse drag on cart: variable force (also inverted)
- Space: pause/resume
- R: reset
- P: random perturbation (±5 rad/s angular, ±1.5 m/s linear)

### 4.4 Energy Display

```
KE = ½(M·ẋ² + m·(ẋ² + L²·θ̇² + 2·ẋ·L·θ̇·cosθ))
PE = mgL(1 + cosθ)    // height above lowest point
TE = KE + PE
```

**Note:** PE uses (1 + cosθ), not (1 − cosθ). PE is maximum when upright (θ=0, cosθ=1) and minimum when hanging (θ=π, cosθ=−1). This was a bug fix — the original had the sign flipped.

---

## 5. Known Issues & Next Steps

### 5.1 RL Controller

The PPO Gaussian implementation (Attempt 6) is working — avg reward ~158, best ~466 at episode 206. Potential improvements:

1. **Training to convergence.** Let training run until avg reward stabilizes near 400–500. Monitor σ (logStd) — it should decrease from ~2.0 to negative values as the policy sharpens.

2. **Network serialization.** Export/import trained weights to localStorage or JSON so training persists across browser sessions. Currently all progress is lost on page reload.

3. **Learning rate scheduling.** A decaying learning rate (e.g., linear decay over training) could improve final performance by reducing oscillation near convergence.

4. **Behavioral cloning warm-start.** Collect LQR trajectories and pre-train the actor via supervised learning before RL fine-tuning. Would dramatically accelerate early training.

5. **Hyperparameter sensitivity.** The current config works but hasn't been tuned. Candidates: nEpochs (try 10, matching SB3 default), lr (try 3e-4), nSteps (try 4096 for lower variance).

### 5.2 Other Potential Improvements

- **MPC controller** could use a proper QP solver instead of gradient descent for more accurate optimization
- **PD controller** position gains (Kx=8, Kxd=6) were hand-tuned and may not be optimal for all parameter combinations
- **Mobile responsiveness** — the 3-panel layout doesn't adapt to narrow screens
- **Export/import trained networks** — serializing the weights to localStorage or a JSON file so training persists across sessions
- **Behavioral cloning** — collecting LQR trajectories and training the NN via supervised learning as a warm-start before RL fine-tuning. This would dramatically accelerate RL convergence.

### 5.3 Deployment

Static file set with no build step:
- No external dependencies (all JS inline or in separate .js files)
- Fonts loaded from Google Fonts CDN (cosmetic only)
- Vercel deployment: drop files in folder, `vercel deploy`

---

## 6. File Structure Reference

```
index.html       Entry point — HTML/CSS layout, UI controls, simulation loop, rendering
physics.js       Nonlinear equations of motion, RK4 integrator, wrapAngle()
pd.js            PD controller
lqr.js           LQR controller (CARE solver via ODE integration)
swingup.js       Swing-Up + LQR (Åström-Furuta energy pumping)
mpc.js           Simplified nonlinear MPC (gradient-descent optimizer)
ppo.js           RL controller — ActorNet, CriticNet, PPO trainer (~500 lines)
```

---

## 7. Summary of All Bugs Fixed

| Bug | Symptom | Root Cause | Fix |
|-----|---------|------------|-----|
| PD sign | Cart accelerates away from tilt | Negative feedback in non-minimum phase system | Flipped to positive feedback |
| PD position | Cart drifts to infinity | Position feedback sign also wrong | Flipped position terms to positive |
| LQR solver | Garbage gains | DARE applied to continuous-time matrices | Replaced with CARE (Riccati ODE integration) |
| Swing-up energy | Energy pumping in wrong direction | PE sign error: used −mgLcosθ instead of +mgLcosθ | Fixed PE formula |
| PE display | Energy bar maximal when hanging | Same sign error in UI | Fixed to mgL(1+cosθ) |
| User input direction | Arrows/mouse push opposite to visual | Raw force direction vs closed-loop response direction | Inverted input signs |
| RL substep noise | Network can't learn; trajectory data inconsistent | Fresh random action every substep, only last recorded | Cache action once per RL step |
| Curriculum threshold | Never advances past 2.9° | Threshold 900 unreachable during noisy training | Lowered to achievable value |
| Entropy bonus | Rewarded popular actions (anti-exploration) | Formula log(N)+log(p) instead of −log(p) | Fixed to −log(p_a) |
| RL output magnitude | Flat reward ~40, no learning for thousands of episodes | Over-normalized inputs (÷[3,5,π,8]) → activations ~0.03 → μ ≈ 0.03N (three orders of magnitude too small) | (1) Raw state values (no normalization), (2) output scaling μ = raw × maxForce, (3) initial logStd = 2.0 for σ ≈ 7.4N exploration |
