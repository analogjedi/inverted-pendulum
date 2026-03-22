# Inverted Pendulum Control Lab — Project Journal

## Project Overview

A single-file interactive web simulation (~2,040 lines HTML/JS/CSS) implementing a nonlinear inverted pendulum on a cart with five classical controllers and a neural network reinforcement learning controller. The simulation runs entirely client-side with no dependencies — pure vanilla JS, canvas rendering, and hand-rolled linear algebra.

**File:** `inverted_pendulum.html`  
**Deployment:** Static file, Vercel-ready (rename to `index.html`)  
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

The RL controller went through five major iterations. Understanding this history is critical for the next agent.

#### Attempt 1: Gaussian Continuous (FAILED)
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
- **Core issue:** Gaussian policy gradient ∇log N(μ,σ²) = (a−μ)/σ² is a single noisy scalar. Insufficient signal for convergence at scale.
- **Result:** Avg ~137, flat learning curve at 9,000+ episodes.

#### Attempt 4: Discrete Softmax, 1 Hidden Layer (FAILED)
- **Network:** 4→32→11, softmax output over 11 force levels
- Clean gradient: ∇log π = one_hot(a) − probs (updates all logits simultaneously)
- **Problem:** Single hidden layer lacks capacity for nonlinear state→action mapping. Can only create linear decision boundaries in 4D state space.
- **Result:** No meaningful learning.

#### Attempt 5: Soft-DAC, 2 Hidden Layers (CURRENT — UNTESTED)
- **Network:** 4→64→64→21, softmax output
- **Training:** Sample discrete actions (clean gradients)
- **Inference:** u = Σ pᵢ · Fᵢ (smooth expected-value output)
- **Reward:** +1 per step alive (simplest possible signal)
- **Entropy bonus:** −β·log(p_a) added to advantage weight
- **Status:** Deployed but not yet validated. This is where the next agent picks up.

### 3.2 Current RL Configuration

```javascript
// Action space
ACTIONS = [-50, -45, -40, ..., 0, ..., 40, 45, 50]  // 21 levels, 5N spacing
NUM_ACTIONS = 21

// Network: 4→64→64→21
// Layer 1: W1[64×4], b1[64], tanh activation
// Layer 2: W2[64×64], b2[64], tanh activation
// Layer 3: W3[21×64], b3[21], softmax activation
// Total parameters: ~5,700

// Training hyperparameters
maxSteps = 1000       // episode length cap
batchSize = 24        // episodes per gradient update
lr = 0.003            // learning rate
gamma = 0.99          // discount factor
entropyCoeff = 0.02   // entropy bonus weight

// Initial state distribution
x ~ U(-0.3, 0.3)
ẋ ~ U(-0.25, 0.25)
θ ~ U(-0.175, 0.175)  // ~±10°
θ̇ ~ U(-0.25, 0.25)

// Termination
|θ| > 60° OR |x| > 4.0m OR step >= 1000

// Reward
r_t = 1.0 (per step alive)
```

### 3.3 The Soft-DAC Concept

The key innovation in the current approach: **train discrete, deploy continuous.**

During **training**, the network samples from the softmax distribution and receives discrete gradients:
```
∇log π(a|s) = one_hot(a) − probs
```
This gradient updates ALL 21 logits simultaneously — taking action i and getting positive reward tells the network "action i was good AND actions 0..i-1, i+1..20 were relatively worse."

During **inference**, the output force is the **expected value** over the probability distribution:
```
u = Σᵢ pᵢ · Fᵢ
```
This is analogous to a weighted resistor DAC: the softmax probabilities are tap weights, the discrete force levels are reference voltages. As the policy becomes confident, probabilities sharpen and the output converges to a specific force — but transitions are always smooth.

With 21 levels spanning ±50N in 5N steps, the effective force resolution is continuous because probability-weighted interpolation between adjacent levels produces arbitrary intermediate values.

### 3.4 Entropy Bonus

The entropy bonus prevents premature policy collapse:

```
w = advantage + β · (−log p_a)
```

When p_a is small (unlikely action), −log(p_a) is large, giving extra gradient weight to explore that action. When β = 0, pure REINFORCE. Setting β too high prevents the policy from ever sharpening (stays uniform). Recommended starting range: 0.01–0.05.

### 3.5 Key Lessons Learned

1. **Action caching is critical.** The physics runs 8 substeps per RL step. The RL action must be computed ONCE and held constant for all substeps. Otherwise the trajectory data is inconsistent with the actual forces applied.

2. **Gaussian continuous policies are a poor match for REINFORCE on this problem.** The gradient ∇log N(μ,σ²) = (a−μ)/σ² is a single scalar — too noisy for credit assignment over 500+ step episodes. Discrete softmax gives a gradient vector that updates all action preferences simultaneously.

3. **Reward simplicity matters.** Elaborate reward shaping (cosine, quadratic penalties, effort terms) creates a complex optimization landscape with many local optima. Survival reward (+1/step) has one clear gradient: "stay upright longer."

4. **Network capacity matters.** Single hidden layer cannot represent the nonlinear regions needed for pendulum control (small corrections near vertical, aggressive corrections at large angles). Two hidden layers are necessary.

5. **Curriculum learning is fragile.** Threshold tuning, σ interactions, and catastrophic forgetting when difficulty increases all add failure modes. Better to start with a reasonable initial state distribution and let the survival reward provide natural curriculum.

6. **The sign convention is everything.** The inverted pendulum is non-minimum phase. Every feedback term "chases" the error rather than opposing it. Getting this wrong produces runaway divergence that looks like a gain problem but is actually a sign problem.

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

### 5.1 RL Controller (Priority)

The Soft-DAC REINFORCE implementation (Attempt 5) has not been validated. The next agent should:

1. **Run training at speed 10–30, LR 0.003, entropy 0 initially.** Monitor the "Steps survived / episode" plot. Success = average approaching 500+ within 1,000–2,000 episodes.

2. **If flat/no learning after 500 episodes:** The initial state distribution may be too wide. Try narrowing θ range from ±0.175 to ±0.05 rad in `randomInitState()`. Or try increasing batch size to 32–48 for lower variance gradients.

3. **If learning plateaus at ~100–200 steps:** The network may be collapsing to a single action. Increase entropy coefficient to 0.02–0.05. Check the softmax distribution — if one action dominates at >90% probability in all states, entropy bonus is needed.

4. **If learning succeeds but inference is jittery:** The soft-DAC expected value should be smooth, but if the policy is very peaked (one action at 99%), it's effectively discrete. Consider temperature scaling the softmax during inference: `probs = softmax(logits / T)` with T > 1 to spread the distribution.

5. **Advanced: transition to PPO.** REINFORCE has high variance. If the basic approach works but convergence is slow, implement PPO (clipped surrogate objective) for more sample-efficient training. The network architecture and action space can stay the same.

### 5.2 Other Potential Improvements

- **MPC controller** could use a proper QP solver instead of gradient descent for more accurate optimization
- **PD controller** position gains (Kx=8, Kxd=6) were hand-tuned and may not be optimal for all parameter combinations
- **Mobile responsiveness** — the 3-panel layout doesn't adapt to narrow screens
- **Export/import trained networks** — serializing the weights to localStorage or a JSON file so training persists across sessions
- **Behavioral cloning** — collecting LQR trajectories and training the NN via supervised learning as a warm-start before RL fine-tuning. This would dramatically accelerate RL convergence.

### 5.3 Deployment

The file is a single static HTML page with:
- No external dependencies (all JS inline)
- Fonts loaded from Google Fonts CDN (cosmetic only)
- No build step required
- Vercel deployment: rename to `index.html`, drop in folder, `vercel deploy`

---

## 6. File Structure Reference

```
Lines 1–400:     HTML structure + CSS styling
Lines 400–475:   Controller buttons, parameter sliders, action buttons
Lines 475–555:   Right panel (state display, energy, equations, performance)
Lines 555–610:   JS globals, state initialization, parameters, gains
Lines 610–660:   Nonlinear dynamics function + RK4 integrator
Lines 660–860:   Classical controllers (PD, LQR with CARE, Swing-Up, MPC)
Lines 860–1170:  Neural network + REINFORCE trainer (PolicyNet class, rl object)
Lines 1170–1180: computeControl() dispatcher
Lines 1180–1210: Canvas setup and resize handling
Lines 1210–1320: Main simulation renderer (drawSim)
Lines 1320–1380: Strip plot renderer (drawPlot)
Lines 1380–1470: State display + energy computation
Lines 1470–1550: Controller parameter UI builder (updateCtrlParams)
Lines 1550–1580: Equation display updater
Lines 1580–1620: Reset, events, keyboard, mouse handlers
Lines 1620–1750: Main loop (with RL training integration)
Lines 1750–1850: Reward plot renderer
Lines 1850–1870: Initialization
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
