# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Interactive web simulation of an inverted pendulum on a cart, with five classical controllers and a neural network RL controller. Runs entirely client-side with zero dependencies — vanilla JS, HTML5 Canvas, hand-rolled linear algebra.

## Running

Open `index.html` in any modern browser. No build step, no server, no install.

## File Structure

| File | Purpose |
|------|---------|
| `index.html` | HTML/CSS layout, UI controls, simulation loop, rendering (~1,300 lines) |
| `physics.js` | Nonlinear equations of motion, RK4 integrator, `wrapAngle()` |
| `pd.js` | PD controller |
| `lqr.js` | LQR controller (solves CARE via ODE integration, hand-rolled) |
| `swingup.js` | Swing-Up + LQR (Åström-Furuta energy pumping with catch) |
| `mpc.js` | Simplified nonlinear MPC via gradient descent |
| `ppo.js` | RL controller — PPO with continuous Gaussian policy (~500 lines) |

## Architecture

1. **Physics engine** (`physics.js`) — Full nonlinear coupled equations of motion, solved with RK4 at 500 Hz (dt=0.002s, 8 substeps per render frame). State vector: `[x, ẋ, θ, θ̇]` where θ=0 is upright.

2. **Classical controllers** (`pd.js`, `lqr.js`, `swingup.js`, `mpc.js`) — PD, LQR, Swing-Up+LQR, and MPC.

3. **RL controller** (`ppo.js`) — Actor-critic PPO with continuous Gaussian policy:
   - **Actor:** 4→64→64→1 MLP, tanh hidden layers, linear output scaled by `maxForce`
   - **Critic:** 4→64→64→1 MLP, tanh hidden layers, linear value output
   - **Policy:** π(a|s) = N(μ_θ(s), σ²) where μ = raw_network_output × maxForce
   - **Training:** PPO clipped surrogate objective, GAE advantages (λ=0.95), 2048-step rollout buffer, 64-sample mini-batches, 4 epochs per update
   - **Inference:** deterministic μ (no sampling noise), clipped to ±maxForce
   - **Key design:** output scaling by `maxForce` maps network's natural [-1,1] operating range to physical force magnitudes; raw state values (no normalization) preserve meaningful input magnitudes

4. **Simulation loop** (`index.html`) — Each frame: compute control → 8× RK4 substeps → RL trajectory caching → canvas render + plot update → episode termination check → PPO training trigger when rollout buffer fills.

5. **Rendering & UI** (`index.html`) — Dark-themed 3-panel layout, canvas drawing (cart, pendulum, trail, force arrow, grid), 3 time-series plots, energy bars, equation display.

## Critical Physics Detail

The system is **non-minimum phase**: positive force → negative angular acceleration near upright (`θ̈ ∝ −u·cos(θ)/LD`). All controller feedback terms share the same sign as the error they correct — you "chase the fall." This sign convention has been a source of subtle bugs (documented in the project journal). Double-check signs when modifying any controller.

## Key Global State

- `state` — `{x, xdot, theta, thetadot}` current system state
- `params` — `{M, m, L, d}` user-tunable physical parameters
- `activeCtrl` — string selecting active controller
- `rl` — RL trainer object (network weights, training buffers, hyperparameters) — defined in `ppo.js`
- `maxForce` — action clipping bound (default 20N) — defined in `ppo.js`
- `history` — 600-sample ring buffers for the time-series plots

## Reference Documents

- `inverted_pendulum_project_journal.md` — Detailed history of all 6 RL iterations, every bug fix, and lessons learned. Read this before modifying controllers or RL.
- `rl-architecture-plan.txt` — Reference notes on PPO + MLP for CartPole (background research).
