# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Single-file interactive web simulation of an inverted pendulum on a cart, with five classical controllers and a neural network RL controller. Runs entirely client-side with zero dependencies — vanilla JS, HTML5 Canvas, hand-rolled linear algebra.

**Main file:** `inverted_pendulum_RL.html` (~2,038 lines of HTML/CSS/JS)

## Running

Open `inverted_pendulum_RL.html` in any modern browser. No build step, no server, no install.

## Architecture (within the single file)

The file is structured in this order:

1. **CSS + HTML** (lines 1–552) — Dark-themed 3-panel layout: left sidebar (controller selection/params), center (canvas + plots), right sidebar (state display, energy bars, equations, performance metrics).

2. **Physics engine** (lines ~553–660) — Full nonlinear coupled equations of motion, solved with RK4 at 500 Hz (dt=0.002s, 8 substeps per render frame). State vector: `[x, ẋ, θ, θ̇]` where θ=0 is upright.

3. **Classical controllers** (lines ~660–860):
   - **PD** — Proportional-derivative on angle and position
   - **LQR** — Solves CARE via ODE integration (hand-rolled, no library)
   - **Swing-Up + LQR** — Åström-Furuta energy pumping with catch transition to LQR
   - **MPC** — Simplified nonlinear MPC via gradient descent with finite-difference gradients

4. **RL controller** (lines ~860–1,214):
   - PolicyNet: 4→64→64→21 MLP with tanh activations and softmax output
   - Training: REINFORCE with entropy bonus, 24-episode batches
   - **Soft-DAC inference**: trains on discrete actions (21 force levels, −50N to +50N), deploys as weighted average `Σ pᵢ·Fᵢ` for smooth continuous control

5. **Simulation loop** (`loop()`, lines ~1,620–1,750) — Each frame: compute control → 8× RK4 substeps → RL trajectory caching → canvas render + plot update → episode termination check → batch training trigger.

6. **Rendering & UI** (lines ~1,210–2,038) — Canvas drawing (cart, pendulum, trail, force arrow, grid), 3 time-series plots (θ, x/reward, force), energy bars, equation display.

## Critical Physics Detail

The system is **non-minimum phase**: positive force → negative angular acceleration near upright (`θ̈ ∝ −u·cos(θ)/LD`). All controller feedback terms share the same sign as the error they correct — you "chase the fall." This sign convention has been a source of subtle bugs (documented in the project journal). Double-check signs when modifying any controller.

## Key Global State

- `state` — `{x, xdot, theta, thetadot}` current system state
- `params` — `{M, m, L, d}` user-tunable physical parameters
- `activeCtrl` — string selecting active controller
- `rl` — RL trainer object (network weights, training buffers, hyperparameters)
- `history` — 600-sample ring buffers for the time-series plots

## Reference Documents

- `inverted_pendulum_project_journal.md` — Detailed history of all 5 RL iterations, every bug fix, and lessons learned. Read this before modifying controllers or RL.
- `rl-architecture-plan.txt` — Reference notes on PPO + MLP for CartPole (not all implemented).
