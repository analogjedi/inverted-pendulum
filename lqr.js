// ============================================================
//  LQR CONTROLLER — Linear Quadratic Regulator
//  Depends on: physics.js (g, params, gains, wrapAngle)
// ============================================================

// LQR gain vector (computed by CARE solver)
let K_lqr = [0, 0, 0, 0];

// Compute LQR gains via CARE (Continuous Algebraic Riccati Equation)
// Solved by integrating the Riccati ODE: Ṗ = A'P + PA − PBR⁻¹B'P + Q
function computeLQR() {
  const { M, m, L, d } = params;
  const totM = M + m;
  const ML = M * L;

  // Linearized A, B around θ=0 (upright)
  // State: [x, ẋ, θ, θ̇]
  // From dynamics: θ̈ = [(M+m)g/(ML)]θ + [d/(ML)]ẋ − [1/(ML)]u
  //                ẍ  = [−d/M]ẋ + [−mg/M]θ + [1/M]u
  const Ac = [
    [0, 1, 0, 0],
    [0, -d/M, -m*g/M, 0],
    [0, 0, 0, 1],
    [0, d/ML, totM*g/ML, 0]
  ];
  const Bc = [0, 1/M, 0, -1/ML];

  // Q and R weights for LQR cost J = ∫(x'Qx + u'Ru)dt
  const Q = [
    [10, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 100, 0],
    [0, 0, 0, 10]
  ];
  const R_val = 0.01;
  const Rinv = 1 / R_val;

  // Integrate Riccati ODE to steady state
  let P = Array.from({length:4}, () => new Array(4).fill(0));
  // Seed with Q for faster convergence
  for (let i = 0; i < 4; i++) P[i][i] = Q[i][i];

  const h = 0.0005;  // integration step (small for stability)
  for (let iter = 0; iter < 40000; iter++) {
    // PB (4-vector)
    const PB = [0,0,0,0];
    for (let i = 0; i < 4; i++)
      for (let k = 0; k < 4; k++)
        PB[i] += P[i][k] * Bc[k];

    // Compute dP/dt = A'P + PA - (1/R)*PB*PB' + Q
    let maxDp = 0;
    const dP = Array.from({length:4}, () => new Array(4).fill(0));
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        let AtP_ij = 0, PA_ij = 0;
        for (let k = 0; k < 4; k++) {
          AtP_ij += Ac[k][i] * P[k][j];  // (A')P = Aᵀ P
          PA_ij  += P[i][k] * Ac[k][j];  // P A
        }
        const dp = AtP_ij + PA_ij - Rinv * PB[i] * PB[j] + Q[i][j];
        dP[i][j] = dp;
        maxDp = Math.max(maxDp, Math.abs(dp));
      }
    }

    // Forward Euler update
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++)
        P[i][j] += h * dP[i][j];

    // Converged?
    if (maxDp < 1e-6) break;
  }

  // K = R⁻¹ B' P
  K_lqr = [0,0,0,0];
  for (let j = 0; j < 4; j++)
    for (let k = 0; k < 4; k++)
      K_lqr[j] += Rinv * Bc[k] * P[k][j];

  // Verify: K[2] should be negative (positive u corrects positive θ via u = -Kx)
  // If CARE didn't converge properly, fall back to known-good gains
  if (K_lqr[2] > 0 || isNaN(K_lqr[0])) {
    console.warn('CARE did not converge, using fallback gains');
    K_lqr = [-10, -12, -120, -30]; // hand-tuned stabilizing gains
  }
}

function controllerLQR(s) {
  const sv = [s.x, s.xdot, wrapAngle(s.theta), s.thetadot];
  let u = 0;
  for (let i = 0; i < 4; i++) u -= K_lqr[i] * sv[i];
  return u;
}
