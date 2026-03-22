// ============================================================
//  INVERTED PENDULUM — PHYSICS ENGINE
// ============================================================

const g = 9.81;
const dt = 0.002;        // physics timestep (2ms = 500Hz)
const stepsPerFrame = 8;  // 8 steps per render frame

// System parameters (user-tunable via UI)
let params = { M: 1.0, m: 0.3, L: 0.5, d: 0.1 };

// Controller gains (user-tunable via UI)
let gains = {
  pd: { Kp: 150, Kd: 30 },
  lqr: { autoCompute: true },
  swingup: { Ke: 5, catchAngle: 55 },
  mpc: { horizon: 15, Q_theta: 100, R: 0.01 }
};

// ============================================================
//  NONLINEAR DYNAMICS
// ============================================================
function dynamics(s, u) {
  const { M, m, L, d } = params;
  const { x, xdot, theta, thetadot } = s;

  const sinT = Math.sin(theta);
  const cosT = Math.cos(theta);

  // Nonlinear equations of motion
  const D = M + m - m * cosT * cosT;

  const xddot = (u - d * xdot + m * L * thetadot * thetadot * sinT
                 - m * g * sinT * cosT) / D;

  const thetaddot = (-u * cosT + d * xdot * cosT
                     - m * L * thetadot * thetadot * sinT * cosT
                     + (M + m) * g * sinT) / (L * D);

  return { xdot, xddot, thetadot, thetaddot };
}

// RK4 integrator
function rk4Step(s, u, h) {
  function derivs(st) {
    const d = dynamics(st, u);
    return { x: d.xdot, xdot: d.xddot, theta: d.thetadot, thetadot: d.thetaddot };
  }

  const k1 = derivs(s);
  const s2 = {
    x: s.x + 0.5*h*k1.x, xdot: s.xdot + 0.5*h*k1.xdot,
    theta: s.theta + 0.5*h*k1.theta, thetadot: s.thetadot + 0.5*h*k1.thetadot
  };
  const k2 = derivs(s2);
  const s3 = {
    x: s.x + 0.5*h*k2.x, xdot: s.xdot + 0.5*h*k2.xdot,
    theta: s.theta + 0.5*h*k2.theta, thetadot: s.thetadot + 0.5*h*k2.thetadot
  };
  const k3 = derivs(s3);
  const s4 = {
    x: s.x + h*k3.x, xdot: s.xdot + h*k3.xdot,
    theta: s.theta + h*k3.theta, thetadot: s.thetadot + h*k3.thetadot
  };
  const k4 = derivs(s4);

  return {
    x: s.x + (h/6)*(k1.x + 2*k2.x + 2*k3.x + k4.x),
    xdot: s.xdot + (h/6)*(k1.xdot + 2*k2.xdot + 2*k3.xdot + k4.xdot),
    theta: s.theta + (h/6)*(k1.theta + 2*k2.theta + 2*k3.theta + k4.theta),
    thetadot: s.thetadot + (h/6)*(k1.thetadot + 2*k2.thetadot + 2*k3.thetadot + k4.thetadot)
  };
}

// Normalize angle to [-π, π]
function wrapAngle(a) {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}
