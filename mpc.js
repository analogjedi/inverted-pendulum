// ============================================================
//  MPC CONTROLLER — Model Predictive Control
//  Depends on: physics.js (params, gains, dt, stepsPerFrame, wrapAngle, rk4Step)
//              lqr.js (controllerLQR)
//              swingup.js (controllerSwingUp)
// ============================================================

function controllerMPC(s) {
  // Simplified MPC: forward-simulate nonlinear model over horizon,
  // optimize control sequence via gradient descent
  const { M, m, L, d: damp } = params;
  const N = gains.mpc.horizon;
  const Qw = gains.mpc.Q_theta;
  const Rw = gains.mpc.R;

  const theta = wrapAngle(s.theta);

  // If far from upright, use swing-up to get close first
  if (Math.abs(theta) > Math.PI * 0.6) {
    return controllerSwingUp(s);
  }

  // Single-shooting with gradient descent, 3 optimization passes
  const dtMPC = dt * stepsPerFrame;
  let uSeq = new Float32Array(N);

  // Warm-start with LQR
  const uLQR = controllerLQR(s);
  for (let k = 0; k < N; k++) uSeq[k] = uLQR * Math.pow(0.95, k);

  for (let opt = 0; opt < 5; opt++) {
    let grad = new Float32Array(N);

    // Forward simulate trajectory
    let traj = [{ ...s, theta }];
    for (let k = 0; k < N; k++) {
      traj.push(rk4Step(traj[k], uSeq[k], dtMPC));
    }

    // Compute gradients via finite differences
    const eps = 0.5;
    for (let k = 0; k < N; k++) {
      // Simulate with u+eps and u-eps to get cost gradient
      let trajPlus = [{ ...traj[k] }];
      let trajMinus = [{ ...traj[k] }];
      trajPlus.push(rk4Step(trajPlus[0], uSeq[k] + eps, dtMPC));
      trajMinus.push(rk4Step(trajMinus[0], uSeq[k] - eps, dtMPC));

      const thetaP = wrapAngle(trajPlus[1].theta);
      const thetaM = wrapAngle(trajMinus[1].theta);
      const costP = Qw * thetaP * thetaP + 10 * trajPlus[1].x * trajPlus[1].x + Rw * (uSeq[k]+eps)**2;
      const costM = Qw * thetaM * thetaM + 10 * trajMinus[1].x * trajMinus[1].x + Rw * (uSeq[k]-eps)**2;

      grad[k] = (costP - costM) / (2 * eps);
    }

    // Gradient descent update
    const lr = 2.0;
    for (let k = 0; k < N; k++) {
      uSeq[k] -= lr * grad[k];
      uSeq[k] = Math.max(-60, Math.min(60, uSeq[k]));
    }
  }

  return uSeq[0];
}
