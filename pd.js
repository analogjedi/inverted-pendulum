// ============================================================
//  PD CONTROLLER — Proportional-Derivative
//  Depends on: physics.js (gains, wrapAngle)
// ============================================================

// Key insight: In this system, θ̈ has a −u·cosθ/(L·D) term.
// Positive u creates NEGATIVE θ̈, so when θ > 0 (tilting right),
// we need POSITIVE force (chase the fall) to correct it.
// This is like balancing a broomstick — move your hand TOWARD the tilt.
function controllerPD(s) {
  const theta = wrapAngle(s.theta);
  // Non-minimum phase system: ALL feedback "chases" the error.
  // Cart right → push right → pendulum tilts left → gravity pulls back left.
  // Same counter-intuitive logic for position as for angle.
  return gains.pd.Kp * theta + gains.pd.Kd * s.thetadot
         + 8.0 * s.x + 6.0 * s.xdot;
}
