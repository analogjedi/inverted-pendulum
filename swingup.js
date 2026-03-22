// ============================================================
//  SWING-UP CONTROLLER — Energy-based + LQR catch
//  Depends on: physics.js (params, gains, g, wrapAngle)
//              lqr.js (controllerLQR)
// ============================================================

function controllerSwingUp(s) {
  const theta = wrapAngle(s.theta);
  const { m, L } = params;

  // If near upright, switch to LQR
  if (Math.abs(theta) < (gains.swingup.catchAngle * Math.PI / 180)) {
    return controllerLQR(s);
  }

  // Energy-based swing-up (Åström & Furuta)
  // Total energy: E = ½mL²θ̇² + mgL·cos(θ)   [PE = height of mass above pivot]
  // At upright (θ=0, static): E_up = mgL
  // At hanging (θ=π, static): E_hang = -mgL
  // We pump energy until E → E_up
  const E_current = 0.5 * m * L * L * s.thetadot * s.thetadot + m * g * L * Math.cos(s.theta);
  const E_upright = m * g * L;
  const E_delta = E_current - E_upright;  // negative when below target → need to add energy

  // Control law: u = Ke · ΔE · sign(θ̇·cosθ)
  // This pumps energy when ΔE<0 and removes energy when ΔE>0
  const u = gains.swingup.Ke * E_delta * Math.sign(s.thetadot * Math.cos(s.theta));

  // Cart position regulation to keep it from running away
  return u - 2.0 * s.x - 1.5 * s.xdot;
}
