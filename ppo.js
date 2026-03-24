// ============================================================
//  NEURAL NETWORK & REINFORCEMENT LEARNING (PPO)
//  Continuous Gaussian policy: outputs mean force μ
//  Actor-critic with clipped surrogate objective
//  Depends on: physics.js (wrapAngle)
// ============================================================

let maxForce = 40;  // N, action clipping bound

function normalizeState(s, prevForce) {
  // 6 inputs: [x, ẋ, sin(θ), cos(θ), θ̇, prevForce/maxForce]
  // sin/cos eliminates angle discontinuity at ±π
  // prevForce lets network smooth its own control output
  const theta = wrapAngle(s.theta);
  return [s.x, s.xdot, Math.sin(theta), Math.cos(theta), s.thetadot,
          (prevForce || 0) / maxForce];
}

function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const LOG2PI = Math.log(2 * Math.PI);

// --- Actor network: 6 → 64 → 64 → 1 (Gaussian mean μ) ---
class ActorNet {
  constructor() {
    const I = 6, H1 = 64, H2 = 64;
    this.I = I; this.H1 = H1; this.H2 = H2;

    // Xavier/Glorot initialization (for tanh)
    this.W1 = new Float64Array(H1 * I);
    this.b1 = new Float64Array(H1);
    this.W2 = new Float64Array(H2 * H1);
    this.b2 = new Float64Array(H2);
    this.W3 = new Float64Array(H2);  // 1 output × H2
    this.b3 = new Float64Array(1);

    const s1 = Math.sqrt(1.0 / I);
    for (let i = 0; i < this.W1.length; i++) this.W1[i] = randn() * s1;
    const s2 = Math.sqrt(1.0 / H1);
    for (let i = 0; i < this.W2.length; i++) this.W2[i] = randn() * s2;
    const s3 = Math.sqrt(1.0 / H2);
    for (let i = 0; i < this.W3.length; i++) this.W3[i] = randn() * s3;

    // Learnable log standard deviation (state-independent)
    this.logStd = 2.0;  // σ ≈ 7.4N — covers useful force range for exploration

    // Cached for backprop
    this.z1 = new Float64Array(H1);
    this.a1 = new Float64Array(H1);
    this.z2 = new Float64Array(H2);
    this.a2 = new Float64Array(H2);
    this.mu = 0;
    this.inp = null;
  }

  // Forward pass → scalar mean μ (linear output, no activation)
  forward(input) {
    this.inp = input;
    const { I, H1, H2 } = this;
    for (let i = 0; i < H1; i++) {
      let s = this.b1[i];
      for (let j = 0; j < I; j++) s += this.W1[i * I + j] * input[j];
      this.z1[i] = s;
      this.a1[i] = Math.tanh(s);
    }
    for (let i = 0; i < H2; i++) {
      let s = this.b2[i];
      for (let j = 0; j < H1; j++) s += this.W2[i * H1 + j] * this.a1[j];
      this.z2[i] = s;
      this.a2[i] = Math.tanh(s);
    }
    let v = this.b3[0];
    for (let j = 0; j < H2; j++) v += this.W3[j] * this.a2[j];
    this.mu = v * maxForce;  // scale network output to force range
    return this.mu;
  }

  // Sample action from N(μ, σ²)
  sample() {
    const sigma = Math.exp(this.logStd);
    return this.mu + sigma * randn();
  }

  // Log probability of action under N(μ, σ²)
  logProb(action) {
    const sigma = Math.exp(this.logStd);
    const diff = action - this.mu;
    return -0.5 * (diff * diff) / (sigma * sigma) - this.logStd - 0.5 * LOG2PI;
  }

  // Gaussian entropy: 0.5 * log(2πeσ²)
  entropy() {
    return 0.5 * (1.0 + LOG2PI) + this.logStd;
  }

  // Gradient of μ w.r.t. all network parameters
  // μ = maxForce * (W3·a2 + b3), so all gradients include maxForce factor
  gradMu() {
    const { I, H1, H2 } = this;
    const scale = maxForce;
    const gW3 = new Float64Array(H2);
    const gb3 = new Float64Array(1);
    for (let j = 0; j < H2; j++) gW3[j] = this.a2[j] * scale;
    gb3[0] = scale;

    const dA2 = new Float64Array(H2);
    for (let j = 0; j < H2; j++) dA2[j] = this.W3[j] * scale * (1 - this.a2[j] * this.a2[j]);

    const gW2 = new Float64Array(H2 * H1);
    const gb2 = new Float64Array(H2);
    for (let i = 0; i < H2; i++) {
      gb2[i] = dA2[i];
      for (let j = 0; j < H1; j++) gW2[i * H1 + j] = dA2[i] * this.a1[j];
    }

    const dA1 = new Float64Array(H1);
    for (let j = 0; j < H1; j++) {
      let s = 0;
      for (let i = 0; i < H2; i++) s += this.W2[i * H1 + j] * dA2[i];
      dA1[j] = s * (1 - this.a1[j] * this.a1[j]);
    }

    const gW1 = new Float64Array(H1 * I);
    const gb1 = new Float64Array(H1);
    for (let i = 0; i < H1; i++) {
      gb1[i] = dA1[i];
      for (let j = 0; j < I; j++) gW1[i * I + j] = dA1[i] * this.inp[j];
    }

    return { gW1, gb1, gW2, gb2, gW3, gb3 };
  }

  clone() {
    const c = new ActorNet();
    c.W1 = new Float64Array(this.W1);
    c.b1 = new Float64Array(this.b1);
    c.W2 = new Float64Array(this.W2);
    c.b2 = new Float64Array(this.b2);
    c.W3 = new Float64Array(this.W3);
    c.b3 = new Float64Array(this.b3);
    c.logStd = this.logStd;
    return c;
  }

  paramCount() {
    return this.W1.length + this.b1.length +
           this.W2.length + this.b2.length +
           this.W3.length + this.b3.length + 1; // +1 for logStd
  }
}

// --- Critic network: 6 → 64 → 64 → 1 (linear output) ---
class CriticNet {
  constructor() {
    const I = 6, H1 = 64, H2 = 64;
    this.I = I; this.H1 = H1; this.H2 = H2;

    this.W1 = new Float64Array(H1 * I);
    this.b1 = new Float64Array(H1);
    this.W2 = new Float64Array(H2 * H1);
    this.b2 = new Float64Array(H2);
    this.W3 = new Float64Array(H2);
    this.b3 = new Float64Array(1);

    const s1 = Math.sqrt(1.0 / I);
    for (let i = 0; i < this.W1.length; i++) this.W1[i] = randn() * s1;
    const s2 = Math.sqrt(1.0 / H1);
    for (let i = 0; i < this.W2.length; i++) this.W2[i] = randn() * s2;
    const s3 = Math.sqrt(1.0 / H2);
    for (let i = 0; i < this.W3.length; i++) this.W3[i] = randn() * s3;

    this.z1 = new Float64Array(H1);
    this.a1 = new Float64Array(H1);
    this.z2 = new Float64Array(H2);
    this.a2 = new Float64Array(H2);
    this.value = 0;
    this.inp = null;
  }

  forward(input) {
    this.inp = input;
    const { I, H1, H2 } = this;
    for (let i = 0; i < H1; i++) {
      let s = this.b1[i];
      for (let j = 0; j < I; j++) s += this.W1[i * I + j] * input[j];
      this.z1[i] = s;
      this.a1[i] = Math.tanh(s);
    }
    for (let i = 0; i < H2; i++) {
      let s = this.b2[i];
      for (let j = 0; j < H1; j++) s += this.W2[i * H1 + j] * this.a1[j];
      this.z2[i] = s;
      this.a2[i] = Math.tanh(s);
    }
    let v = this.b3[0];
    for (let j = 0; j < H2; j++) v += this.W3[j] * this.a2[j];
    this.value = v;
    return v;
  }

  gradValue() {
    const { I, H1, H2 } = this;
    const gW3 = new Float64Array(H2);
    const gb3 = new Float64Array(1);
    for (let j = 0; j < H2; j++) gW3[j] = this.a2[j];
    gb3[0] = 1.0;

    const dA2 = new Float64Array(H2);
    for (let j = 0; j < H2; j++) dA2[j] = this.W3[j] * (1 - this.a2[j] * this.a2[j]);

    const gW2 = new Float64Array(H2 * H1);
    const gb2 = new Float64Array(H2);
    for (let i = 0; i < H2; i++) {
      gb2[i] = dA2[i];
      for (let j = 0; j < H1; j++) gW2[i * H1 + j] = dA2[i] * this.a1[j];
    }

    const dA1 = new Float64Array(H1);
    for (let j = 0; j < H1; j++) {
      let s = 0;
      for (let i = 0; i < H2; i++) s += this.W2[i * H1 + j] * dA2[i];
      dA1[j] = s * (1 - this.a1[j] * this.a1[j]);
    }

    const gW1 = new Float64Array(H1 * I);
    const gb1 = new Float64Array(H1);
    for (let i = 0; i < H1; i++) {
      gb1[i] = dA1[i];
      for (let j = 0; j < I; j++) gW1[i * I + j] = dA1[i] * this.inp[j];
    }

    return { gW1, gb1, gW2, gb2, gW3, gb3 };
  }

  paramCount() {
    return this.W1.length + this.b1.length +
           this.W2.length + this.b2.length +
           this.W3.length + this.b3.length;
  }
}

// --- PPO Trainer (Continuous Gaussian) ---
const rl = {
  active: false,
  actor: null,
  critic: null,
  bestActor: null,

  episode: 0,
  step: 0,
  maxSteps: 1000,
  episodeReward: 0,
  rewardHistory: [],
  avgRewards: [],
  bestReward: -Infinity,

  // PPO hyperparameters
  lr: 1e-3,
  gamma: 0.99,
  gaeLambda: 0.95,
  clipRange: 0.2,
  entCoeff: 0.001,
  vfCoeff: 0.5,
  maxGradNorm: 5.0,
  nSteps: 2048,
  batchSize: 64,
  nEpochs: 4,

  speed: 10,
  cachedAction: 0,
  cachedForce: 0,
  cachedLogProb: 0,
  cachedValue: 0,
  cachedState: null,

  rollout: { states: [], actions: [], rewards: [], dones: [], logProbs: [], values: [] },

  init() {
    this.actor = new ActorNet();
    this.critic = new CriticNet();
    this.bestActor = null;
    this.episode = 0;
    this.step = 0;
    this.episodeReward = 0;
    this.rewardHistory = [];
    this.avgRewards = [];
    this.bestReward = -Infinity;
    this.rollout = { states: [], actions: [], rewards: [], dones: [], logProbs: [], values: [] };
    this.cachedAction = 0;
    this.cachedForce = 0;
    this.prevForce = 0;
    this.perturbStep = -1;  // step when last perturbation occurred (-1 = none)
    this.active = true;
  },

  stop() { this.active = false; },

  randomInitState() {
    return {
      x: (Math.random() - 0.5) * 1.0,        // ±0.5m
      xdot: (Math.random() - 0.5) * 1.0,      // ±0.5 m/s
      theta: (Math.random() - 0.5) * 0.7,      // ±0.35 rad (~20°)
      thetadot: (Math.random() - 0.5) * 2.0    // ±1.0 rad/s
    };
  },

  computeAction(s) {
    const ns = normalizeState(s, this.prevForce);
    this.actor.forward(ns);
    const rawAction = this.actor.sample();
    this.cachedAction = rawAction;  // store unclipped for log prob
    this.cachedForce = Math.max(-maxForce, Math.min(maxForce, rawAction));
    this.cachedLogProb = this.actor.logProb(rawAction);
    this.cachedValue = this.critic.forward(ns);
    this.cachedState = ns;
    this.prevForce = this.cachedForce;
    return this.cachedForce;
  },

  recordStep(reward, done) {
    this.rollout.states.push(this.cachedState);
    this.rollout.actions.push(this.cachedAction);
    this.rollout.rewards.push(reward);
    this.rollout.dones.push(done);
    this.rollout.logProbs.push(this.cachedLogProb);
    this.rollout.values.push(this.cachedValue);
    this.step++;
    this.episodeReward += reward;
  },

  checkDone(s) {
    return Math.abs(wrapAngle(s.theta)) > Math.PI / 3.0
        || Math.abs(s.x) > 4.0
        || this.step >= this.maxSteps;
  },

  endEpisode() {
    this.rewardHistory.push(this.episodeReward);
    this.episode++;
    if (this.episodeReward > this.bestReward) {
      this.bestReward = this.episodeReward;
      this.bestActor = this.actor.clone();
    }
    this.step = 0;
    this.episodeReward = 0;
    this.prevForce = 0;
    this.perturbStep = -1;
    const win = 20;
    const recent = this.rewardHistory.slice(-win);
    this.avgRewards.push(recent.reduce((a, b) => a + b, 0) / recent.length);
  },

  shouldTrain() { return this.rollout.states.length >= this.nSteps; },

  trainPPO(bootstrapValue) {
    const T = this.rollout.states.length;
    if (T === 0) return;

    // --- GAE advantages & returns ---
    const advantages = new Float64Array(T);
    const returns = new Float64Array(T);
    let lastGae = 0;
    for (let t = T - 1; t >= 0; t--) {
      const nextNonTerm = this.rollout.dones[t] ? 0 : 1;
      const nextVal = (t === T - 1)
        ? (this.rollout.dones[t] ? 0 : bootstrapValue)
        : (this.rollout.dones[t] ? 0 : this.rollout.values[t + 1]);
      const delta = this.rollout.rewards[t] + this.gamma * nextVal - this.rollout.values[t];
      lastGae = delta + this.gamma * this.gaeLambda * nextNonTerm * lastGae;
      advantages[t] = lastGae;
    }
    for (let t = 0; t < T; t++) returns[t] = advantages[t] + this.rollout.values[t];

    // Normalize advantages
    let advMean = 0;
    for (let t = 0; t < T; t++) advMean += advantages[t];
    advMean /= T;
    let advVar = 0;
    for (let t = 0; t < T; t++) advVar += (advantages[t] - advMean) ** 2;
    const advStd = Math.sqrt(advVar / T) + 1e-8;
    for (let t = 0; t < T; t++) advantages[t] = (advantages[t] - advMean) / advStd;

    // --- PPO mini-batch updates ---
    const indices = Array.from({length: T}, (_, i) => i);
    const { I, H1, H2 } = this.actor;

    for (let epoch = 0; epoch < this.nEpochs; epoch++) {
      // Fisher-Yates shuffle
      for (let i = T - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      let logStdGrad = 0;  // accumulate logStd gradient over all batches this epoch

      for (let start = 0; start < T; start += this.batchSize) {
        const end = Math.min(start + this.batchSize, T);
        const bLen = end - start;

        // Actor gradient accumulators (scalar output → same shape as critic)
        const aG = {
          gW1: new Float64Array(H1 * I), gb1: new Float64Array(H1),
          gW2: new Float64Array(H2 * H1), gb2: new Float64Array(H2),
          gW3: new Float64Array(H2), gb3: new Float64Array(1)
        };
        // Critic gradient accumulators
        const cG = {
          gW1: new Float64Array(H1 * I), gb1: new Float64Array(H1),
          gW2: new Float64Array(H2 * H1), gb2: new Float64Array(H2),
          gW3: new Float64Array(H2), gb3: new Float64Array(1)
        };

        const sigma = Math.exp(this.actor.logStd);
        const sigSq = sigma * sigma;

        for (let b = start; b < end; b++) {
          const idx = indices[b];
          const st = this.rollout.states[idx];
          const act = this.rollout.actions[idx];
          const oldLP = this.rollout.logProbs[idx];
          const adv = advantages[idx];
          const ret = returns[idx];

          // --- Actor ---
          this.actor.forward(st);
          const newLP = this.actor.logProb(act);
          const ratio = Math.exp(newLP - oldLP);
          const clipped = Math.max(1 - this.clipRange, Math.min(1 + this.clipRange, ratio));
          const surr1 = ratio * adv;
          const surr2 = clipped * adv;

          // PPO clip: gradient from surrogate when not clipped, else 0
          const ppoWeight = (surr1 <= surr2) ? adv * ratio : 0;

          // ∂log p/∂μ = (a - μ) / σ² → this is the chain rule factor for network params
          const dLogP_dMu = (act - this.actor.mu) / sigSq;

          // Actor network gradient: ppoWeight * dLogP_dMu * ∂μ/∂θ
          const tw = ppoWeight * dLogP_dMu / bLen;
          const ag = this.actor.gradMu();
          for (let i = 0; i < aG.gW1.length; i++) aG.gW1[i] += tw * ag.gW1[i];
          for (let i = 0; i < aG.gb1.length; i++) aG.gb1[i] += tw * ag.gb1[i];
          for (let i = 0; i < aG.gW2.length; i++) aG.gW2[i] += tw * ag.gW2[i];
          for (let i = 0; i < aG.gb2.length; i++) aG.gb2[i] += tw * ag.gb2[i];
          for (let i = 0; i < aG.gW3.length; i++) aG.gW3[i] += tw * ag.gW3[i];
          for (let i = 0; i < aG.gb3.length; i++) aG.gb3[i] += tw * ag.gb3[i];

          // logStd gradient: ppoWeight * ((a-μ)²/σ² - 1) + entCoeff * 1
          const diff = act - this.actor.mu;
          const dLogP_dLogStd = diff * diff / sigSq - 1.0;
          logStdGrad += (ppoWeight * dLogP_dLogStd + this.entCoeff) / bLen;

          // --- Critic ---
          this.critic.forward(st);
          const vErr = this.vfCoeff * (this.critic.value - ret) / bLen;
          const cg = this.critic.gradValue();
          for (let i = 0; i < cG.gW1.length; i++) cG.gW1[i] += vErr * cg.gW1[i];
          for (let i = 0; i < cG.gb1.length; i++) cG.gb1[i] += vErr * cg.gb1[i];
          for (let i = 0; i < cG.gW2.length; i++) cG.gW2[i] += vErr * cg.gW2[i];
          for (let i = 0; i < cG.gb2.length; i++) cG.gb2[i] += vErr * cg.gb2[i];
          for (let i = 0; i < cG.gW3.length; i++) cG.gW3[i] += vErr * cg.gW3[i];
          for (let i = 0; i < cG.gb3.length; i++) cG.gb3[i] += vErr * cg.gb3[i];
        }

        // Clip gradients and apply
        this._clipApply(this.actor, aG, this.lr);    // ascent
        this._clipApply(this.critic, cG, -this.lr);  // descent
      }

      // Update logStd once per epoch (averaged over all batches)
      const nBatches = Math.ceil(T / this.batchSize);
      this.actor.logStd += this.lr * logStdGrad / nBatches;
      // Clamp to prevent collapse or explosion
      this.actor.logStd = Math.max(-3, Math.min(2, this.actor.logStd));
    }

    this.rollout = { states: [], actions: [], rewards: [], dones: [], logProbs: [], values: [] };
  },

  _clipApply(net, grads, lr) {
    let normSq = 0;
    for (const k of ['gW1','gb1','gW2','gb2','gW3','gb3']) {
      const g = grads[k];
      for (let i = 0; i < g.length; i++) normSq += g[i] * g[i];
    }
    const norm = Math.sqrt(normSq) + 1e-8;
    const step = lr * (norm > this.maxGradNorm ? this.maxGradNorm / norm : 1.0);
    for (let i = 0; i < net.W1.length; i++) net.W1[i] += step * grads.gW1[i];
    for (let i = 0; i < net.b1.length; i++) net.b1[i] += step * grads.gb1[i];
    for (let i = 0; i < net.W2.length; i++) net.W2[i] += step * grads.gW2[i];
    for (let i = 0; i < net.b2.length; i++) net.b2[i] += step * grads.gb2[i];
    for (let i = 0; i < net.W3.length; i++) net.W3[i] += step * grads.gW3[i];
    for (let i = 0; i < net.b3.length; i++) net.b3[i] += step * grads.gb3[i];
  },

  getAction(s) {
    const net = this.bestActor || this.actor;
    if (!net) return 0;
    net.forward(normalizeState(s, this.prevForce));
    const force = Math.max(-maxForce, Math.min(maxForce, net.mu));
    this.prevForce = force;
    return force;
  },

  // --- Save/Load network state ---

  _serializeNet(net) {
    return {
      W1: Array.from(net.W1), b1: Array.from(net.b1),
      W2: Array.from(net.W2), b2: Array.from(net.b2),
      W3: Array.from(net.W3), b3: Array.from(net.b3),
      logStd: net.logStd !== undefined ? net.logStd : undefined
    };
  },

  _loadNet(net, data) {
    net.W1.set(data.W1); net.b1.set(data.b1);
    net.W2.set(data.W2); net.b2.set(data.b2);
    net.W3.set(data.W3); net.b3.set(data.b3);
    if (data.logStd !== undefined) net.logStd = data.logStd;
  },

  save() {
    if (!this.actor) return null;
    const data = {
      version: 2,
      maxForce: maxForce,
      actor: this._serializeNet(this.actor),
      critic: this._serializeNet(this.critic),
      bestActor: this.bestActor ? this._serializeNet(this.bestActor) : null,
      episode: this.episode,
      bestReward: this.bestReward,
      rewardHistory: this.rewardHistory,
      avgRewards: this.avgRewards
    };
    return JSON.stringify(data);
  },

  load(json) {
    const data = JSON.parse(json);
    if (!this.actor) this.init();
    this.stop();
    this._loadNet(this.actor, data.actor);
    this._loadNet(this.critic, data.critic);
    if (data.bestActor) {
      this.bestActor = new ActorNet();
      this._loadNet(this.bestActor, data.bestActor);
    }
    this.episode = data.episode || 0;
    this.bestReward = data.bestReward || -Infinity;
    this.rewardHistory = data.rewardHistory || [];
    this.avgRewards = data.avgRewards || [];
    if (data.maxForce) maxForce = data.maxForce;
    this.step = 0;
    this.episodeReward = 0;
    this.prevForce = 0;
    this.perturbStep = -1;
    this.rollout = { states: [], actions: [], rewards: [], dones: [], logProbs: [], values: [] };
  },

  saveToFile() {
    const json = this.save();
    if (!json) return;
    const blob = new Blob([json], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `pendulum-ppo-ep${this.episode}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  },

  loadFromFile() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        this.load(ev.target.result);
        if (typeof updateCtrlParams === 'function') updateCtrlParams();
      };
      reader.readAsText(file);
    };
    input.click();
  }
};

function controllerNN(s) {
  if (!rl.actor) return 0;
  if (rl.active) return rl.cachedForce;
  return rl.getAction(s);
}
