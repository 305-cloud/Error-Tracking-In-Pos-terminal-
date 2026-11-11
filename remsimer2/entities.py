"""Payment entities and their fault processes.

Each entity (a PTSA, a switch, a bank, ...) is a SimPy-driven object with:

* a health state UP / DEGRADED / DOWN driven by a two-way Markov process:
  time to failure ~ Exp(MTBF), repair time ~ lognormal(MTTR median);
* load-dependent latency: base lognormal x (1 + alpha * (inflight/capacity)^2)
  x degraded multiplier, so failures cluster at peaks;
* its own random stream, so one entity's faults can be toggled and the
  scenario replayed with everything else unchanged (counterfactuals).

Every fault episode is written to the hidden ground-truth log, never to the
observable spans.
"""

import zlib

import numpy as np

UP, DEGRADED, DOWN = "UP", "DEGRADED", "DOWN"


def rng_for(seed, name):
    """Independent, reproducible random stream per named component."""
    key = zlib.crc32(name.encode())
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(key,)))


class Entity:
    def __init__(self, env, role, name, params, seed, episodes, calendar, faults_enabled=True):
        self.env = env
        self.role = role
        self.name = name
        self.state = UP
        self.inflight = 0
        self.episode_id = None
        self.episode_kind = None
        self.episode_start = None
        t = params["timing"]
        self.base_ms = t["base_latency_ms"][role]
        self.sigma = t["latency_sigma"]
        self.capacity = t["capacity_inflight"].get(role)
        self.alpha = t["overload_alpha"]
        f = params["faults"]
        self.degraded_x = f["degraded_latency_x"]
        self.degraded_error_p = f["degraded_error_p"]
        self.rng = rng_for(seed, f"latency:{role}:{name}")
        self._episodes = episodes
        self._calendar = calendar
        cfg = f["by_role"].get(role)
        if cfg and faults_enabled:
            self.mtbf_s = cfg["mtbf_h"] * 3600
            self.mttr_median_s = cfg["mttr_median_min"] * 60
            self.mttr_sigma = f["mttr_sigma"]
            self.p_down = cfg["p_down"]
            self.peak_x = f["peak_fault_multiplier"]
            self.fault_rng = rng_for(seed, f"faults:{role}:{name}")
            env.process(self._fault_process())

    @property
    def fqn(self):
        return f"{self.role}:{self.name}"

    def _fault_process(self):
        while True:
            # Failure rate is raised on peak days (month-end, December).
            rate_x = self.peak_x if self._calendar.is_peak_day(self.env.now) else 1.0
            yield self.env.timeout(self.fault_rng.exponential(self.mtbf_s / rate_x))
            kind = DOWN if self.fault_rng.random() < self.p_down else DEGRADED
            repair = self.fault_rng.lognormal(np.log(self.mttr_median_s), self.mttr_sigma)
            self.episode_id = f"EP-{self.role}-{self.name}-{len(self._episodes):05d}"
            self.episode_kind = kind
            self.state = kind
            self.episode_start = self.env.now
            yield self.env.timeout(repair)
            self.close_episode()

    def close_episode(self):
        """Record the current episode (also called at the end of a run for open ones)."""
        if self.episode_id is None:
            return
        self._episodes.append({
            "episode_id": self.episode_id, "entity": self.fqn, "role": self.role,
            "fault_type": self.episode_kind, "t_start": self.episode_start, "t_end": self.env.now,
        })
        self.state, self.episode_id, self.episode_kind = UP, None, None

    def sample_latency_s(self):
        """Return (latency_s, nominal_s). The excess over nominal is used for ground truth."""
        nominal = self.base_ms * float(np.exp(self.rng.normal(0, self.sigma))) / 1000
        lat = nominal
        if self.capacity:
            lat *= 1 + self.alpha * (self.inflight / self.capacity) ** 2
        if self.state == DEGRADED:
            lat *= self.degraded_x
        return lat, nominal

    def degraded_error(self):
        return self.state == DEGRADED and self.rng.random() < self.degraded_error_p
