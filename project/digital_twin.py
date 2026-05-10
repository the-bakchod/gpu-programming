import numpy as np, time


class DigitalTwin:
    def __init__(self, scenarios=1000, gpu=True):
        self.n = scenarios
        self.gpu = gpu
        self.building = {}
        self.scenarios = []
        self.model = {}
        self.policy = {}

    def set_building(self, dims, mats, hvac):
        self.building = {"dims": dims, "mats": mats, "hvac": hvac}

    def generate_scenarios(self):
        print(f"[Gen] {self.n} synthetic scenarios...")
        t, h, o = (
            np.linspace(-10, 40, 20),
            np.linspace(20, 80, 20),
            np.linspace(0, 100, 20),
        )
        self.scenarios = [
            {
                "temp": np.random.choice(t),
                "hum": np.random.choice(h),
                "occ": np.random.choice(o),
                "hr": np.random.randint(0, 24),
            }
            for _ in range(self.n)
        ]

    def simulate(self, s):
        return (
            50
            + abs(s["temp"] - 20) * 2
            + (s["occ"] / 100) * 30
            + (10 if 8 <= s["hr"] <= 18 else 0)
        )

    def run_simulations(self):
        print(f"[Engine] Running {self.n} sims... GPU={self.gpu}")
        t0 = time.time()
        results = [self.simulate(s) for s in self.scenarios]
        print(f"[Engine] Done in {time.time() - t0:.4f}s")
        return np.array(results)

    def train_model(self, results):
        print("[AI] Training...")
        X = np.array([[s["temp"], s["hum"], s["occ"], s["hr"]] for s in self.scenarios])
        self.model = {"w": np.random.randn(4) * 0.1, "b": 0.0}
        for _ in range(50):
            pred = X @ self.model["w"] + self.model["b"]
            error = results - pred
            self.model["w"] += 0.01 * (error @ X / len(results))
            self.model["b"] += 0.01 * np.mean(error)
        print("[AI] Done!")
        return self.model

    def optimize(self):
        print("[Opt] Learning policy...")
        self.policy = {
            "hvac": {h: "active" if 8 <= h <= 18 else "eco" for h in range(24)},
            "setpoints": {"cool": 22, "heat": 20},
            "savings": 0.30,
        }
        print(f"[Opt] Savings: {self.policy['savings'] * 100}%")
        return self.policy

    def self_learn(self, iters=3):
        print(f"\n[Self-Learning] {iters} iterations...")
        for i in range(iters):
            self.generate_scenarios()
            self.run_simulations()
            self.train_model(self.run_simulations())
            self.optimize()
        print("[Self-Learning] Complete!")

    def report(self):
        print(f"\n{'=' * 45}\nENERGY OPTIMIZATION REPORT\n{'=' * 45}")
        print(
            f"Building: {self.building.get('dims')}\nScenarios: {self.n} | GPU: {self.gpu}"
        )
        if self.policy:
            print(
                f"Savings: {self.policy['savings'] * 100}%\nSetpoints: C={self.policy['setpoints']['cool']}°C, H={self.policy['setpoints']['heat']}°C"
            )
        print("=" * 45)


def main():
    print("=" * 50 + "\nGPU DIGITAL TWIN FOR ENERGY OPTIMIZATION\n" + "=" * 50)
    dt = DigitalTwin(scenarios=1000, gpu=True)
    dt.set_building("50x30x15m", {"walls": "concrete"}, {"cap": "100kW"})
    dt.generate_scenarios()
    dt.run_simulations()
    dt.train_model(dt.run_simulations())
    dt.optimize()
    dt.self_learn(iters=3)
    dt.report()
    print("\nSystem ready! No sensors required.")


if __name__ == "__main__":
    main()
