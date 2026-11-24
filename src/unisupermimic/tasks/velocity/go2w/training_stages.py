"""Evaluation-driven Go2W stages and bounded withdrawal of reference collection."""

from dataclasses import asdict, dataclass

ORDER = ("flat_prior", "flat_autonomous", "terrain")


@dataclass
class TrainingStage:
    name: str = "flat_prior"
    max_stage: str = "flat_prior"
    auto_transition: bool = True
    consecutive_passes: int = 0
    entered_at: float = 0.0
    training_seconds: float = 0.0
    ready_for_next_stage: bool = False
    terrain_review: bool = True

    def __post_init__(self):
        if ORDER.index(self.name) > ORDER.index(self.max_stage):
            raise ValueError("Initial stage exceeds stage limit")

    def fraction(self, config):
        if self.name == "flat_prior":
            return 1.0
        if self.name == "terrain":
            return 0.0
        return max(0.0, 1 - (self.training_seconds - self.entered_at) / config["fade_seconds"])

    def collection_fractions(self, config):
        fraction = self.fraction(config)
        return config["reference_guided_fraction"] * fraction, config[
            "reference_reset_fraction"
        ] * fraction

    def bank_fraction(self, config):
        return config["motion_bank_command_fraction"] * self.fraction(config)

    def assess(self, evaluation, config):
        if self.name == "terrain":
            if evaluation.get("terrain_pass_rate", 0.0) >= config["terrain_review_pass_rate"]:
                self.terrain_review = False
            return False
        rates = evaluation["pass_rates"]
        pass_flat = evaluation["fall_rate"] <= config["max_fall_rate"] and all(
            rates.get(name, 0.0) >= threshold for name, threshold in config["pass_rates"].items()
        )
        if self.name == "flat_autonomous":
            pass_flat &= (
                self.fraction(config) == 0
                and evaluation.get("switch_pass_rate", 0.0) >= config["switch_pass_rate"]
            )
        self.consecutive_passes = self.consecutive_passes + 1 if pass_flat else 0
        self.ready_for_next_stage = self.consecutive_passes >= config["consecutive_evaluations"]
        if not self.ready_for_next_stage or not self.auto_transition or self.name == self.max_stage:
            return False
        self.name = ORDER[ORDER.index(self.name) + 1]
        self.entered_at = self.training_seconds
        self.ready_for_next_stage = False
        self.consecutive_passes = 0
        return True

    def state_dict(self):
        return asdict(self)

    def restore(self, state):
        limit, auto = self.max_stage, self.auto_transition
        for name in asdict(self):
            setattr(self, name, state[name])
        self.max_stage, self.auto_transition = limit, auto
        self.__post_init__()
