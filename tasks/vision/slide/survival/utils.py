from dataclasses import dataclass


@dataclass
class Metrics:
    loss: float
    c_index: float
