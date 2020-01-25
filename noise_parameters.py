from enum import Enum

class NOISE(Enum):
    LANGEVIN_CONSTANT = 1
    LANGEVIN_LINEAR = 2
    LANGEVIN_SQRT = 3
    RICKER_LINEAR = 4
    ARATO_LINEAR = 5
    ORNSTEIN_UHLENBECK = 6
    SQRT_MILSTEIN = 7
    # TODO change name to LINEAR_AND_INTERACTION
    GROWTH_AND_INTERACTION_LINEAR = 8
    LANGEVIN_LINEAR_SQRT = 9

class MODEL(Enum):
    GLV = 1
    QSMI = 2 # quadratic species metabolite interaction