from enum import IntEnum

class FormationType(IntEnum):
    LINE = 1      # Flying side-by-side (Abreast)
    V_SHAPE = 2   # Flying in a swept-back V
    CIRCLE = 3    # Flying in a ring
    COLUMN = 4    # Flying single-file (Trail)