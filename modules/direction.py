from enum import IntEnum

class Direction(IntEnum):
    NORTH = 1
    NORTH_EAST = 2
    NORTH_WEST = 3
    SOUTH = 4
    SOUTH_EAST = 5
    SOUTH_WEST = 6
    EAST = 7
    WEST = 8

    def heading_degrees(self):
        if self == Direction.EAST:
            return 0        # Math: +X
        elif self == Direction.NORTH_EAST:
            return 45
        elif self == Direction.NORTH:
            return 90       # Math: +Y
        elif self == Direction.NORTH_WEST:
            return 135
        elif self == Direction.WEST:
            return 180      # Math: -X
        elif self == Direction.SOUTH_WEST:
            return 225
        elif self == Direction.SOUTH:
            return 270      # Math: -Y (or -90)
        elif self == Direction.SOUTH_EAST:
            return 315
        else:
            raise ValueError("Invalid direction")