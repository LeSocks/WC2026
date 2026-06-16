from __future__ import annotations

from dataclasses import dataclass

from src.models.player import Position


@dataclass(frozen=True)
class Formation:
    name: str
    slots: tuple[Position, ...]
    coordinates: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.slots) != 11:
            raise ValueError(f"{self.name} must define 11 position slots")
        if len(self.coordinates) != 11:
            raise ValueError(f"{self.name} must define 11 coordinates")


FORMATIONS: dict[str, Formation] = {
    "4-3-3": Formation(
        name="4-3-3",
        slots=(
            Position.GK,
            Position.RB,
            Position.CB,
            Position.CB,
            Position.LB,
            Position.CDM,
            Position.CM,
            Position.CM,
            Position.RW,
            Position.ST,
            Position.LW,
        ),
        coordinates=((8, 40), (28, 68), (25, 50), (25, 30), (28, 12), (48, 40), (62, 55), (62, 25), (88, 68), (96, 40), (88, 12)),
    ),
    "4-2-3-1": Formation(
        name="4-2-3-1",
        slots=(
            Position.GK,
            Position.RB,
            Position.CB,
            Position.CB,
            Position.LB,
            Position.CDM,
            Position.CM,
            Position.RW,
            Position.CAM,
            Position.LW,
            Position.ST,
        ),
        coordinates=((8, 40), (28, 68), (25, 50), (25, 30), (28, 12), (48, 50), (48, 30), (72, 68), (76, 40), (72, 12), (96, 40)),
    ),
    "4-5-1": Formation(
        name="4-5-1",
        slots=(
            Position.GK,
            Position.RB,
            Position.CB,
            Position.CB,
            Position.LB,
            Position.RW,
            Position.CM,
            Position.CDM,
            Position.CM,
            Position.LW,
            Position.ST,
        ),
        coordinates=((8, 40), (24, 68), (22, 50), (22, 30), (24, 12), (52, 68), (52, 52), (48, 40), (52, 28), (52, 12), (92, 40)),
    ),
    "3-4-3": Formation(
        name="3-4-3",
        slots=(
            Position.GK,
            Position.CB,
            Position.CB,
            Position.CB,
            Position.RB,
            Position.CM,
            Position.CM,
            Position.LB,
            Position.RW,
            Position.ST,
            Position.LW,
        ),
        coordinates=((8, 40), (25, 58), (22, 40), (25, 22), (52, 70), (55, 50), (55, 30), (52, 10), (86, 68), (96, 40), (86, 12)),
    ),
}


def get_formation(name: str) -> Formation:
    try:
        return FORMATIONS[name]
    except KeyError as error:
        valid = ", ".join(sorted(FORMATIONS))
        raise ValueError(f"Unsupported formation '{name}'. Expected one of: {valid}") from error
