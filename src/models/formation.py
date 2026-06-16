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
    "4-4-2": Formation(
        name="4-4-2",
        slots=(
            Position.GK,
            Position.RB,
            Position.CB,
            Position.CB,
            Position.LB,
            Position.RW,
            Position.CM,
            Position.CM,
            Position.LW,
            Position.ST,
            Position.ST,
        ),
        coordinates=((8, 40), (26, 68), (24, 50), (24, 30), (26, 12), (56, 68), (56, 50), (56, 30), (56, 12), (92, 48), (92, 32)),
    ),
    "4-1-4-1": Formation(
        name="4-1-4-1",
        slots=(
            Position.GK,
            Position.RB,
            Position.CB,
            Position.CB,
            Position.LB,
            Position.CDM,
            Position.RW,
            Position.CM,
            Position.CM,
            Position.LW,
            Position.ST,
        ),
        coordinates=((8, 40), (26, 68), (24, 50), (24, 30), (26, 12), (44, 40), (64, 68), (64, 50), (64, 30), (64, 12), (94, 40)),
    ),
    "4-3-2-1": Formation(
        name="4-3-2-1",
        slots=(
            Position.GK,
            Position.RB,
            Position.CB,
            Position.CB,
            Position.LB,
            Position.CDM,
            Position.CM,
            Position.CM,
            Position.CAM,
            Position.CAM,
            Position.ST,
        ),
        coordinates=((8, 40), (26, 68), (24, 50), (24, 30), (26, 12), (48, 40), (60, 56), (60, 24), (78, 50), (78, 30), (96, 40)),
    ),
    "4-3-1-2": Formation(
        name="4-3-1-2",
        slots=(
            Position.GK,
            Position.RB,
            Position.CB,
            Position.CB,
            Position.LB,
            Position.CDM,
            Position.CM,
            Position.CM,
            Position.CAM,
            Position.ST,
            Position.ST,
        ),
        coordinates=((8, 40), (26, 68), (24, 50), (24, 30), (26, 12), (48, 40), (60, 56), (60, 24), (78, 40), (94, 50), (94, 30)),
    ),
    "4-2-2-2": Formation(
        name="4-2-2-2",
        slots=(
            Position.GK,
            Position.RB,
            Position.CB,
            Position.CB,
            Position.LB,
            Position.CDM,
            Position.CM,
            Position.RW,
            Position.LW,
            Position.ST,
            Position.ST,
        ),
        coordinates=((8, 40), (26, 68), (24, 50), (24, 30), (26, 12), (48, 50), (48, 30), (72, 58), (72, 22), (94, 50), (94, 30)),
    ),
    "4-1-2-1-2": Formation(
        name="4-1-2-1-2",
        slots=(
            Position.GK,
            Position.RB,
            Position.CB,
            Position.CB,
            Position.LB,
            Position.CDM,
            Position.CM,
            Position.CM,
            Position.CAM,
            Position.ST,
            Position.ST,
        ),
        coordinates=((8, 40), (26, 68), (24, 50), (24, 30), (26, 12), (46, 40), (60, 54), (60, 26), (76, 40), (94, 50), (94, 30)),
    ),
    "3-5-2": Formation(
        name="3-5-2",
        slots=(
            Position.GK,
            Position.CB,
            Position.CB,
            Position.CB,
            Position.RB,
            Position.CM,
            Position.CDM,
            Position.CM,
            Position.LB,
            Position.ST,
            Position.ST,
        ),
        coordinates=((8, 40), (24, 58), (22, 40), (24, 22), (54, 70), (58, 54), (52, 40), (58, 26), (54, 10), (94, 50), (94, 30)),
    ),
    "3-4-2-1": Formation(
        name="3-4-2-1",
        slots=(
            Position.GK,
            Position.CB,
            Position.CB,
            Position.CB,
            Position.RB,
            Position.CM,
            Position.CM,
            Position.LB,
            Position.CAM,
            Position.CAM,
            Position.ST,
        ),
        coordinates=((8, 40), (24, 58), (22, 40), (24, 22), (54, 70), (56, 50), (56, 30), (54, 10), (78, 50), (78, 30), (96, 40)),
    ),
    "3-4-1-2": Formation(
        name="3-4-1-2",
        slots=(
            Position.GK,
            Position.CB,
            Position.CB,
            Position.CB,
            Position.RB,
            Position.CM,
            Position.CM,
            Position.LB,
            Position.CAM,
            Position.ST,
            Position.ST,
        ),
        coordinates=((8, 40), (24, 58), (22, 40), (24, 22), (54, 70), (56, 50), (56, 30), (54, 10), (78, 40), (94, 50), (94, 30)),
    ),
    "5-3-2": Formation(
        name="5-3-2",
        slots=(
            Position.GK,
            Position.RB,
            Position.CB,
            Position.CB,
            Position.CB,
            Position.LB,
            Position.CDM,
            Position.CM,
            Position.CM,
            Position.ST,
            Position.ST,
        ),
        coordinates=((8, 40), (28, 72), (24, 56), (22, 40), (24, 24), (28, 8), (52, 40), (62, 55), (62, 25), (94, 50), (94, 30)),
    ),
    "5-4-1": Formation(
        name="5-4-1",
        slots=(
            Position.GK,
            Position.RB,
            Position.CB,
            Position.CB,
            Position.CB,
            Position.LB,
            Position.RW,
            Position.CM,
            Position.CM,
            Position.LW,
            Position.ST,
        ),
        coordinates=((8, 40), (28, 72), (24, 56), (22, 40), (24, 24), (28, 8), (58, 68), (58, 50), (58, 30), (58, 12), (94, 40)),
    ),
    "5-2-3": Formation(
        name="5-2-3",
        slots=(
            Position.GK,
            Position.RB,
            Position.CB,
            Position.CB,
            Position.CB,
            Position.LB,
            Position.CM,
            Position.CM,
            Position.RW,
            Position.ST,
            Position.LW,
        ),
        coordinates=((8, 40), (28, 72), (24, 56), (22, 40), (24, 24), (28, 8), (56, 50), (56, 30), (86, 68), (96, 40), (86, 12)),
    ),
}


def get_formation(name: str) -> Formation:
    try:
        return FORMATIONS[name]
    except KeyError as error:
        valid = ", ".join(sorted(FORMATIONS))
        raise ValueError(f"Unsupported formation '{name}'. Expected one of: {valid}") from error
