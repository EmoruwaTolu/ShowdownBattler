from typing import Any

from bot.model.ctx import EvalContext
from bot.scoring.helpers import hp_frac


def score_switch(pokemon: Any, battle: Any, ctx: EvalContext) -> float:
    """
    Placeholder switch scorer.
    You can port your existing switch logic here later.

    For now: prefer healthier switches lightly.
    """
    if pokemon is None or pokemon.fainted:
        return -999.0

    # mild preference to keep healthy options
    return 10.0 * hp_frac(pokemon)
