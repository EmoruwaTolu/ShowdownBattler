from dataclasses import dataclass
from typing import Optional, Dict, Any

from poke_env.battle import Battle


@dataclass
class EvalContext:
    battle: Battle
    me: Any
    opp: Optional[Any]

    cache: Dict[str, Any]

    @staticmethod
    def from_battle(battle: Battle) -> "EvalContext":
        return EvalContext(
            battle=battle,
            me=battle.active_pokemon,
            opp=battle.opponent_active_pokemon,
            cache={},
        )
