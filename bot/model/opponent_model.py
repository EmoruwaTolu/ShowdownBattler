from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

from poke_env.data import GenData, to_id_str

@dataclass(frozen=True)
class SetCandidate:
    """
    One RandBats role candidate for a species.
    This is the unit we maintain belief over.
    """
    id: str  # e.g. "alcremie:bulkysetup"
    species_id: str

    # Constraints from the randbats JSON role
    moves: Set[str]          # move ids
    abilities: Set[str]      # ability ids
    items: Set[str]          # item ids
    tera_types: Set[str]     # tera type ids

    # Useful tags (cheap inference for current heuristics)
    is_physical: bool
    has_setup: bool
    has_priority: bool

    # Optional uncertainty knobs you already started
    speed_mult: float = 1.0
    physical_threat: float = 0.6


@dataclass
class OpponentBelief:
    """
    Persistent belief state for ONE opponent Pokémon.
    Stores P(role) and updates as we observe info.
    """
    species_id: str
    gen: int
    dist: List[Tuple[SetCandidate, float]]  # (candidate, prob), sums to 1

    # revealed info that can constrain roles / sampled move subsets
    revealed_moves: Set[str]
    revealed_item: Optional[str] = None
    revealed_ability: Optional[str] = None
    revealed_tera: Optional[str] = None

    def normalize(self) -> None:
        s = sum(w for _, w in self.dist)
        if s <= 0:
            # if we got inconsistent, fallback to uniform over candidates we still have
            if self.dist:
                u = 1.0 / len(self.dist)
                self.dist = [(c, u) for c, _ in self.dist]
            return
        self.dist = [(c, w / s) for c, w in self.dist]

    def _filter_and_renorm(self, keep: List[Tuple[SetCandidate, float]]) -> None:
        if keep:
            self.dist = keep
            self.normalize()

    def observe_move(self, move_id: str) -> None:
        mid = to_id_str(move_id)
        if not mid:
            return
        self.revealed_moves.add(mid)

        keep = [(c, w) for c, w in self.dist if mid in c.moves]
        self._filter_and_renorm(keep)

    def observe_item(self, item_id: Optional[str]) -> None:
        if not item_id:
            return
        iid = to_id_str(item_id)
        if not iid:
            return
        self.revealed_item = iid

        keep = [(c, w) for c, w in self.dist if (not c.items) or (iid in c.items)]
        self._filter_and_renorm(keep)

    def observe_ability(self, ability_id: Optional[str]) -> None:
        if not ability_id:
            return
        aid = to_id_str(ability_id)
        if not aid:
            return
        self.revealed_ability = aid

        keep = [(c, w) for c, w in self.dist if (not c.abilities) or (aid in c.abilities)]
        self._filter_and_renorm(keep)

    def observe_hazard_interaction(self, has_hazards: bool, took_damage: bool) -> None:
        """
        Infer item from hazard interaction on switch-in.

        If hazards are present:
        - took_damage=False → must have Heavy-Duty Boots → filter to HDB candidates
        - took_damage=True → doesn't have HDB → penalize HDB candidates
        """
        if not has_hazards:
            return

        hdb_id = "heavydutyboots"

        if not took_damage:
            # Must have HDB
            self.observe_item(hdb_id)
        else:
            # Doesn't have HDB - heavily penalize candidates with HDB
            adjusted = []
            for c, w in self.dist:
                if c.items and hdb_id in c.items:
                    adjusted.append((c, w * 0.1))
                else:
                    adjusted.append((c, w))
            self._filter_and_renorm(adjusted)

    def observe_tera(self, tera_type: Optional[str]) -> None:
        if not tera_type:
            return
        tid = to_id_str(tera_type)
        if not tid:
            return
        self.revealed_tera = tid

        keep = [(c, w) for c, w in self.dist if (not c.tera_types) or (tid in c.tera_types)]
        self._filter_and_renorm(keep)

    def as_distribution(self) -> List[Tuple[SetCandidate, float]]:
        return list(self.dist)

    def sample_role(self, rng: random.Random) -> SetCandidate:
        cands = [c for c, _ in self.dist]
        weights = [w for _, w in self.dist]
        if not cands:
            raise ValueError("OpponentBelief has empty dist")
        return rng.choices(cands, weights=weights, k=1)[0]


@dataclass(frozen=True)
class DeterminizedOpponent:
    """
    A single sampled "world" for MCTS rollout:
    - specific role candidate
    - specific 4-move subset consistent with revealed moves
    """
    candidate: SetCandidate
    moves4: Tuple[str, str, str, str]


@dataclass
class TeamBelief:
    """
    Team-level belief over unseen opponent slots.
    pool: species_key -> weight (P(species) for remaining unseen mons)
    """
    revealed_species: Set[str]
    pool: Dict[str, float]

    def has_mass(self) -> bool:
        """True if there is nonzero probability mass to sample from."""
        return sum(self.pool.values()) > 1e-9

    def without(self, species_key: str) -> "TeamBelief":
        """Return a new TeamBelief with this species removed (without replacement)."""
        new_pool = {k: v for k, v in self.pool.items() if to_id_str(k) != to_id_str(species_key)}
        total = sum(new_pool.values())
        if total <= 0:
            return TeamBelief(revealed_species=self.revealed_species, pool={})
        return TeamBelief(
            revealed_species=self.revealed_species,
            pool={k: v / total for k, v in new_pool.items()},
        )


def build_team_belief(gen: int, revealed_species: Set[str]) -> TeamBelief:
    """
    Build TeamBelief from RandBats prior, excluding revealed species.
    pool is P(species) for unseen slots, renormalized.
    """
    data = _load_randbats_json(gen)
    if not data:
        return TeamBelief(revealed_species=revealed_species, pool={})
    revealed_norm = {to_id_str(s) for s in revealed_species if s}
    pool: Dict[str, float] = {}
    for species_key in data.keys():
        if not species_key:
            continue
        if to_id_str(species_key) in revealed_norm:
            continue
        pool[species_key] = 1.0
    total = sum(pool.values())
    if total <= 0:
        return TeamBelief(revealed_species=revealed_species, pool={})
    normalized = {k: v / total for k, v in pool.items()}
    return TeamBelief(revealed_species=revealed_species, pool=normalized)


def sample_unseen_mon_from_team_belief(
    team_belief: TeamBelief,
    gen: int,
    rng: random.Random,
) -> Tuple[Optional[Any], Optional[TeamBelief]]:
    """
    Sample a Pokémon from team_belief.pool (species) then RandBats set prior.
    Returns (mon, new_team_belief) with species removed from pool (without replacement).
    """
    if not team_belief.has_mass():
        return (None, None)
    species_key = rng.choices(
        list(team_belief.pool.keys()),
        weights=list(team_belief.pool.values()),
        k=1,
    )[0]
    candidates = lookup_randbats_candidates(species_key, gen)
    if not candidates:
        cand = None
    else:
        cand = rng.choice(candidates)
    if cand is None:
        return (None, team_belief.without(species_key))
    mon = _create_pokemon_proxy(cand, species_key, gen, rng)
    new_belief = team_belief.without(species_key)
    return (mon, new_belief)


_DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "randbats")
)

_GEN_TO_FILENAME: Dict[int, str] = {
    9: "gen9randombattle.json",
}


@lru_cache(maxsize=32)
def _load_randbats_json(gen: int) -> Dict[str, Any]:
    filename = _GEN_TO_FILENAME.get(int(gen))
    if not filename:
        return {}
    path = os.path.normpath(os.path.join(_DEFAULT_DATA_DIR, filename))
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=32)
def _species_key_index(gen: int) -> Dict[str, str]:
    """
    to_id_str(speciesName) -> exact JSON key
    """
    data = _load_randbats_json(gen)
    idx: Dict[str, str] = {}
    for k in data.keys():
        idx[to_id_str(k)] = k
    return idx


def _move_entry(gen: int, move_id: str) -> Optional[Dict[str, Any]]:
    try:
        return GenData.from_gen(gen).moves.get(move_id)
    except Exception:
        return None


def _revealed_move_ids(opp: Any) -> Set[str]:
    try:
        # poke_env usually stores moves as dict keyed by move_id_str
        return set((getattr(opp, "moves", {}) or {}).keys())
    except Exception:
        return set()


def _species_id_from_opp(opp: Any) -> Optional[str]:
    for attr in ("species", "species_id", "base_species"):
        try:
            v = getattr(opp, attr, None)
            if isinstance(v, str) and v.strip():
                return to_id_str(v)
        except Exception:
            pass
    return None


def _revealed_item_id(opp: Any) -> Optional[str]:
    # poke_env sometimes exposes item as opp.item or opp.item_id
    for attr in ("item", "item_id"):
        try:
            v = getattr(opp, attr, None)
            if isinstance(v, str) and v.strip():
                return to_id_str(v)
        except Exception:
            pass
    return None


def _revealed_ability_id(opp: Any) -> Optional[str]:
    for attr in ("ability", "ability_id"):
        try:
            v = getattr(opp, attr, None)
            if isinstance(v, str) and v.strip():
                return to_id_str(v)
        except Exception:
            pass
    return None


def _revealed_tera_type(opp: Any) -> Optional[str]:
    for attr in ("tera_type", "teratype", "terastallized_type", "terastallizedType"):
        try:
            v = getattr(opp, attr, None)
            if isinstance(v, str) and v.strip():
                return to_id_str(v)
        except Exception:
            pass
    return None


def _infer_candidate_tags(gen: int, move_ids: Set[str]) -> Tuple[bool, bool, bool]:
    """
    Infer (is_physical, has_setup, has_priority) from move data.
    """
    phys_dmg = 0
    spec_dmg = 0
    dmg_total = 0

    has_priority = False
    has_setup = False

    setup_name_fallback = {
        "swordsdance", "bulkup", "dragondance", "nastyplot", "calmmind", "quiverdance",
        "shellsmash", "agility", "rockpolish", "shiftgear", "bellydrum",
        "curse", "coils", "cosmicpower", "workup",
    }

    for mid in move_ids:
        entry = _move_entry(gen, mid)

        # priority
        if entry is not None:
            try:
                if int(entry.get("priority", 0) or 0) > 0:
                    has_priority = True
            except Exception:
                pass

        # setup via boosts/selfBoost/self.boosts
        if entry is not None:
            boosts = entry.get("boosts")
            if isinstance(boosts, dict) and any((v or 0) > 0 for v in boosts.values()):
                has_setup = True

            sb = entry.get("selfBoost")
            if isinstance(sb, dict):
                sb2 = sb.get("boosts")
                if isinstance(sb2, dict) and any((v or 0) > 0 for v in sb2.values()):
                    has_setup = True

            self_obj = entry.get("self")
            if isinstance(self_obj, dict):
                sb3 = self_obj.get("boosts")
                if isinstance(sb3, dict) and any((v or 0) > 0 for v in sb3.values()):
                    has_setup = True

        if mid in setup_name_fallback:
            has_setup = True

        # physical vs special: count damaging moves only
        if entry is None:
            continue
        try:
            bp = int(entry.get("basePower", 0) or 0)
            if bp <= 0:
                continue
            dmg_total += 1
            cat = str(entry.get("category", "")).upper()
            if cat == "PHYSICAL":
                phys_dmg += 1
            elif cat == "SPECIAL":
                spec_dmg += 1
        except Exception:
            pass

    if dmg_total == 0:
        is_physical = False
    else:
        is_physical = phys_dmg >= max(1, int(0.6 * dmg_total))

    return is_physical, has_setup, has_priority


def lookup_randbats_candidates(species_id: str, gen: int) -> Optional[List[SetCandidate]]:
    """
    Parse candidates from the RandBats roles JSON.

    Each role becomes one SetCandidate.
    """
    data = _load_randbats_json(gen)
    if not data:
        return None

    idx = _species_key_index(gen)
    key = idx.get(to_id_str(species_id))
    if not key:
        return None

    entry = data.get(key) or {}
    roles = entry.get("roles") or {}
    if not isinstance(roles, dict) or not roles:
        return None

    species_id_norm = to_id_str(key)

    candidates: List[SetCandidate] = []
    for role_name, role_obj in roles.items():
        moves = role_obj.get("moves") or []
        abilities = role_obj.get("abilities") or []
        items = role_obj.get("items") or []
        tera_types = role_obj.get("teraTypes") or []

        move_ids = {to_id_str(m) for m in moves if isinstance(m, str)}
        ability_ids = {to_id_str(a) for a in abilities if isinstance(a, str)}
        item_ids = {to_id_str(i) for i in items if isinstance(i, str)}
        tera_ids = {to_id_str(t) for t in tera_types if isinstance(t, str)}

        is_phys, has_setup, has_prio = _infer_candidate_tags(gen, move_ids)
        phys_threat = 0.65 if is_phys else 0.35

        candidates.append(
            SetCandidate(
                id=f"{species_id_norm}:{to_id_str(str(role_name))}",
                species_id=species_id_norm,
                moves=move_ids,
                abilities=ability_ids,
                items=item_ids,
                tera_types=tera_ids,
                is_physical=is_phys,
                has_setup=has_setup,
                has_priority=has_prio,
                speed_mult=1.0,
                physical_threat=phys_threat,
            )
        )

    return candidates


def _default_candidates_from_seen(opp: Any, gen: int) -> List[SetCandidate]:
    """
    Fallback if we can't load a species entry from the randbats DB.
    Creates a single candidate consistent with revealed moves.
    """
    seen_moves = _revealed_move_ids(opp)

    # crude physical guess from base stats if available
    is_phys = False
    try:
        bs = getattr(opp, "base_stats", None) or {}
        atk, spa = bs.get("atk", 100), bs.get("spa", 100)
        is_phys = atk >= spa * 1.10
    except Exception:
        pass

    is_phys2, has_setup, has_prio = _infer_candidate_tags(gen, set(seen_moves))
    is_phys = is_phys or is_phys2

    sid = _species_id_from_opp(opp) or "unknown"

    return [
        SetCandidate(
            id=f"{to_id_str(sid)}:fallback",
            species_id=to_id_str(sid),
            moves=set(seen_moves),
            abilities=set(),
            items=set(),
            tera_types=set(),
            is_physical=is_phys,
            has_setup=has_setup,
            has_priority=has_prio,
            speed_mult=1.0,
            physical_threat=0.65 if is_phys else 0.35,
        )
    ]


def build_opponent_belief(opp: Any, gen: int) -> OpponentBelief:
    """
    Build a belief object from the current opp snapshot (moves, maybe item/ability/tera). 
    """
    species_id = _species_id_from_opp(opp)
    revealed_moves = set(_revealed_move_ids(opp))

    if not species_id:
        cands = _default_candidates_from_seen(opp, gen)
        return OpponentBelief(
            species_id=cands[0].species_id,
            gen=gen,
            dist=[(cands[0], 1.0)],
            revealed_moves=revealed_moves,
        )

    candidates = lookup_randbats_candidates(species_id, gen)
    if not candidates:
        candidates = _default_candidates_from_seen(opp, gen)

    # Start uniform
    base_w = 1.0 / float(len(candidates)) if candidates else 1.0
    dist: List[Tuple[SetCandidate, float]] = [(c, base_w) for c in candidates]

    belief = OpponentBelief(
        species_id=to_id_str(species_id),
        gen=gen,
        dist=dist,
        revealed_moves=set(),
    )

    # Apply observations (hard filters)
    for m in revealed_moves:
        belief.observe_move(m)

    belief.observe_item(_revealed_item_id(opp))
    belief.observe_ability(_revealed_ability_id(opp))
    belief.observe_tera(_revealed_tera_type(opp))

    # store the revealed moves (observe_move already inserted them)
    return belief


def get_opponent_set_distribution(opp: Any, gen: int) -> List[Tuple[SetCandidate, float]]:
    """
    Backwards-compatible function: returns (candidate, weight) sum to 1.
    Uses build_opponent_belief and returns its distribution.
    """
    return build_opponent_belief(opp, gen).as_distribution()


def physical_prob(dist: List[Tuple[SetCandidate, float]]) -> float:
    if not dist:
        return 0.0
    return float(sum(w for c, w in dist if c.is_physical))

def build_move_pool(belief: OpponentBelief, gen: int) -> Dict[str, Any]:
    """
    Pre-create Move objects for all candidate moves across all roles.
    Returns: dict of move_id -> Move object
    """
    from poke_env.battle import Move

    all_move_ids: Set[str] = set()
    for cand, _ in belief.dist:
        all_move_ids.update(cand.moves)

    pool: Dict[str, Any] = {}
    for mid in all_move_ids:
        try:
            pool[mid] = Move(mid, gen=gen)
        except Exception:
            continue
    return pool


def determinize_opponent(belief: OpponentBelief, rng: random.Random) -> DeterminizedOpponent:
    """
    Samples:
      1) a role candidate from belief
      2) a 4-move subset consistent with revealed moves
    """
    cand = belief.sample_role(rng)

    revealed = set(belief.revealed_moves)
    legal_pool = list(cand.moves - revealed)

    # If revealed already > 4 (rare in practice due to PP/transform weirdness), trim deterministically
    if len(revealed) > 4:
        revealed = set(sorted(revealed)[:4])

    need = max(0, 4 - len(revealed))
    if need > 0:
        if len(legal_pool) < need:
            # if role lists fewer than 4 moves (can happen), just use all
            chosen = legal_pool
        else:
            chosen = rng.sample(legal_pool, k=need)
        sampled = list(revealed) + chosen
    else:
        sampled = list(revealed)

    # pad if still < 4 (extreme edge cases)
    while len(sampled) < 4:
        sampled.append(sampled[-1] if sampled else "struggle")

    sampled4 = tuple(sorted(sampled)[:4])
    return DeterminizedOpponent(candidate=cand, moves4=sampled4)


def sample_unseen_mon(
    gen: int,
    known_species_ids: Set[str],
    rng: random.Random,
) -> Any:
    """
    Sample a Pokémon from the RandBats prior, excluding known species.
    Returns a Pokémon-like proxy suitable for damage calc and heuristic scoring.
    """
    data = _load_randbats_json(gen)
    if not data:
        return None

    known_norm = {to_id_str(s) for s in known_species_ids if s}
    available = [k for k in data.keys() if k and to_id_str(k) not in known_norm]
    if not available:
        return None

    species_key = rng.choice(available)
    candidates = lookup_randbats_candidates(species_key, gen)
    if not candidates:
        return None

    cand = rng.choice(candidates)
    return _create_pokemon_proxy(cand, species_key, gen, rng)


def _create_pokemon_proxy(cand: SetCandidate, species_key: str, gen: int, rng: random.Random) -> Any:
    """Create a lightweight Pokémon proxy from a SetCandidate."""
    from types import SimpleNamespace
    from poke_env.battle import Move, PokemonType
    from poke_env.data import GenData

    data = _load_randbats_json(gen)
    try:
        gd = GenData.from_gen(gen)
    except Exception:
        return None

    # Types and base stats from GenData pokedex
    sid = to_id_str(species_key)
    entry = (gd.pokedex or {}).get(species_key) or (gd.pokedex or {}).get(sid) or {}
    if isinstance(entry, dict):
        type_list = entry.get("types", ["Normal"])
        try:
            t1 = PokemonType.from_name(str(type_list[0])) if type_list else PokemonType.NORMAL
            t2 = PokemonType.from_name(str(type_list[1])) if len(type_list) > 1 else None
        except Exception:
            t1, t2 = PokemonType.NORMAL, None
        base_stats_raw = entry.get("baseStats", {})
        base_stats = {
            "hp": int(base_stats_raw.get("hp", 100)),
            "atk": int(base_stats_raw.get("atk", 100)),
            "def": int(base_stats_raw.get("def", 100)),
            "spa": int(base_stats_raw.get("spa", 100)),
            "spd": int(base_stats_raw.get("spd", 100)),
            "spe": int(base_stats_raw.get("spe", 100)),
        }
    else:
        t1, t2 = PokemonType.NORMAL, None
        base_stats = {"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}

    level = 50
    try:
        entry = (data or {}).get(species_key, {})
        level = int(entry.get("level", 50))
    except Exception:
        pass

    # Stat calculation (simple: base * 2 + 5 at level 50, hp uses different formula)
    def _stat(v: int, is_hp: bool = False) -> int:
        if is_hp:
            return int(2 * v * level / 100 + level + 10)
        return int((2 * v * level / 100 + 5))

    stats = {
        "hp": _stat(base_stats["hp"], True),
        "atk": _stat(base_stats["atk"]),
        "def": _stat(base_stats["def"]),
        "spa": _stat(base_stats["spa"]),
        "spd": _stat(base_stats["spd"]),
        "spe": _stat(base_stats["spe"]),
    }

    # Item and ability from candidate
    item = list(cand.items)[0] if cand.items else None
    ability = list(cand.abilities)[0] if cand.abilities else None

    # Moves: sample 4 from candidate
    move_list = list(cand.moves)
    if len(move_list) >= 4:
        chosen = rng.sample(move_list, 4)
    else:
        chosen = list(move_list)
        while len(chosen) < 4 and move_list:
            chosen.append(rng.choice(move_list))

    moves_dict: Dict[str, Any] = {}
    for mid in chosen:
        try:
            moves_dict[mid] = Move(mid, gen=gen)
        except Exception:
            continue

    mon = SimpleNamespace()
    mon.species = species_key
    mon.base_species = species_key
    mon.stats = stats
    mon.base_stats = base_stats
    mon.boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0}
    mon.current_hp_fraction = 1.0
    mon.max_hp = stats["hp"]
    mon.current_hp = stats["hp"]
    mon.level = level
    mon.type_1 = t1
    mon.type_2 = t2
    mon.types = [t1] + ([t2] if t2 else [])
    mon.original_types = mon.types.copy()
    mon.ability = ability
    mon.item = to_id_str(item) if item else None
    mon.status = None
    mon.effects = {}
    mon.weight = 100.0
    mon.is_terastallized = False
    mon.tera_type = None
    mon.gender = None
    mon.fainted = False
    mon.moves = moves_dict
    mon._identifier_string = f"p2: {species_key}"
    mon.identifier = lambda role=None: mon._identifier_string
    mon._data = type("_Data", (), {"pokedex": {species_key: {"evos": []}}})()

    return mon