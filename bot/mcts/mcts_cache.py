from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib


@dataclass
class MCTSCache:
    """
    Centralized cache for MCTS computations.
    
    The cache is designed to be:
    - Thread-safe for read operations (immutable keys)
    - Cleared between different battles
    - Shared across all nodes in a single MCTS search
    """
    
    # Role weight cache: pokemon_id -> (role_weight, gen)
    role_weights: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    # Damage calculation cache: (attacker_id, defender_id, move_id, crit) -> damage_fraction
    damage_cache: Dict[Tuple[int, int, str, bool], float] = field(default_factory=dict)
    
    # Action prior cache: (state_hash, action_tuple) -> prior_probability
    prior_cache: Dict[Tuple[str, Tuple], float] = field(default_factory=dict)
    
    # Move property cache: move_id -> properties dict
    move_props: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Pokemon property cache: pokemon_id -> properties dict
    mon_props: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    # Statistics
    hits: int = 0
    misses: int = 0
    
    def clear(self) -> None:
        """Clear all caches. Call this between battles."""
        self.role_weights.clear()
        self.damage_cache.clear()
        self.prior_cache.clear()
        self.move_props.clear()
        self.mon_props.clear()
        self.hits = 0
        self.misses = 0
    
    def get_role_weight(self, pokemon: Any, gen: int, compute_fn) -> float:
        """
        Get cached role weight or compute and cache it.
        
        Args:
            pokemon: The Pokemon object
            gen: Generation number
            compute_fn: Function to compute role weight if not cached
                       Should be: lambda mon, g: expected_role_weight_for_mon(mon, g)
        """
        key = (id(pokemon), gen)
        
        if key in self.role_weights:
            self.hits += 1
            return self.role_weights[key]
        
        self.misses += 1
        weight = float(compute_fn(pokemon, gen))
        self.role_weights[key] = weight
        return weight
    
    def get_damage(
        self,
        attacker: Any,
        defender: Any,
        move: Any,
        is_crit: bool,
        compute_fn,
    ) -> float:
        """
        Get cached damage calculation or compute and cache it.
        
        Args:
            attacker: Attacking Pokemon
            defender: Defending Pokemon
            move: Move object
            is_crit: Whether this is a critical hit
            compute_fn: Function to compute damage
                       Should be: lambda: dmg_fn(move, attacker, defender, battle)
        """
        move_id = str(getattr(move, 'id', '') or getattr(move, 'name', ''))
        key = (id(attacker), id(defender), move_id, is_crit)
        
        if key in self.damage_cache:
            self.hits += 1
            return self.damage_cache[key]
        
        self.misses += 1
        damage = float(compute_fn())
        self.damage_cache[key] = damage
        return damage
    
    def get_move_properties(self, move: Any) -> Dict[str, Any]:
        """
        Get cached move properties or extract and cache them.
        
        Caches: priority, accuracy, base_power, is_pivot, category, type
        """
        move_id = str(getattr(move, 'id', '') or getattr(move, 'name', ''))
        
        if move_id in self.move_props:
            self.hits += 1
            return self.move_props[move_id]
        
        self.misses += 1
        
        # Extract properties once
        props = {
            'priority': int(getattr(move, 'priority', 0) or 0),
            'accuracy': float(getattr(move, 'accuracy', 1.0) or 1.0),
            'base_power': int(getattr(move, 'base_power', 0) or 0),
            'category': getattr(move, 'category', None),
            'type': getattr(move, 'type', None),
            'id': move_id,
        }
        
        # Check if pivot
        mid = move_id.lower().replace(' ', '').replace('-', '')
        props['is_pivot'] = mid in {'voltswitch', 'uturn', 'flipturn', 'partingshot'}
        
        self.move_props[move_id] = props
        return props
    
    def get_mon_properties(self, pokemon: Any) -> Dict[str, Any]:
        """
        Get cached Pokemon properties or extract and cache them.
        
        Caches: species, base_stats, has_removal, has_priority
        """
        key = id(pokemon)
        
        if key in self.mon_props:
            self.hits += 1
            return self.mon_props[key]
        
        self.misses += 1
        
        props = {
            'species': str(getattr(pokemon, 'species', 'unknown')),
            'base_stats': dict(getattr(pokemon, 'base_stats', {}) or {}),
        }
        
        # Check for removal moves
        props['has_removal'] = False
        props['has_priority'] = False
        
        moves = getattr(pokemon, 'moves', {}) or {}
        for move in moves.values():
            if move is None:
                continue
            
            move_id = str(getattr(move, 'id', '') or '').lower().replace(' ', '').replace('-', '')
            
            # Check removal
            if move_id in {'rapidspin', 'defog'}:
                props['has_removal'] = True
            
            # Check damaging priority
            priority = int(getattr(move, 'priority', 0) or 0)
            bp = int(getattr(move, 'base_power', 0) or 0)
            if priority > 0 and bp > 0:
                props['has_priority'] = True
        
        self.mon_props[key] = props
        return props
    
    def warm_up(self, state: Any) -> None:
        """
        Pre-populate cache with common lookups from initial state.
        
        Call this once at the start of MCTS search to avoid cache misses
        on the first simulation.
        
        Args:
            state: Initial ShadowState
        """
        gen = int(getattr(state.battle, 'gen', 9) or 9)
        
        # Pre-cache all Pokemon properties
        all_mons = list(state.my_team) + list(state.opp_team)
        for mon in all_mons:
            if mon is not None:
                self.get_mon_properties(mon)
        
        # Pre-cache move properties from active Pokemon
        for move in (getattr(state.my_active, 'moves', {}) or {}).values():
            if move is not None:
                self.get_move_properties(move)
        
        for move in (getattr(state.opp_active, 'moves', {}) or {}).values():
            if move is not None:
                self.get_move_properties(move)
    
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total,
            'hit_rate_pct': hit_rate,
            'role_weights_cached': len(self.role_weights),
            'damage_calcs_cached': len(self.damage_cache),
            'move_props_cached': len(self.move_props),
            'mon_props_cached': len(self.mon_props),
        }


# Global LRU cache for static computations
@lru_cache(maxsize=1024)
def normalize_move_id(move_name: str) -> str:
    """Normalize move name to lowercase, no spaces/dashes."""
    return str(move_name or '').lower().replace(' ', '').replace('-', '')


@lru_cache(maxsize=256)
def is_pivot_move_cached(move_id: str) -> bool:
    """Check if a move is a pivot move (U-turn, Volt Switch, etc.)."""
    normalized = normalize_move_id(move_id)
    return normalized in {'voltswitch', 'uturn', 'flipturn', 'partingshot'}


@lru_cache(maxsize=256)
def is_removal_move_cached(move_id: str) -> bool:
    """Check if a move removes hazards."""
    normalized = normalize_move_id(move_id)
    return normalized in {'rapidspin', 'defog'}


@lru_cache(maxsize=256)
def is_setup_move_cached(move_id: str) -> bool:
    """Check if a move is a setup move (boosts stats)."""
    normalized = normalize_move_id(move_id)
    setup_moves = {
        'swordsdance', 'nastyplot', 'dragondance', 'calmmind', 'bulkup', 'quiverdance',
        'shellsmash', 'bellydrum', 'shiftgear', 'agility', 'tailglow', 'coil', 'curse', 'growth',
    }
    return normalized in setup_moves


@lru_cache(maxsize=256)
def is_priority_move_cached(move_id: str) -> bool:
    """Check if a move has priority."""
    normalized = normalize_move_id(move_id)
    priority_moves = {
        'extremespeed', 'aquajet', 'machpunch', 'iceshard', 'suckerpunch', 'bulletpunch',
        'shadowsneak', 'quickattack', 'vacuumwave', 'firstimpression',
    }
    return normalized in priority_moves


# ============================================================================
# Integration helpers for existing code
# ============================================================================

def create_cached_role_weight_fn(cache: MCTSCache, base_fn):
    """
    Create a cached version of expected_role_weight_for_mon.
    
    Args:
        cache: MCTSCache instance
        base_fn: Original function(mon, gen) -> float
    
    Returns:
        Cached version of the function
    """
    def cached_fn(mon: Any, gen: int) -> float:
        return cache.get_role_weight(mon, gen, lambda m, g: base_fn(m, g))
    return cached_fn


def create_cached_damage_fn(cache: MCTSCache, base_fn, battle: Any):
    """
    Create a cached version of damage calculation.
    
    Args:
        cache: MCTSCache instance
        base_fn: Original function(move, attacker, defender, battle) -> float
        battle: Battle object
    
    Returns:
        Cached version of the function
    """
    def cached_fn(move: Any, attacker: Any, defender: Any, _battle: Any = None) -> float:
        # Use passed battle or captured battle
        b = _battle or battle
        return cache.get_damage(
            attacker, defender, move, False,
            lambda: base_fn(move, attacker, defender, b)
        )
    return cached_fn


# ============================================================================
# State hashing for prior cache
# ============================================================================

def hash_state_for_priors(state: Any) -> str:
    """
    Create a hash of the state for caching action priors.
    
    Only includes aspects that affect heuristic scoring:
    - Active Pokemon
    - HP values
    - Status conditions
    - Boosts
    """
    try:
        # Create a stable representation
        my_active_id = id(state.my_active)
        opp_active_id = id(state.opp_active)
        
        my_hp = state.my_hp.get(my_active_id, 0.0)
        opp_hp = state.opp_hp.get(opp_active_id, 0.0)
        
        my_status = str(state.my_status.get(my_active_id, None))
        opp_status = str(state.opp_status.get(opp_active_id, None))
        
        my_boosts = str(sorted((state.my_boosts.get(my_active_id, {}) or {}).items()))
        opp_boosts = str(sorted((state.opp_boosts.get(opp_active_id, {}) or {}).items()))
        
        # Create hash
        key_str = f"{my_active_id}:{opp_active_id}:{my_hp:.2f}:{opp_hp:.2f}:{my_status}:{opp_status}:{my_boosts}:{opp_boosts}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    except Exception:
        # Fallback: use ply as a weak hash
        return f"ply_{getattr(state, 'ply', 0)}"


if __name__ == "__main__":
    # Example usage and testing
    print("MCTS Cache System")
    print("=" * 60)
    
    cache = MCTSCache()