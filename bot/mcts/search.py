from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import random

from bot.mcts.shadow_state import ShadowState, Action
from bot.mcts.eval import evaluate_state, evaluate_state_terms


@dataclass
class MCTSConfig:
    num_simulations: int = 300
    max_depth: int = 4
    c_puct: float = 1.6

    # Exploration noise at root (optional)
    dirichlet_alpha: float = 0.0
    dirichlet_eps: float = 0.0

    # Action selection at root
    temperature: float = 0.0  # 0 => argmax visits

    # Reproducibility
    seed: Optional[int] = None

    # stochastic knobs (wired into ShadowState)
    model_miss: bool = True
    model_crit: bool = True
    crit_chance: float = 1.0 / 24.0
    crit_multiplier: float = 1.5
    
    # Hybrid expansion (explicit branching for critical moves)
    use_hybrid_expansion: bool = False  # Start disabled
    branch_low_accuracy: bool = True
    low_accuracy_threshold: float = 0.85 # Threshold for considering miss branching
    branch_potential_ohko: bool = True
    ohko_threshold: float = 0.80
    branch_crit_matters: bool = True
    crit_matters_min_normal: float = 0.60
    min_branch_probability: float = 0.01 # Probability before we consider a branch

    # Switch exploration
    switch_prior_boost: float = 1.5
    prior_ceiling: float = 0.3  # Cap max prior per action; 1.0 = no cap

    # Learned-prior mixing weight at root: 0 = heuristic only, 1 = model only
    policy_alpha: float = 0.3

    # Learned value model for leaf evaluation (replaces evaluate_state when set)
    value_model: Optional[Any] = None


class Node:
    __slots__ = ("state", "parent", "prior", "children", "N", "W", "base_priors")

    def __init__(self, state: ShadowState, parent: Optional["Node"] = None, prior: float = 1.0):
        self.state: ShadowState = state
        self.parent: Optional["Node"] = parent
        self.prior: float = float(prior)  # P(a|s) from heuristics
        self.children: Dict[Action, "Node"] = {}
        self.N: int = 0
        self.W: float = 0.0  # total value backed up
        # For hybrid (branched) actions: base_action -> base prior before splitting.
        # Branch priors are base_prior * prob and may not sum to base_prior if some
        # branches are filtered by min_branch_probability; this stores the true value.
        self.base_priors: Dict = {}

    @property
    def Q(self) -> float:
        return 0.0 if self.N == 0 else (self.W / self.N)

    def is_expanded(self) -> bool:
        return len(self.children) > 0

def softmax(scores: List[float], tau: float = 12.0) -> List[float]:
    if not scores:
        return []
    tau = max(float(tau), 1e-9)
    m = max(scores)
    exps = [math.exp((s - m) / tau) for s in scores]
    z = sum(exps)
    if z <= 0.0 or not math.isfinite(z):
        return [1.0 / len(scores)] * len(scores)
    return [e / z for e in exps]


def action_priors(
    state: ShadowState,
    actions: List[Action],
    switch_prior_boost: float = 1.0,
    prior_ceiling: float = 1.0,
    eval_terms: Optional[Dict[str, float]] = None,
    policy_model: Optional[Any] = None,
    policy_alpha: float = 0.3,
    battle: Optional[Any] = None,
    ctx_me: Optional[Any] = None,
) -> Dict[Action, float]:
    """
    Convert heuristic scores into priors P(a|s) for PUCT.
    switch_prior_boost: multiply switch priors by this, then renormalize (1.0 = no boost).
    prior_ceiling: cap max prior per action; redistribute excess (1.0 = no cap).
    eval_terms + policy_model: if both provided (root only), α-mix heuristic and learned priors.
    policy_alpha: mixing weight (0 = heuristic only, 1 = learned only).
    """
    # Always compute heuristic scores
    heuristic_scores: List[float] = []
    with state._patched_status(), state._patched_boosts():
        for kind, obj in actions:
            if kind == "move":
                heuristic_scores.append(float(state.score_move_fn(obj, state.battle, state.ctx_me)))
            else:
                heuristic_scores.append(float(state.score_switch_fn(obj, state.battle, state.ctx_me)))

    heuristic_probs = softmax(heuristic_scores, tau=12.0)

    using_policy = policy_model is not None and eval_terms is not None
    if using_policy:
        # Compute learned logits with action-specific features + heuristic scores
        base_actions = [(a[0], a[1]) for a in actions]
        try:
            learned_logits = policy_model.score_actions(
                eval_terms, base_actions, battle=battle, ctx_me=ctx_me,
                heuristic_scores=heuristic_scores,
            )
        except Exception:
            learned_logits = [0.0] * len(actions)
        learned_probs = softmax(learned_logits, tau=1.0)

        # α-mix: blend heuristic and learned distributions
        alpha = float(policy_alpha)
        mixed = [(1.0 - alpha) * h + alpha * l
                 for h, l in zip(heuristic_probs, learned_probs)]
        total = sum(mixed) or 1.0
        probs = [m / total for m in mixed]
    else:
        probs = heuristic_probs

    priors: Dict[Action, float] = {}
    for a, p in zip(actions, probs):
        kind = a[0]
        if kind == "switch" or kind == "switch_unknown":
            if not using_policy:
                priors[a] = float(p) * switch_prior_boost
            else:
                priors[a] = float(p)
        else:
            priors[a] = float(p)
    total = sum(priors.values())
    if total > 0:
        priors = {a: p / total for a, p in priors.items()}

    # Prior ceiling: cap dominant actions so others get more exploration
    if prior_ceiling < 1.0 and priors:
        capped = {a: min(p, prior_ceiling) for a, p in priors.items()}
        total_capped = sum(capped.values())
        excess = 1.0 - total_capped
        if excess > 1e-9:
            recipients = [a for a in priors if capped[a] < prior_ceiling]
            if recipients:
                headroom = {a: prior_ceiling - capped[a] for a in recipients}
                total_headroom = sum(headroom.values())
                if total_headroom > 1e-9:
                    for a in recipients:
                        capped[a] += excess * (headroom[a] / total_headroom)
                else:
                    u = excess / len(recipients)
                    for a in recipients:
                        capped[a] += u
            else:
                # All at ceiling; renormalize to sum to 1
                s = sum(capped.values()) or 1.0
                capped = {a: p / s for a, p in capped.items()}
            priors = capped
    return priors


def add_dirichlet_noise(priors: Dict[Action, float], alpha: float, eps: float, rng: random.Random) -> Dict[Action, float]:
    if eps <= 0.0 or alpha <= 0.0 or len(priors) <= 1:
        return priors

    actions = list(priors.keys())
    noise_raw = [rng.gammavariate(alpha, 1.0) for _ in actions]
    z = sum(noise_raw) or 1.0
    noise = [x / z for x in noise_raw]

    out: Dict[Action, float] = {}
    for a, n in zip(actions, noise):
        out[a] = (1.0 - eps) * priors[a] + eps * n

    s = sum(out.values()) or 1.0
    for a in out:
        out[a] /= s
    return out

def select_child(node: Node, c_puct: float) -> Tuple[Action, Node]:
    """
    Pick child that maximizes:
      Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
    
    For hybrid actions (multiple outcome branches), uses expected Q across all branches.
    """
    sqrt_N = math.sqrt(max(1, node.N))
    best_a: Optional[Action] = None
    best_u = -1e18
    best_child: Optional[Node] = None
    
    # Group children by base action (for hybrid branches)
    base_action_groups = {}
    for a, child in node.children.items():
        # Base action is (kind, obj) without outcome label
        base_action = (a[0], a[1]) if len(a) >= 2 else a
        if base_action not in base_action_groups:
            base_action_groups[base_action] = []
        base_action_groups[base_action].append((a, child))
    
    for base_action, action_children in base_action_groups.items():
        if len(action_children) == 1:
            # Single outcome (not hybrid) - use normal PUCT
            a, child = action_children[0]
            if child.N == 0:
                u = float('inf')
            else:
                q = child.Q
                u = q + c_puct * child.prior * (sqrt_N / (1 + child.N))
            
            if u > best_u:
                best_u = u
                best_a = a
                best_child = child
        else:
            # Multiple outcomes (hybrid) - use EXPECTED Q across branches
            # total_prior (from base_priors) is used for the exploration scale.
            # sum_branch_prior is used for weighting — keeps the two concepts separate.
            total_prior = node.base_priors.get(
                base_action, sum(child.prior for _, child in action_children)
            )
            sum_branch_prior = sum(child.prior for _, child in action_children)

            if total_prior > 0:
                # Expected Q = sum(weight * Q) for each visited outcome branch
                expected_q = 0.0
                total_n = 0

                for a, child in action_children:
                    if child.N > 0:
                        weight = child.prior / sum_branch_prior if sum_branch_prior > 0 else 1.0 / len(action_children)
                        expected_q += weight * child.Q
                        total_n += child.N

                # Use total visits across all branches for exploration term
                if total_n > 0:
                    u = expected_q + c_puct * total_prior * (sqrt_N / (1 + total_n))
                else:
                    u = float('inf')  # Unexplored hybrid action
                
                if u > best_u:
                    best_u = u
                    # Select branch proportionally to probability
                    # Pick the branch that is most under-visited relative
                    # to its probability share
                    best_deficit = -1e18
                    for a, child in action_children:
                        target_frac = child.prior / sum_branch_prior if sum_branch_prior > 0 else 1.0 / len(action_children)
                        actual_frac = child.N / total_n if total_n > 0 else 0.0
                        deficit = target_frac - actual_frac
                        if deficit > best_deficit:
                            best_deficit = deficit
                            best_a = a
                            best_child = child

    return best_a, best_child

def should_branch_move(move: Any, state: ShadowState, cfg: MCTSConfig) -> bool:
    """
    Decide whether a move should create multiple outcome branches.
    Returns True for "critical" moves where we want explicit branching.
    """
    if move is None:
        return False
    
    # Get move properties
    accuracy = getattr(move, 'accuracy', 1.0) or 1.0
    if accuracy > 1.0:
        accuracy /= 100.0
    accuracy = max(0.0, min(1.0, accuracy))
    
    # Check 1: Low accuracy moves
    if cfg.branch_low_accuracy and accuracy < cfg.low_accuracy_threshold:
        return True
    
    # Check 2: Potential OHKO scenarios
    if cfg.branch_potential_ohko and accuracy < 1.0:
        try:
            with state._patched_status(), state._patched_boosts():
                dmg_frac = float(state.dmg_fn(move, state.my_active, state.opp_active, state.battle))
            
            opp_hp = state.opp_active_hp()
            if dmg_frac >= opp_hp * cfg.ohko_threshold:
                return True
        except Exception:
            pass
    
    # Check 3: Crit matters (normal hit doesn't KO but crit does)
    if cfg.branch_crit_matters:
        try:
            with state._patched_status(), state._patched_boosts():
                dmg_frac = float(state.dmg_fn(move, state.my_active, state.opp_active, state.battle))
            
            opp_hp = state.opp_active_hp()
            crit_dmg = dmg_frac * state.crit_multiplier
            
            if dmg_frac >= opp_hp * cfg.crit_matters_min_normal and crit_dmg >= opp_hp and dmg_frac < opp_hp:
                return True
        except Exception:
            pass
    
    return False


def _crit_changes_ko(state: ShadowState, move: Any) -> bool:
    """Check if a crit changes the KO outcome (normal doesn't KO but crit does, or vice versa)."""
    try:
        with state._patched_status(), state._patched_boosts():
            dmg_frac = float(state.dmg_fn(move, state.my_active, state.opp_active, state.battle))
        opp_hp = state.opp_active_hp()
        crit_dmg = dmg_frac * state.crit_multiplier
        normal_kos = dmg_frac >= opp_hp
        crit_kos = crit_dmg >= opp_hp
        return normal_kos != crit_kos
    except Exception:
        return True  # If we can't tell, assume it matters


def create_move_branches(
    state: ShadowState,
    move: Any,
    cfg: MCTSConfig,
    base_rng: random.Random,
) -> List[Tuple[float, ShadowState, str]]:
    """
    Create explicit branches for a move's possible outcomes.
    Returns: List of (probability, next_state, outcome_label) tuples

    Optimization: merges hit+crit into hit when the crit doesn't change
    the KO outcome (both KO or both don't KO).
    """
    accuracy = getattr(move, 'accuracy', 1.0) or 1.0
    if accuracy > 1.0:
        accuracy /= 100.0
    accuracy = max(0.0, min(1.0, accuracy))

    crit_chance = state.crit_chance

    branches = []

    # Calculate probabilities
    p_miss = 1.0 - accuracy
    p_hit_crit = accuracy * crit_chance
    p_hit_no_crit = accuracy * (1.0 - crit_chance)

    # Check if crit actually changes the KO outcome
    crit_matters = _crit_changes_ko(state, move)

    if crit_matters:
        # Separate branches for crit and non-crit
        if p_hit_crit >= cfg.min_branch_probability:
            rng_crit = random.Random(hash((id(base_rng), "hit_crit")))
            state_temp = state.with_forced_outcome(hit=True, crit=True)
            next_state = state_temp.step(("move", move), rng=rng_crit)
            branches.append((p_hit_crit, next_state, "hit+crit"))

        if p_hit_no_crit >= cfg.min_branch_probability:
            rng_hit = random.Random(hash((id(base_rng), "hit_no_crit")))
            state_temp = state.with_forced_outcome(hit=True, crit=False)
            next_state = state_temp.step(("move", move), rng=rng_hit)
            branches.append((p_hit_no_crit, next_state, "hit"))
    else:
        # Crit doesn't change outcome — merge into single "hit" branch
        p_hit = p_hit_crit + p_hit_no_crit
        if p_hit >= cfg.min_branch_probability:
            rng_hit = random.Random(hash((id(base_rng), "hit_no_crit")))
            state_temp = state.with_forced_outcome(hit=True, crit=False)
            next_state = state_temp.step(("move", move), rng=rng_hit)
            branches.append((p_hit, next_state, "hit"))

    # Miss branch
    if p_miss >= cfg.min_branch_probability:
        rng_miss = random.Random(hash((id(base_rng), "miss")))
        state_temp = state.with_forced_outcome(hit=False, crit=False)
        next_state = state_temp.step(("move", move), rng=rng_miss)
        branches.append((p_miss, next_state, "miss"))

    # Renormalize so kept-branch probs sum to 1.0.
    # Pruning by min_branch_probability can leave a probability mass < 1, which
    # would make branch_prior * prob < base_prior and distort PUCT + expected-Q.
    if branches:
        total_p = sum(prob for prob, _, _ in branches)
        if total_p > 0.0 and abs(total_p - 1.0) > 1e-9:
            branches = [(prob / total_p, s, lbl) for prob, s, lbl in branches]

    return branches


def expand(
    node: Node,
    rng: random.Random,
    cfg: MCTSConfig,
    is_root: bool,
    eval_terms: Optional[Dict[str, float]] = None,
    policy_model: Optional[Any] = None,
    battle: Optional[Any] = None,
    ctx_me: Optional[Any] = None,
) -> None:
    """
    Expansion with hybrid approach:
    - Critical moves create multiple outcome branches
    - Routine moves use single-sample expansion
    eval_terms + policy_model: passed through only at the root node.
    """
    if node.state.is_terminal():
        return

    actions = node.state.legal_actions()
    if not actions:
        return

    priors = action_priors(
        node.state, actions,
        switch_prior_boost=cfg.switch_prior_boost,
        prior_ceiling=cfg.prior_ceiling,
        eval_terms=eval_terms if (is_root and eval_terms) else None,
        policy_model=policy_model if (is_root and eval_terms) else None,
        policy_alpha=cfg.policy_alpha,
        battle=battle if is_root else None,
        ctx_me=ctx_me if is_root else None,
    )
    if is_root:
        priors = add_dirichlet_noise(priors, cfg.dirichlet_alpha, cfg.dirichlet_eps, rng)
    
    for a in actions:
        # Check if already expanded (handles both branched and non-branched)
        if a in node.children:
            continue
        
        # For branched actions, check if any branch exists
        base_action = (a[0], a[1]) if len(a) >= 2 else a
        existing_branches = [k for k in node.children.keys() 
                            if len(k) >= 2 and (k[0], k[1]) == (base_action[0], base_action[1])]
        if existing_branches:
            continue
        
        kind, obj = a
        
        # Decide: should we branch this move?
        if cfg.use_hybrid_expansion and kind == "move" and should_branch_move(obj, node.state, cfg):
            # CRITICAL MOVE: Create multiple outcome branches
            branches = create_move_branches(node.state, obj, cfg, rng)

            base_prior = priors.get(a, 1.0 / len(actions))
            node.base_priors[(kind, obj)] = base_prior  # explicit 2-tuple; rows-building uses same key

            for prob, next_state, outcome_label in branches:
                # Create unique action key: (kind, obj, outcome_label)
                branch_action = (kind, obj, outcome_label)
                
                # Weight prior by probability
                branch_prior = base_prior * prob
                
                node.children[branch_action] = Node(
                    next_state,
                    parent=node,
                    prior=branch_prior
                )
        else:
            # ROUTINE MOVE/SWITCH: Standard single-sample expansion
            next_state = node.state.step(a, rng=rng)
            node.children[a] = Node(
                next_state,
                parent=node,
                prior=priors.get(a, 1.0 / len(actions))
            )


def evaluate_leaf(node: Node, cfg: Optional["MCTSConfig"] = None) -> float:
    """
    Leaf evaluation from OUR perspective, bounded ~[-1,1].
    If cfg.value_model is set, use the learned value function instead of the heuristic.
    """
    if cfg is not None and cfg.value_model is not None:
        try:
            _, terms = evaluate_state_terms(node.state)
            if terms:
                return cfg.value_model.predict_value(terms)
        except Exception:
            pass  # fall through to heuristic
    return float(evaluate_state(node.state))


def get_action_expected_q(node: Node, base_action: Tuple) -> Tuple[float, int]:
    """
    For hybrid actions with multiple outcome branches, compute expected Q.
    Returns (expected_q, total_visits)
    """
    # Find all children that match this base action
    matching_children = []
    for a, child in node.children.items():
        child_base = (a[0], a[1]) if len(a) >= 2 else a
        if child_base == base_action:
            matching_children.append(child)
    
    if not matching_children:
        return 0.0, 0
    
    if len(matching_children) == 1:
        # Single outcome - just return its Q
        child = matching_children[0]
        return child.Q, child.N
    
    # Multiple outcomes - compute expected value weighted by branch priors.
    sum_branch_prior = sum(child.prior for child in matching_children)
    total_visits = sum(child.N for child in matching_children)

    if sum_branch_prior <= 0 or total_visits == 0:
        return 0.0, total_visits

    expected_q = sum(
        (child.prior / sum_branch_prior) * child.Q
        for child in matching_children
        if child.N > 0
    )
    
    return expected_q, total_visits


def backup(node: Node, value: float) -> None:
    cur: Optional[Node] = node
    while cur is not None:
        cur.N += 1
        cur.W += value
        cur = cur.parent


def action_name(a: Action) -> Tuple[str, str]:
    kind = a[0]
    obj = a[1]
    outcome = a[2] if len(a) > 2 else None
    
    if kind == "move":
        name = str(getattr(obj, "id", "") or getattr(obj, "name", "") or "move")
        if outcome:
            name += f" [{outcome}]"
        return "move", name

    if kind == "switch_unknown":
        return "switch", f"??{obj}" if isinstance(obj, int) else "??unknown"

    name = str(getattr(obj, "species", "") or getattr(obj, "name", "") or "switch")
    return "switch", name


def search(
    *,
    battle: Any,
    ctx_me: Any,
    ctx_opp: Any,
    score_move_fn,
    score_switch_fn,
    dmg_fn,
    cfg: MCTSConfig,
    opp_tau: float = 8.0,
    status_threshold: float = 0.30,
    allow_switches: bool = True,
    return_stats: bool = False,
    return_tree: bool = False,
    opp_beliefs: Optional[Dict[int, Any]] = None,
    opp_move_pools: Optional[Dict[int, Dict[str, Any]]] = None,
    policy_model: Optional[Any] = None,
) -> Union[Action, Tuple[Action, Dict[str, Any]]]:
    """
    Runs PUCT MCTS and returns best root action.
    If return_stats=True, also returns a dict with root child visits/Q/prior.

    allow_switches=False => only expands root with moves (still allows pivot-triggered switches inside rollouts).
    policy_model: if provided, used to compute learned priors at the root node.
    """
    master_rng = random.Random(cfg.seed)

    root_state = ShadowState.from_battle(
        battle=battle,
        ctx_me=ctx_me,
        ctx_opp=ctx_opp,
        score_move_fn=score_move_fn,
        score_switch_fn=score_switch_fn,
        dmg_fn=dmg_fn,
        opp_tau=opp_tau,
        status_threshold=status_threshold,
        model_miss=cfg.model_miss,
        model_crit=cfg.model_crit,
        crit_chance=cfg.crit_chance,
        crit_multiplier=cfg.crit_multiplier,
        debug=False,
        opp_beliefs=opp_beliefs,
        opp_move_pools=opp_move_pools,
    )
    root = Node(root_state, parent=None, prior=1.0)
    root_n_legal = len(root_state.legal_actions())

    # Always compute eval_terms at root (used for data collection and learned prior)
    root_eval_terms: Optional[Dict[str, float]] = None
    root_eval_value: float = 0.0
    try:
        root_eval_value, root_eval_terms = evaluate_state_terms(root_state)
        # Force-switch fallback: active is fainted → early return gives no terms.
        # Proxy with the healthiest bench pokemon so state features are meaningful.
        if not root_eval_terms:
            best_bench = max(
                (p for p in root_state.my_team if p is not root_state.my_active),
                key=lambda p: root_state.my_hp.get(id(p), 0.0),
                default=None,
            )
            if best_bench is not None:
                proxy = _dc_replace(root_state, my_active=best_bench)
                root_eval_value, root_eval_terms = evaluate_state_terms(proxy)
    except Exception:
        root_eval_terms = None

    # Expand root once (using master_rng so it's reproducible)
    expand(root, master_rng, cfg, is_root=True,
           eval_terms=root_eval_terms, policy_model=policy_model,
           battle=battle, ctx_me=ctx_me)

    if not allow_switches and root.children:
        root.children = {a: n for a, n in root.children.items() if a[0] == "move"}
        if not root.children:
            actions = root_state.legal_actions()
            moves = [a for a in actions if a[0] == "move"]
            if moves:
                picked = moves[0]
                return (picked, {"top": [], "sims": 0}) if return_stats else picked

    if not root.children:
        return (None, {"top": [], "sims": 0}) if return_stats else None

    # Simulations
    for sim_i in range(int(cfg.num_simulations)):
        # Each rollout gets its own deterministic RNG stream (reproducible but different)
        sim_seed = hash((cfg.seed, sim_i, "rollout")) if cfg.seed is not None else master_rng.getrandbits(64)
        sim_rng = random.Random(sim_seed)

        node = root
        depth = 0

        # Selection
        while node.is_expanded() and not node.state.is_terminal() and depth < cfg.max_depth:
            _, node = select_child(node, cfg.c_puct)
            depth += 1

        # Expansion (sample transitions with sim_rng the first time we expand this node)
        if not node.state.is_terminal() and depth < cfg.max_depth and not node.is_expanded():
            expand(node, sim_rng, cfg, is_root=False)

        # Evaluation
        value = evaluate_leaf(node, cfg)

        # Backup
        backup(node, value)

    # Build root stats
    # Group actions by base action to handle hybrid branches
    base_action_map = {}
    for a in list(root.children.keys()):
        base_action = (a[0], a[1]) if len(a) >= 2 else a
        if base_action not in base_action_map:
            base_action_map[base_action] = []
        base_action_map[base_action].append(a)
    
    rows: List[Dict[str, Any]] = []
    for base_action, action_list in base_action_map.items():
        if len(action_list) == 1:
            a = action_list[0]
            child = root.children[a]
            kind, name = action_name(a)
            rows.append({
                "kind":      kind,
                "name":      name,
                "visits":    int(child.N),
                "q":         float(child.Q),
                "prior":     float(child.prior),
                "is_hybrid": False,
                "action":    a,
            })
        else:
            # Multiple outcomes (hybrid) - compute expected Q across branches
            expected_q, total_visits = get_action_expected_q(root, base_action)
            # Use the stashed base_prior (pre-split); fall back to branch sum if unavailable
            prior = root.base_priors.get(base_action,
                        sum(root.children[a].prior for a in action_list))
            kind = base_action[0]
            obj = base_action[1]
            if kind == "move":
                name = str(getattr(obj, "id", "") or getattr(obj, "name", "") or "move")
            else:
                name = str(getattr(obj, "species", "") or getattr(obj, "name", "") or "switch")
            best_branch = max(action_list, key=lambda a: root.children[a].N)
            rows.append({
                "kind":      kind,
                "name":      name,
                "visits":    int(total_visits),
                "q":         float(expected_q),
                "prior":     float(prior),
                "is_hybrid": True,
                "action":    best_branch,
            })
    
    rows.sort(key=lambda d: (d["q"], d["visits"]), reverse=True)

    # Compute raw (pre-ceiling) priors for root actions — only for logging.
    # Call action_priors again with prior_ceiling=1.0 so we can compare to
    # what the heuristic/model actually wanted vs what the cap allowed.
    root_base_actions = list(base_action_map.keys())
    if root_base_actions:
        try:
            raw_prior_map = action_priors(
                root_state, root_base_actions,
                switch_prior_boost=cfg.switch_prior_boost,
                prior_ceiling=1.0,  # no cap
                eval_terms=root_eval_terms if root_eval_terms else None,
                policy_model=policy_model,
                policy_alpha=cfg.policy_alpha,
                battle=root_state.battle,
                ctx_me=root_state.ctx_me,
            )
        except Exception:
            raw_prior_map = {}
    else:
        raw_prior_map = {}

    for row in rows:
        base_a = (row["action"][0], row["action"][1])
        row["prior_raw"] = round(float(raw_prior_map.get(base_a, row["prior"])), 4)

    # Search diagnostics
    all_child_visits = [root.children[a].N for a in root.children]
    root_visits_total = sum(all_child_visits) or 1
    max_visits = max(all_child_visits) if all_child_visits else 0
    q_values = [row["q"] for row in rows if row["visits"] > 0]
    search_info = {
        "root_visits_total": root_visits_total,
        "max_visits_frac":   round(max_visits / root_visits_total, 4),
        "q_range":           round(max(q_values) - min(q_values), 4) if len(q_values) >= 2 else 0.0,
        "n_actions":         len(base_action_map),
        "n_actions_legal":   root_n_legal,
        "using_policy":      bool(policy_model is not None and root_eval_terms),
    }

    # Pick best action from root children (max-Q policy).
    # Max-Q is preferred over max-N because visit counts can be hijacked by
    # a large heuristic prior even when the search found a better Q elsewhere.
    # Guard: require at least 1 visit so we never pick an unvisited child.
    if cfg.temperature and cfg.temperature > 1e-9:
        # Temperature-based selection (still over visits — exploration phase)
        base_action_visits = {}
        for base_action, action_list in base_action_map.items():
            total_visits = sum(root.children[a].N for a in action_list)
            base_action_visits[base_action] = total_visits

        actions_for_temp = list(base_action_visits.keys())
        visits = list(base_action_visits.values())
        m = max(visits)
        weights = [math.exp((v - m) / float(cfg.temperature)) for v in visits]
        z = sum(weights) or 1.0
        r = master_rng.random() * z
        acc = 0.0
        picked_base = actions_for_temp[-1]
        for ba, w in zip(actions_for_temp, weights):
            acc += w
            if acc >= r:
                picked_base = ba
                break

        # Within the chosen base action, pick highest-Q visited branch
        visited = [a for a in base_action_map[picked_base] if root.children[a].N > 0]
        pool = visited if visited else base_action_map[picked_base]
        picked = max(pool, key=lambda a: root.children[a].Q)
    else:
        # Argmax expected-Q across all base actions (visited children only)
        def _base_q(ba):
            q, n = get_action_expected_q(root, ba)
            return q if n > 0 else -1e9

        best_base_action = max(base_action_map.keys(), key=_base_q)
        # Within the best base action, pick highest-Q visited branch
        visited = [a for a in base_action_map[best_base_action] if root.children[a].N > 0]
        pool = visited if visited else base_action_map[best_base_action]
        picked = max(pool, key=lambda a: root.children[a].Q)

    payload = {
        "top": rows,
        "sims": int(cfg.num_simulations),
        "root_state": root_state,
        "eval_terms": root_eval_terms or {},
        "eval_value": root_eval_value,
        "search_info": search_info,
    }

    if return_tree:
        payload["root"] = root

    if return_stats:
        return picked, payload
    return picked


def mcts_pick_action(
    *,
    battle: Any,
    ctx: Any,
    ctx_opp: Any,
    score_move_fn,
    score_switch_fn,
    dmg_fn,
    iters: int,
    max_depth: int = 4,
    include_switches: bool = True,
    opp_tau: float = 8.0,
    status_threshold: float = 0.30,
    seed: Optional[int] = None,
    model_miss: bool = True,
    model_crit: bool = True,
    crit_chance: float = 1.0 / 24.0,
    crit_multiplier: float = 1.5,
    opp_beliefs: Optional[Dict[int, Any]] = None,
    opp_move_pools: Optional[Dict[int, Dict[str, Any]]] = None,
    policy_model: Optional[Any] = None,
    policy_alpha: float = 0.3,
    value_model: Optional[Any] = None,
) -> Tuple[Optional[Action], Optional[Dict[str, Any]]]:

    cfg = MCTSConfig(
        num_simulations=int(iters),
        max_depth=int(max_depth),
        c_puct=1.6,
        dirichlet_alpha=0.0,
        dirichlet_eps=0.0,
        temperature=0.0,
        seed=seed,
        model_miss=model_miss,
        model_crit=model_crit,
        crit_chance=crit_chance,
        crit_multiplier=crit_multiplier,
        policy_alpha=float(policy_alpha),
        value_model=value_model,
    )

    picked, stats = search(
        battle=battle,
        ctx_me=ctx,
        ctx_opp=ctx_opp,
        score_move_fn=score_move_fn,
        score_switch_fn=score_switch_fn,
        dmg_fn=dmg_fn,
        cfg=cfg,
        opp_tau=opp_tau,
        status_threshold=status_threshold,
        allow_switches=bool(include_switches),
        return_stats=True,
        opp_beliefs=opp_beliefs,
        opp_move_pools=opp_move_pools,
        policy_model=policy_model,
    )
    return picked, stats


def format_tree(
    root: "Node",
    *,
    max_depth: int = 3,
    top_k: int = 5,
    show_state: bool = True,
) -> str:
    """
    Pretty-print the MCTS tree from root.

    max_depth: how many plies to print
    top_k: print top_k children per node (by visits)
    show_state: include (my_active vs opp_active, hp, status) snapshot
    """
    lines: List[str] = []

    def snap(node: "Node") -> str:
        if not show_state:
            return ""
        s = node.state
        
        try:
            my = getattr(s.my_active, "species", "ME")
            op = getattr(s.opp_active, "species", "OPP")
            my_hp = s.my_hp.get(id(s.my_active), 1.0)
            op_hp = s.opp_hp.get(id(s.opp_active), 1.0)
            my_st = s.my_status.get(id(s.my_active))
            op_st = s.opp_status.get(id(s.opp_active))

            def st(x):
                return "none" if x is None else str(x).split(".")[-1].lower()

            base = f" [{my} {my_hp:.2f} {st(my_st)} vs {op} {op_hp:.2f} {st(op_st)}]"

            if getattr(s, "debug", False) and getattr(s, "events", ()):
                # show last 1–2 events only (prevents clutter)
                ev = "; ".join(s.events[-2:])
                base += f" | {ev}"

            return base
        except Exception:
            return ""

    def action_str(a: Action) -> str:
        kind = a[0]
        obj = a[1]
        outcome = a[2] if len(a) > 2 else None 
        if kind == "move":
            return f"move {getattr(obj, 'id', getattr(obj, 'name', 'move'))}"
        return f"switch {getattr(obj, 'species', getattr(obj, 'name', 'switch'))}"

    def rec(node: "Node", depth: int, prefix: str):
        kids = sorted(
            node.children.items(),
            key=lambda kv: (kv[1].N, kv[1].Q),
            reverse=True,
        )[:top_k]

        for i, (a, child) in enumerate(kids):
            branch = "└─" if i == len(kids) - 1 else "├─"
            lines.append(
                f"{prefix}{branch} {action_str(a):18s}  "
                f"N={child.N:4d}  Q={child.Q:+.3f}  P={child.prior:.3f}"
                f"{snap(child)}"
            )
            if depth + 1 < max_depth and child.children:
                ext = "   " if i == len(kids) - 1 else "│  "
                rec(child, depth + 1, prefix + ext)

    lines.append(f"ROOT N={root.N} Q={root.Q:+.3f}{snap(root)}")
    rec(root, 0, "")
    return "\n".join(lines)