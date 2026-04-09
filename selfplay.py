import asyncio
import argparse
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from poke_env import AccountConfiguration, LocalhostServerConfiguration, ServerConfiguration
from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer

from bot.player import AdvancedHeuristicPlayer


BASELINES = {
    "random":    RandomPlayer,
    "maxpower":  MaxBasePowerPlayer,
    "heuristic": SimpleHeuristicsPlayer,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the bot against itself or a baseline to collect training data."
    )
    p.add_argument("--n", type=int, default=10,
                   help="Number of battles to play (default: 10)")
    p.add_argument("--opponent", default="random",
                   choices=[*BASELINES, "clone"],
                   help="Opponent type: random / maxpower / heuristic / clone (default: random)")
    p.add_argument("--output", default="data/turns.jsonl",
                   help="Output JSONL path for bot's turn data (default: data/turns.jsonl)")
    p.add_argument("--port", type=int, default=8000,
                   help="Local Showdown server port (default: 8000)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-turn output from the main bot")
    p.add_argument("--no-policy", action="store_true",
                   help="Disable learned policy model (use heuristic priors only)")
    p.add_argument("--policy-alpha", type=float, default=None,
                   help="Override policy mixing weight (default: MCTSConfig.policy_alpha = 0.3)")
    p.add_argument("--timeout", type=int, default=300,
                   help="Max seconds per battle before it is abandoned (default: 300)")
    p.add_argument("--no-opp-mcts", action="store_true",
                   help="Disable MCTS on the clone opponent (faster, heuristic-only moves)")
    return p.parse_args()


async def main():
    args = parse_args()

    if args.port == 8000:
        server_cfg = LocalhostServerConfiguration
    else:
        server_cfg = ServerConfiguration(
            f"ws://localhost:{args.port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        )

    bot = AdvancedHeuristicPlayer(
        account_configuration=AccountConfiguration.generate("Bot", rand=True),
        battle_format="gen9randombattle",
        server_configuration=server_cfg,
    )
    bot.verbose = not args.quiet
    bot._collector._path = args.output
    if args.no_policy:
        bot._policy = None
        print("[selfplay] Policy model disabled (--no-policy)")
    if args.policy_alpha is not None:
        bot.policy_alpha = args.policy_alpha
        print(f"[selfplay] policy_alpha = {args.policy_alpha}")

    if args.opponent == "clone":
        opp = AdvancedHeuristicPlayer(
            account_configuration=AccountConfiguration.generate("Opp", rand=True),
            battle_format="gen9randombattle",
            server_configuration=server_cfg,
        )
        opp.verbose = False
        if args.no_opp_mcts:
            opp.use_mcts = False
            print("[selfplay] Opponent MCTS disabled (--no-opp-mcts)")
        # Write opponent-side data to a sibling file so they don't interleave
        base, ext = args.output.rsplit(".", 1)
        opp._collector._path = f"{base}_opp.{ext}"
    else:
        opp = BASELINES[args.opponent](
            battle_format="gen9randombattle",
            server_configuration=server_cfg,
        )

    print(
        f"[selfplay] {args.n} battle(s) vs {args.opponent} | "
        f"output → {args.output} | timeout={args.timeout}s"
    )

    timeouts = 0
    # Run one battle at a time so each challenge is sent only after
    # the previous battle fully completes (avoids duplicate-challenge warnings).
    for i in range(args.n):
        try:
            await asyncio.wait_for(
                bot.battle_against(opp, n_battles=1),
                timeout=args.timeout,
            )
        except asyncio.TimeoutError:
            timeouts += 1
            print(
                f"[selfplay] Battle {i+1} timed out after {args.timeout}s "
                f"(total timeouts: {timeouts}) — skipping"
            )
            # 1. Cancel any pending outgoing challenge (prevents "already challenging" errors)
            try:
                await bot.ps_client.send_message(f"/cancelchallenge {opp.username}", "")
            except Exception:
                pass
            # 2. Forfeit any active battles so the server closes those rooms
            for player in (bot, opp):
                for battle in list(player._battles.values()):
                    if not battle.finished:
                        try:
                            await player.ps_client.send_message("/forfeit", battle.battle_tag)
                        except Exception:
                            pass
            await asyncio.sleep(1)
            # 3. Clear Python-side state so next battle_against starts fresh
            bot._battles.clear()
            opp._battles.clear()
            await asyncio.sleep(2)

    print(
        f"\n[selfplay] Done — {bot.n_won_battles}W / {bot.n_lost_battles}L "
        f"({bot.n_won_battles + bot.n_lost_battles} battles, {timeouts} timeout(s))"
    )


if __name__ == "__main__":
    asyncio.run(main())
