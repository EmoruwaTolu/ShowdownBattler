import asyncio
import argparse

from poke_env import AccountConfiguration, ShowdownServerConfiguration, ServerConfiguration

from bot.player import AdvancedHeuristicPlayer


def parse_args():
    p = argparse.ArgumentParser(description="Run the ShowdownBattler bot against a Showdown server.")

    p.add_argument("--username", default="BattlerBot",
                   help="Bot username (default: BattlerBot)")
    p.add_argument("--password", default=None,
                   help="Account password. Not required for local servers.")
    p.add_argument("--format", default="gen9randombattle",
                   help="Battle format (default: gen9randombattle)")

    p.add_argument("--mode", choices=["accept", "challenge", "ladder"], default="accept",
                   help="How to find battles (default: accept)")
    p.add_argument("--opponent", default=None,
                   help="Opponent username for 'challenge' mode, or filter for 'accept' mode.")
    p.add_argument("--n", type=int, default=1,
                   help="Number of battles to play (default: 1)")

    p.add_argument("--server", choices=["local", "showdown"], default="local",
                   help="Which server to connect to (default: local)")
    p.add_argument("--port", type=int, default=8000,
                   help="Port for local server (default: 8000)")

    p.add_argument("--no-mcts", action="store_true",
                   help="Disable MCTS — use pure heuristics only")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-turn output")

    return p.parse_args()


async def main():
    args = parse_args()

    if args.server == "local":
        server_config = ServerConfiguration(
            f"ws://localhost:{args.port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        )
    else:
        server_config = ShowdownServerConfiguration

    player = AdvancedHeuristicPlayer(
        account_configuration=AccountConfiguration(args.username, args.password),
        battle_format=args.format,
        server_configuration=server_config,
    )
    player.use_mcts = not args.no_mcts
    player.verbose = not args.quiet

    print(f"[BattlerBot] Connected as '{args.username}' | format={args.format} | "
          f"MCTS={'on' if player.use_mcts else 'off'} | server={args.server}:{args.port if args.server == 'local' else 'default'}")

    if args.mode == "accept":
        target = args.opponent or "anyone"
        print(f"[BattlerBot] Waiting to accept {args.n} challenge(s) from {target}...")
        await player.accept_challenges(args.opponent, args.n)

    elif args.mode == "challenge":
        if not args.opponent:
            print("Error: --opponent is required for challenge mode.")
            return
        print(f"[BattlerBot] Challenging '{args.opponent}' to {args.n} battle(s)...")
        await player.send_challenges(args.opponent, args.n)

    elif args.mode == "ladder":
        print(f"[BattlerBot] Playing {args.n} ladder game(s)...")
        await player.ladder(args.n)

    # Summary
    print(f"\n[BattlerBot] Done — {player.n_won_battles}W / {player.n_lost_battles}L "
          f"({player.n_won_battles + player.n_lost_battles} battles)")


if __name__ == "__main__":
    asyncio.run(main())
