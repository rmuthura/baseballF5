"""
Full game F5 prediction CLI.

Takes both teams' pitchers and lineups, outputs:
- Win probability for each team
- Projected F5 score
- Spread and total

Usage:
    python -m f5_model.model.game_predict \
        --away-pitcher "Gerrit Cole" \
        --away-lineup "Betts,Ohtani,Freeman,..." \
        --home-pitcher "Corbin Burnes" \
        --home-lineup "Soto,Judge,Stanton,..." \
        --date 2026-04-05 \
        --park NYY
"""

import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import poisson

from f5_model.model.predict import (
    load_model_and_features,
    lookup_player_id,
    predict_f5_runs
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_win_probability(lambda_away: float, lambda_home: float, max_runs: int = 15) -> Dict:
    """
    Compute win probability for each team using Poisson distributions.

    Args:
        lambda_away: Predicted F5 runs for away team
        lambda_home: Predicted F5 runs for home team
        max_runs: Maximum runs to consider in calculation

    Returns:
        Dictionary with win probabilities and tie probability
    """
    # Build probability mass functions
    away_probs = np.array([poisson.pmf(k, lambda_away) for k in range(max_runs + 1)])
    home_probs = np.array([poisson.pmf(k, lambda_home) for k in range(max_runs + 1)])

    # P(away wins) = sum over all i,j where i > j of P(away=i) * P(home=j)
    p_away_wins = 0.0
    p_home_wins = 0.0
    p_tie = 0.0

    for i in range(max_runs + 1):
        for j in range(max_runs + 1):
            joint_prob = away_probs[i] * home_probs[j]
            if i > j:
                p_away_wins += joint_prob
            elif j > i:
                p_home_wins += joint_prob
            else:
                p_tie += joint_prob

    return {
        'away_win': p_away_wins,
        'home_win': p_home_wins,
        'tie': p_tie
    }


def compute_spread_probabilities(
    lambda_away: float,
    lambda_home: float,
    spreads: List[float] = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
    max_runs: int = 15
) -> Dict:
    """
    Compute probability of covering various spreads.

    Spread is from away team perspective (negative = away favored).

    Args:
        lambda_away: Predicted F5 runs for away team
        lambda_home: Predicted F5 runs for home team
        spreads: List of spread lines to evaluate
        max_runs: Maximum runs to consider

    Returns:
        Dictionary with cover probabilities for each spread
    """
    away_probs = np.array([poisson.pmf(k, lambda_away) for k in range(max_runs + 1)])
    home_probs = np.array([poisson.pmf(k, lambda_home) for k in range(max_runs + 1)])

    results = {}

    for spread in spreads:
        # Away team covers if: away_runs - home_runs > spread (for positive spread)
        # or away_runs - home_runs > spread (away needs to win by more than |spread| if negative)
        p_cover = 0.0

        for i in range(max_runs + 1):
            for j in range(max_runs + 1):
                margin = i - j  # away margin
                if margin > spread:  # away covers
                    p_cover += away_probs[i] * home_probs[j]

        results[spread] = p_cover

    return results


def compute_total_probabilities(
    lambda_away: float,
    lambda_home: float,
    totals: List[float] = [3.5, 4.5, 5.5, 6.5, 7.5],
    max_runs: int = 15
) -> Dict:
    """
    Compute over/under probabilities for total runs.

    Args:
        lambda_away: Predicted F5 runs for away team
        lambda_home: Predicted F5 runs for home team
        totals: List of total lines to evaluate
        max_runs: Maximum runs to consider

    Returns:
        Dictionary with over probabilities for each total
    """
    away_probs = np.array([poisson.pmf(k, lambda_away) for k in range(max_runs + 1)])
    home_probs = np.array([poisson.pmf(k, lambda_home) for k in range(max_runs + 1)])

    results = {}

    for total in totals:
        p_over = 0.0

        for i in range(max_runs + 1):
            for j in range(max_runs + 1):
                if i + j > total:
                    p_over += away_probs[i] * home_probs[j]

        results[total] = {
            'over': p_over,
            'under': 1 - p_over
        }

    return results


def prob_to_american_odds(prob: float) -> str:
    """Convert probability to American odds format."""
    if prob <= 0:
        return "N/A"
    if prob >= 1:
        return "N/A"

    if prob >= 0.5:
        odds = -100 * prob / (1 - prob)
        return f"{int(odds)}"
    else:
        odds = 100 * (1 - prob) / prob
        return f"+{int(odds)}"


def format_game_output(
    away_team: str,
    home_team: str,
    away_pitcher: str,
    home_pitcher: str,
    away_lambda: float,
    home_lambda: float,
    win_probs: Dict,
    spread_probs: Dict,
    total_probs: Dict
) -> str:
    """Format the full game prediction output."""
    lines = []

    lines.append("=" * 70)
    lines.append("F5 GAME PREDICTION")
    lines.append("=" * 70)

    # Matchup
    lines.append(f"\n{away_team} @ {home_team}")
    lines.append(f"Away: {away_pitcher}")
    lines.append(f"Home: {home_pitcher}")

    # Projected Score
    lines.append(f"\n{'=' * 70}")
    lines.append("PROJECTED F5 SCORE")
    lines.append("=" * 70)
    lines.append(f"\n  {away_team}: {away_lambda:.2f}")
    lines.append(f"  {home_team}: {home_lambda:.2f}")
    lines.append(f"\n  Total: {away_lambda + home_lambda:.2f}")
    lines.append(f"  Spread: {home_team} -{abs(home_lambda - away_lambda):.2f}" if home_lambda < away_lambda
                 else f"  Spread: {away_team} -{abs(away_lambda - home_lambda):.2f}" if away_lambda < home_lambda
                 else "  Spread: PICK")

    # Win Probability
    lines.append(f"\n{'=' * 70}")
    lines.append("F5 WIN PROBABILITY")
    lines.append("=" * 70)
    lines.append(f"\n  {away_team}: {win_probs['away_win']:6.1%}  ({prob_to_american_odds(win_probs['away_win'])})")
    lines.append(f"  {home_team}: {win_probs['home_win']:6.1%}  ({prob_to_american_odds(win_probs['home_win'])})")
    lines.append(f"  Tie:        {win_probs['tie']:6.1%}  ({prob_to_american_odds(win_probs['tie'])})")

    # Moneyline (excluding ties)
    ml_away = win_probs['away_win'] / (1 - win_probs['tie'])
    ml_home = win_probs['home_win'] / (1 - win_probs['tie'])
    lines.append(f"\n  Moneyline (excl. ties):")
    lines.append(f"    {away_team}: {ml_away:6.1%}  ({prob_to_american_odds(ml_away)})")
    lines.append(f"    {home_team}: {ml_home:6.1%}  ({prob_to_american_odds(ml_home)})")

    # Spread
    lines.append(f"\n{'=' * 70}")
    lines.append("F5 SPREAD (Away Team Perspective)")
    lines.append("=" * 70)
    lines.append(f"\n  {'Line':<12} {'Away Cover':>12} {'Home Cover':>12}")
    lines.append("  " + "-" * 38)

    for spread in sorted(spread_probs.keys()):
        p_away_cover = spread_probs[spread]
        p_home_cover = 1 - p_away_cover

        if spread < 0:
            label = f"{away_team} {spread}"
        elif spread > 0:
            label = f"{away_team} +{spread}"
        else:
            label = "PICK"

        lines.append(f"  {label:<12} {p_away_cover:>11.1%} {p_home_cover:>11.1%}")

    # Total
    lines.append(f"\n{'=' * 70}")
    lines.append("F5 TOTAL")
    lines.append("=" * 70)
    lines.append(f"\n  {'Line':<8} {'Over':>10} {'Under':>10}")
    lines.append("  " + "-" * 30)

    for total in sorted(total_probs.keys()):
        p_over = total_probs[total]['over']
        p_under = total_probs[total]['under']
        lines.append(f"  {total:<8} {p_over:>9.1%} {p_under:>9.1%}")

    # Most likely scores
    lines.append(f"\n{'=' * 70}")
    lines.append("MOST LIKELY F5 SCORES")
    lines.append("=" * 70)

    # Calculate joint probabilities for most likely scores
    score_probs = []
    for i in range(8):
        for j in range(8):
            p = poisson.pmf(i, away_lambda) * poisson.pmf(j, home_lambda)
            score_probs.append((i, j, p))

    score_probs.sort(key=lambda x: x[2], reverse=True)

    lines.append(f"\n  {'Score':<12} {'Probability':>12}")
    lines.append("  " + "-" * 26)
    for away_score, home_score, prob in score_probs[:10]:
        lines.append(f"  {away_score}-{home_score:<10} {prob:>11.1%}")

    return "\n".join(lines)


def parse_lineup(lineup_str: str) -> List[str]:
    """Parse comma-separated lineup string."""
    return [name.strip() for name in lineup_str.split(',')]


def lookup_lineup_ids(names: List[str]) -> Tuple[List[int], List[str]]:
    """Look up MLB IDs for a list of player names."""
    ids = []
    found_names = []

    for name in names:
        player_id = lookup_player_id(name)
        if player_id:
            ids.append(player_id)
            found_names.append(name)
            print(f"    {name}: {player_id}")
        else:
            print(f"    {name}: NOT FOUND")

    return ids, found_names


def main():
    parser = argparse.ArgumentParser(
        description="Predict F5 outcome for a full game matchup"
    )

    # Away team
    parser.add_argument(
        "--away-pitcher", "-ap",
        required=True,
        help="Away team starting pitcher name"
    )
    parser.add_argument(
        "--away-lineup", "-al",
        required=True,
        help="Away team lineup (comma-separated names)"
    )
    parser.add_argument(
        "--away-pitcher-id",
        type=int,
        help="Away pitcher MLB ID (skip name lookup)"
    )
    parser.add_argument(
        "--away-lineup-ids",
        help="Away lineup MLB IDs (comma-separated)"
    )
    parser.add_argument(
        "--away-hand",
        choices=['L', 'R'],
        default='R',
        help="Away pitcher handedness"
    )
    parser.add_argument(
        "--away-team",
        default="AWAY",
        help="Away team name for display"
    )

    # Home team
    parser.add_argument(
        "--home-pitcher", "-hp",
        required=True,
        help="Home team starting pitcher name"
    )
    parser.add_argument(
        "--home-lineup", "-hl",
        required=True,
        help="Home team lineup (comma-separated names)"
    )
    parser.add_argument(
        "--home-pitcher-id",
        type=int,
        help="Home pitcher MLB ID (skip name lookup)"
    )
    parser.add_argument(
        "--home-lineup-ids",
        help="Home lineup MLB IDs (comma-separated)"
    )
    parser.add_argument(
        "--home-hand",
        choices=['L', 'R'],
        default='R',
        help="Home pitcher handedness"
    )
    parser.add_argument(
        "--home-team",
        default="HOME",
        help="Home team name for display"
    )

    # Game context
    parser.add_argument(
        "--date", "-d",
        required=True,
        help="Game date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--park",
        default="NYY",
        help="Park code for park factor"
    )

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, feature_names = load_model_and_features()

    # Parse lineups
    away_lineup_names = parse_lineup(args.away_lineup)
    home_lineup_names = parse_lineup(args.home_lineup)

    # Look up away pitcher
    if args.away_pitcher_id:
        away_pitcher_id = args.away_pitcher_id
    else:
        print(f"\nLooking up away pitcher: {args.away_pitcher}...")
        away_pitcher_id = lookup_player_id(args.away_pitcher)
        if away_pitcher_id is None:
            print(f"ERROR: Could not find pitcher '{args.away_pitcher}'")
            return
        print(f"  {args.away_pitcher}: {away_pitcher_id}")

    # Look up home pitcher
    if args.home_pitcher_id:
        home_pitcher_id = args.home_pitcher_id
    else:
        print(f"\nLooking up home pitcher: {args.home_pitcher}...")
        home_pitcher_id = lookup_player_id(args.home_pitcher)
        if home_pitcher_id is None:
            print(f"ERROR: Could not find pitcher '{args.home_pitcher}'")
            return
        print(f"  {args.home_pitcher}: {home_pitcher_id}")

    # Look up away lineup
    if args.away_lineup_ids:
        away_lineup_ids = [int(x) for x in args.away_lineup_ids.split(',')]
    else:
        print(f"\nLooking up away lineup...")
        away_lineup_ids, _ = lookup_lineup_ids(away_lineup_names)

    if len(away_lineup_ids) == 0:
        print("ERROR: No batters found in away lineup")
        return

    # Look up home lineup
    if args.home_lineup_ids:
        home_lineup_ids = [int(x) for x in args.home_lineup_ids.split(',')]
    else:
        print(f"\nLooking up home lineup...")
        home_lineup_ids, _ = lookup_lineup_ids(home_lineup_names)

    if len(home_lineup_ids) == 0:
        print("ERROR: No batters found in home lineup")
        return

    # Predict F5 runs for each side
    print(f"\nPredicting F5 runs...")

    # Away team scoring = Home pitcher vs Away lineup
    # Home pitcher is pitching at home
    away_result = predict_f5_runs(
        model=model,
        feature_names=feature_names,
        pitcher_id=home_pitcher_id,
        pitcher_hand=args.home_hand,
        lineup_ids=away_lineup_ids,
        date=args.date,
        starter_is_home=True,
        park=args.park
    )
    away_lambda = away_result['predicted_runs']

    # Home team scoring = Away pitcher vs Home lineup
    # Away pitcher is pitching away
    home_result = predict_f5_runs(
        model=model,
        feature_names=feature_names,
        pitcher_id=away_pitcher_id,
        pitcher_hand=args.away_hand,
        lineup_ids=home_lineup_ids,
        date=args.date,
        starter_is_home=False,
        park=args.park
    )
    home_lambda = home_result['predicted_runs']

    # Compute probabilities
    win_probs = compute_win_probability(away_lambda, home_lambda)
    spread_probs = compute_spread_probabilities(away_lambda, home_lambda)
    total_probs = compute_total_probabilities(away_lambda, home_lambda)

    # Format and print output
    output = format_game_output(
        away_team=args.away_team,
        home_team=args.home_team,
        away_pitcher=args.away_pitcher,
        home_pitcher=args.home_pitcher,
        away_lambda=away_lambda,
        home_lambda=home_lambda,
        win_probs=win_probs,
        spread_probs=spread_probs,
        total_probs=total_probs
    )

    print(output)


if __name__ == "__main__":
    main()
