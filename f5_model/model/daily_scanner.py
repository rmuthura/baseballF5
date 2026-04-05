"""
Daily F5 Scanner.

Scans all games for a given day, runs predictions, compares to FanDuel odds,
and outputs the highest edge bets.

Usage:
    python -m f5_model.model.daily_scanner --date 2026-04-04

    # With API key directly
    python -m f5_model.model.daily_scanner --date 2026-04-04 --api-key YOUR_KEY

    # Without odds (model predictions only)
    python -m f5_model.model.daily_scanner --date 2026-04-04 --no-odds

    # With manual odds from CSV file
    python -m f5_model.model.daily_scanner --date 2026-04-04 --odds-file odds.csv

    # Interactive mode - prompt for odds
    python -m f5_model.model.daily_scanner --date 2026-04-04 --manual-odds

CSV Format for odds file:
    away_team,home_team,away_ml,home_ml,total,over_odds,under_odds
    NYY,BOS,-150,+130,4.5,-110,-110
    LAD,SF,+120,-140,5.0,-105,-115
"""

import argparse
import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import poisson

from f5_model.model.predict import (
    load_model_and_features,
    lookup_player_id,
    predict_f5_runs
)
from f5_model.utils.lineup_scraper import get_daily_matchups, Matchup

# Try to import odds API
try:
    from f5_model.utils.odds_api import get_daily_odds, GameOdds
    ODDS_API_AVAILABLE = True
except ImportError:
    ODDS_API_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ManualOdds:
    """Odds entered manually or from CSV."""
    away_team: str
    home_team: str
    away_ml: Optional[int] = None
    home_ml: Optional[int] = None
    total: Optional[float] = None
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None


def parse_odds_csv(filepath: str) -> Dict[str, ManualOdds]:
    """
    Parse odds from a CSV file.

    Expected columns: away_team,home_team,away_ml,home_ml,total,over_odds,under_odds
    All odds should be in American format (e.g., -150, +130)

    Returns:
        Dict mapping "AWAY @ HOME" to ManualOdds
    """
    odds = {}

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                away = row['away_team'].strip().upper()
                home = row['home_team'].strip().upper()

                manual = ManualOdds(
                    away_team=away,
                    home_team=home,
                    away_ml=int(row['away_ml']) if row.get('away_ml') else None,
                    home_ml=int(row['home_ml']) if row.get('home_ml') else None,
                    total=float(row['total']) if row.get('total') else None,
                    over_odds=int(row['over_odds']) if row.get('over_odds') else None,
                    under_odds=int(row['under_odds']) if row.get('under_odds') else None
                )

                key = f"{away} @ {home}"
                odds[key] = manual
                logger.info(f"  Loaded odds: {key}")

            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid row: {row} - {e}")
                continue

    return odds


def prompt_for_odds(games: List[Tuple[str, str]]) -> Dict[str, ManualOdds]:
    """
    Interactively prompt user for odds.

    Args:
        games: List of (away_team, home_team) tuples

    Returns:
        Dict mapping "AWAY @ HOME" to ManualOdds
    """
    odds = {}

    print("\n" + "=" * 60)
    print("ENTER FANDUEL F5 ODDS")
    print("=" * 60)
    print("Enter odds in American format (e.g., -150, +130)")
    print("Press Enter to skip a field, 'q' to quit early\n")

    for away, home in games:
        key = f"{away} @ {home}"
        print(f"\n--- {key} ---")

        try:
            # Moneyline
            away_ml_str = input(f"  {away} F5 ML: ").strip()
            if away_ml_str.lower() == 'q':
                break
            away_ml = int(away_ml_str) if away_ml_str else None

            home_ml_str = input(f"  {home} F5 ML: ").strip()
            if home_ml_str.lower() == 'q':
                break
            home_ml = int(home_ml_str) if home_ml_str else None

            # Total
            total_str = input(f"  F5 Total (line): ").strip()
            if total_str.lower() == 'q':
                break
            total = float(total_str) if total_str else None

            over_odds = None
            under_odds = None
            if total:
                over_str = input(f"  Over {total} odds: ").strip()
                over_odds = int(over_str) if over_str else None

                under_str = input(f"  Under {total} odds: ").strip()
                under_odds = int(under_str) if under_str else None

            odds[key] = ManualOdds(
                away_team=away,
                home_team=home,
                away_ml=away_ml,
                home_ml=home_ml,
                total=total,
                over_odds=over_odds,
                under_odds=under_odds
            )

        except ValueError as e:
            print(f"  Invalid input, skipping: {e}")
            continue

    return odds


def convert_manual_to_gameodds(manual_odds: Dict[str, ManualOdds]) -> Dict:
    """
    Convert ManualOdds to the format expected by find_edges.

    Creates a dict with fg_away_ml, fg_home_ml, etc. attributes.
    """
    class OddsWrapper:
        """Wrapper to mimic GameOdds interface."""
        def __init__(self, mo: ManualOdds):
            self.fg_away_ml = mo.away_ml
            self.fg_home_ml = mo.home_ml
            self.fg_total = mo.total
            self.fg_over_odds = mo.over_odds
            self.fg_under_odds = mo.under_odds

    return {key: OddsWrapper(mo) for key, mo in manual_odds.items()}


@dataclass
class GamePrediction:
    """Full prediction for a single game."""
    away_team: str
    home_team: str
    away_pitcher: str
    home_pitcher: str
    away_runs: float
    home_runs: float
    total: float
    away_win_prob: float
    home_win_prob: float
    tie_prob: float
    game_time: str = ""
    # Edges vs book
    edges: Dict = field(default_factory=dict)


@dataclass
class BetRecommendation:
    """A single bet recommendation."""
    game: str
    market: str
    pick: str
    model_prob: float
    model_odds: int
    book_odds: Optional[int]
    edge: float
    ev: float  # Expected value per $100 bet


def prob_to_american(prob: float) -> int:
    """Convert probability to American odds."""
    if prob <= 0 or prob >= 1:
        return 0
    if prob >= 0.5:
        return int(round(-100 * prob / (1 - prob)))
    else:
        return int(round(100 * (1 - prob) / prob))


def american_to_prob(odds: int) -> float:
    """Convert American odds to probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def calculate_ev(model_prob: float, book_odds: int) -> float:
    """
    Calculate expected value per $100 bet.

    EV = (prob * payout) - (1 - prob) * stake
    """
    if book_odds > 0:
        payout = book_odds
    else:
        payout = 100 * 100 / abs(book_odds)

    ev = (model_prob * payout) - ((1 - model_prob) * 100)
    return ev


def compute_game_probs(away_lambda: float, home_lambda: float) -> Dict:
    """Compute all probabilities for a game."""
    max_runs = 15

    away_probs = np.array([poisson.pmf(k, away_lambda) for k in range(max_runs + 1)])
    home_probs = np.array([poisson.pmf(k, home_lambda) for k in range(max_runs + 1)])
    joint = np.outer(away_probs, home_probs)

    # Win probabilities
    p_away_win = sum(joint[i, j] for i in range(max_runs + 1) for j in range(max_runs + 1) if i > j)
    p_home_win = sum(joint[i, j] for i in range(max_runs + 1) for j in range(max_runs + 1) if j > i)
    p_tie = sum(joint[i, i] for i in range(max_runs + 1))

    # 2-way ML
    ml_total = p_away_win + p_home_win
    p_away_ml = p_away_win / ml_total
    p_home_ml = p_home_win / ml_total

    # Totals
    totals = {}
    for line in [3.5, 4.5, 5.5]:
        p_over = sum(joint[i, j] for i in range(max_runs + 1) for j in range(max_runs + 1) if i + j > line)
        totals[line] = {'over': p_over, 'under': 1 - p_over}

    # Run lines (away perspective)
    run_lines = {}
    for spread in [0.5, 1.5]:
        p_away_cover = sum(joint[i, j] for i in range(max_runs + 1) for j in range(max_runs + 1) if i - j > -spread)
        run_lines[spread] = {'away': p_away_cover, 'home': 1 - p_away_cover}

    return {
        'away_win_3way': p_away_win,
        'home_win_3way': p_home_win,
        'tie': p_tie,
        'away_ml': p_away_ml,
        'home_ml': p_home_ml,
        'totals': totals,
        'run_lines': run_lines
    }


def lookup_lineup_ids(lineup_names: List[str]) -> List[int]:
    """Look up MLB IDs for a lineup."""
    ids = []
    for name in lineup_names:
        if not name:
            continue
        player_id = lookup_player_id(name)
        if player_id:
            ids.append(player_id)
    return ids


def predict_game(
    model,
    feature_names: List[str],
    matchup: Matchup,
    date: str
) -> Optional[GamePrediction]:
    """
    Run prediction for a single game.

    Args:
        model: Trained model
        feature_names: Feature names
        matchup: Matchup object with lineups
        date: Game date

    Returns:
        GamePrediction or None if can't predict
    """
    # Skip if lineups not available
    if not matchup.away_lineup or not matchup.home_lineup:
        logger.warning(f"Skipping {matchup.away_abbrev} @ {matchup.home_abbrev}: lineups not available")
        return None

    if matchup.away_pitcher == 'TBD' or matchup.home_pitcher == 'TBD':
        logger.warning(f"Skipping {matchup.away_abbrev} @ {matchup.home_abbrev}: pitcher TBD")
        return None

    # Look up pitcher IDs
    away_pitcher_id = lookup_player_id(matchup.away_pitcher)
    home_pitcher_id = lookup_player_id(matchup.home_pitcher)

    if not away_pitcher_id or not home_pitcher_id:
        logger.warning(f"Skipping {matchup.away_abbrev} @ {matchup.home_abbrev}: pitcher not found")
        return None

    # Look up lineup IDs
    away_lineup_ids = lookup_lineup_ids(matchup.away_lineup)
    home_lineup_ids = lookup_lineup_ids(matchup.home_lineup)

    if len(away_lineup_ids) < 5 or len(home_lineup_ids) < 5:
        logger.warning(f"Skipping {matchup.away_abbrev} @ {matchup.home_abbrev}: not enough batters found")
        return None

    try:
        # Away team runs = Home pitcher vs Away lineup
        away_result = predict_f5_runs(
            model=model,
            feature_names=feature_names,
            pitcher_id=home_pitcher_id,
            pitcher_hand=matchup.home_pitcher_hand,
            lineup_ids=away_lineup_ids,
            date=date,
            starter_is_home=True,
            park=matchup.park
        )

        # Home team runs = Away pitcher vs Home lineup
        home_result = predict_f5_runs(
            model=model,
            feature_names=feature_names,
            pitcher_id=away_pitcher_id,
            pitcher_hand=matchup.away_pitcher_hand,
            lineup_ids=home_lineup_ids,
            date=date,
            starter_is_home=False,
            park=matchup.park
        )

        away_lambda = away_result['predicted_runs']
        home_lambda = home_result['predicted_runs']

        probs = compute_game_probs(away_lambda, home_lambda)

        return GamePrediction(
            away_team=matchup.away_abbrev,
            home_team=matchup.home_abbrev,
            away_pitcher=matchup.away_pitcher,
            home_pitcher=matchup.home_pitcher,
            away_runs=away_lambda,
            home_runs=home_lambda,
            total=away_lambda + home_lambda,
            away_win_prob=probs['away_ml'],
            home_win_prob=probs['home_ml'],
            tie_prob=probs['tie'],
            game_time=matchup.game_time,
            edges={'probs': probs}
        )

    except Exception as e:
        logger.error(f"Error predicting {matchup.away_abbrev} @ {matchup.home_abbrev}: {e}")
        return None


def find_edges(
    predictions: List[GamePrediction],
    odds: Dict[str, 'GameOdds'],
    min_edge: float = 0.02
) -> List[BetRecommendation]:
    """
    Find betting edges by comparing model to book odds.

    Args:
        predictions: List of game predictions
        odds: Dict of game odds from API
        min_edge: Minimum edge to include (default 2%)

    Returns:
        List of bet recommendations sorted by edge
    """
    recommendations = []

    for pred in predictions:
        game_key = f"{pred.away_team} @ {pred.home_team}"
        probs = pred.edges.get('probs', {})

        # Try to find matching odds
        game_odds = None
        for key, go in odds.items():
            if pred.away_team in key and pred.home_team in key:
                game_odds = go
                break

        if not game_odds:
            continue

        # Check moneyline edges
        if game_odds.fg_away_ml:
            model_prob = probs.get('away_ml', 0)
            book_prob = american_to_prob(game_odds.fg_away_ml)
            edge = model_prob - book_prob

            if edge >= min_edge:
                recommendations.append(BetRecommendation(
                    game=game_key,
                    market="F5 ML",
                    pick=pred.away_team,
                    model_prob=model_prob,
                    model_odds=prob_to_american(model_prob),
                    book_odds=game_odds.fg_away_ml,
                    edge=edge,
                    ev=calculate_ev(model_prob, game_odds.fg_away_ml)
                ))

        if game_odds.fg_home_ml:
            model_prob = probs.get('home_ml', 0)
            book_prob = american_to_prob(game_odds.fg_home_ml)
            edge = model_prob - book_prob

            if edge >= min_edge:
                recommendations.append(BetRecommendation(
                    game=game_key,
                    market="F5 ML",
                    pick=pred.home_team,
                    model_prob=model_prob,
                    model_odds=prob_to_american(model_prob),
                    book_odds=game_odds.fg_home_ml,
                    edge=edge,
                    ev=calculate_ev(model_prob, game_odds.fg_home_ml)
                ))

        # Check totals
        if game_odds.fg_total and game_odds.fg_over_odds:
            total_line = game_odds.fg_total
            # Find closest line we calculated
            for line, probs_dict in probs.get('totals', {}).items():
                if abs(line - total_line) < 1:
                    # Over
                    model_prob = probs_dict['over']
                    book_prob = american_to_prob(game_odds.fg_over_odds)
                    edge = model_prob - book_prob

                    if edge >= min_edge:
                        recommendations.append(BetRecommendation(
                            game=game_key,
                            market=f"F5 Total",
                            pick=f"Over {total_line}",
                            model_prob=model_prob,
                            model_odds=prob_to_american(model_prob),
                            book_odds=game_odds.fg_over_odds,
                            edge=edge,
                            ev=calculate_ev(model_prob, game_odds.fg_over_odds)
                        ))

                    # Under
                    if game_odds.fg_under_odds:
                        model_prob = probs_dict['under']
                        book_prob = american_to_prob(game_odds.fg_under_odds)
                        edge = model_prob - book_prob

                        if edge >= min_edge:
                            recommendations.append(BetRecommendation(
                                game=game_key,
                                market=f"F5 Total",
                                pick=f"Under {total_line}",
                                model_prob=model_prob,
                                model_odds=prob_to_american(model_prob),
                                book_odds=game_odds.fg_under_odds,
                                edge=edge,
                                ev=calculate_ev(model_prob, game_odds.fg_under_odds)
                            ))

    # Sort by edge descending
    recommendations.sort(key=lambda x: x.edge, reverse=True)

    return recommendations


def format_output(
    predictions: List[GamePrediction],
    recommendations: List[BetRecommendation],
    date: str
) -> str:
    """Format the daily scanner output."""
    lines = []

    lines.append("=" * 80)
    lines.append(f"F5 DAILY SCANNER - {date}")
    lines.append("=" * 80)

    # Summary of all games
    lines.append(f"\n{'=' * 80}")
    lines.append("ALL GAMES - MODEL PREDICTIONS")
    lines.append("=" * 80)
    lines.append(f"\n  {'Game':<20} {'Pitchers':<35} {'Proj Score':<12} {'Total':<8} {'Fav':<8}")
    lines.append("  " + "-" * 85)

    for pred in sorted(predictions, key=lambda x: x.game_time):
        pitchers = f"{pred.away_pitcher[:15]} vs {pred.home_pitcher[:15]}"
        score = f"{pred.away_runs:.1f} - {pred.home_runs:.1f}"

        if pred.away_win_prob > pred.home_win_prob:
            fav = f"{pred.away_team} {prob_to_american(pred.away_win_prob)}"
        else:
            fav = f"{pred.home_team} {prob_to_american(pred.home_win_prob)}"

        lines.append(f"  {pred.away_team} @ {pred.home_team:<13} {pitchers:<35} {score:<12} {pred.total:<8.1f} {fav:<8}")

    # Top edges
    if recommendations:
        lines.append(f"\n{'=' * 80}")
        lines.append("TOP EDGES vs FANDUEL")
        lines.append("=" * 80)
        lines.append(f"\n  {'Game':<22} {'Market':<12} {'Pick':<15} {'Model':<8} {'Book':<8} {'Edge':<8} {'EV':<8}")
        lines.append("  " + "-" * 85)

        for rec in recommendations[:15]:  # Top 15
            model_odds = f"{rec.model_odds:+d}" if rec.model_odds else "N/A"
            book_odds = f"{rec.book_odds:+d}" if rec.book_odds else "N/A"
            edge_str = f"{rec.edge:.1%}"
            ev_str = f"${rec.ev:+.1f}"

            # Mark strong edges
            if rec.edge >= 0.05:
                edge_str += " **"
            elif rec.edge >= 0.03:
                edge_str += " *"

            lines.append(f"  {rec.game:<22} {rec.market:<12} {rec.pick:<15} {model_odds:<8} {book_odds:<8} {edge_str:<8} {ev_str:<8}")

        lines.append("\n  Legend: ** = 5%+ edge (strong), * = 3-5% edge")
        lines.append("  EV = Expected Value per $100 bet")

    else:
        lines.append(f"\n{'=' * 80}")
        lines.append("NO EDGES FOUND")
        lines.append("=" * 80)
        lines.append("\n  No bets with 2%+ edge found against current FanDuel lines.")
        lines.append("  This could mean:")
        lines.append("    - Lines are efficient")
        lines.append("    - Odds API doesn't have F5-specific markets")
        lines.append("    - Model sees no value today")

    # Best bets summary
    if recommendations:
        strong_edges = [r for r in recommendations if r.edge >= 0.05]
        if strong_edges:
            lines.append(f"\n{'=' * 80}")
            lines.append("BEST BETS (5%+ Edge)")
            lines.append("=" * 80)
            for rec in strong_edges[:5]:
                lines.append(f"\n  {rec.pick} ({rec.market})")
                lines.append(f"    Game: {rec.game}")
                lines.append(f"    Model: {rec.model_prob:.1%} ({rec.model_odds:+d})")
                lines.append(f"    Book:  {american_to_prob(rec.book_odds):.1%} ({rec.book_odds:+d})")
                lines.append(f"    Edge:  {rec.edge:.1%}")
                lines.append(f"    EV:    ${rec.ev:+.1f} per $100")

    return "\n".join(lines)


def run_daily_scan(
    date: str,
    api_key: str = None,
    use_odds: bool = True,
    min_edge: float = 0.02,
    odds_file: str = None,
    manual_odds_input: bool = False
) -> Tuple[List[GamePrediction], List[BetRecommendation]]:
    """
    Run the full daily scan.

    Args:
        date: Date to scan (YYYY-MM-DD)
        api_key: Odds API key
        use_odds: Whether to fetch odds from API
        min_edge: Minimum edge threshold
        odds_file: Path to CSV file with odds
        manual_odds_input: Whether to prompt for manual odds

    Returns:
        Tuple of (predictions, recommendations)
    """
    logger.info(f"Starting daily scan for {date}")

    # Load model
    logger.info("Loading model...")
    model, feature_names = load_model_and_features()

    # Get matchups
    logger.info("Fetching lineups...")
    matchups = get_daily_matchups(date)
    logger.info(f"Found {len(matchups)} games")

    if not matchups:
        logger.warning("No games found for this date")
        return [], []

    # Get odds - priority: CSV file > manual input > API
    odds = {}

    if odds_file:
        # Load from CSV
        logger.info(f"Loading odds from {odds_file}...")
        manual_odds = parse_odds_csv(odds_file)
        odds = convert_manual_to_gameodds(manual_odds)
        logger.info(f"Loaded odds for {len(odds)} games from CSV")

    elif manual_odds_input:
        # Interactive prompt
        games = [(m.away_abbrev, m.home_abbrev) for m in matchups]
        manual_odds = prompt_for_odds(games)
        odds = convert_manual_to_gameodds(manual_odds)
        logger.info(f"Entered odds for {len(odds)} games")

    elif use_odds and ODDS_API_AVAILABLE:
        # API fetch
        try:
            if api_key:
                os.environ['ODDS_API_KEY'] = api_key
            logger.info("Fetching odds from API...")
            odds = get_daily_odds()
            logger.info(f"Got odds for {len(odds)} games from API")
        except Exception as e:
            logger.warning(f"Could not fetch odds: {e}")

    # Run predictions
    logger.info("Running predictions...")
    predictions = []
    for matchup in matchups:
        logger.info(f"  Predicting {matchup.away_abbrev} @ {matchup.home_abbrev}...")
        pred = predict_game(model, feature_names, matchup, date)
        if pred:
            predictions.append(pred)

    logger.info(f"Successfully predicted {len(predictions)} games")

    # Find edges
    recommendations = []
    if odds and predictions:
        recommendations = find_edges(predictions, odds, min_edge)
        logger.info(f"Found {len(recommendations)} edges >= {min_edge:.0%}")

    return predictions, recommendations


def generate_odds_template(predictions: List[GamePrediction], filepath: str):
    """Generate a CSV template for entering odds."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['away_team', 'home_team', 'away_ml', 'home_ml', 'total', 'over_odds', 'under_odds'])

        for pred in predictions:
            writer.writerow([pred.away_team, pred.home_team, '', '', '', '', ''])

    print(f"Odds template saved to {filepath}")
    print("Fill in the odds and run again with --odds-file")


def main():
    parser = argparse.ArgumentParser(
        description="Daily F5 Scanner - Find betting edges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Predictions only (no odds)
    python -m f5_model.model.daily_scanner --date 2026-04-04 --no-odds

    # With manual odds from CSV
    python -m f5_model.model.daily_scanner --date 2026-04-04 --odds-file odds.csv

    # Interactive mode - enter odds manually
    python -m f5_model.model.daily_scanner --date 2026-04-04 --manual-odds

    # Generate template CSV for entering odds
    python -m f5_model.model.daily_scanner --date 2026-04-04 --template odds.csv

    # With API (requires ODDS_API_KEY environment variable)
    python -m f5_model.model.daily_scanner --date 2026-04-04 --api-key YOUR_KEY
        """
    )
    parser.add_argument("--date", "-d", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date to scan (YYYY-MM-DD)")
    parser.add_argument("--api-key", help="Odds API key (or set ODDS_API_KEY env var)")
    parser.add_argument("--no-odds", action="store_true", help="Skip fetching odds (model only)")
    parser.add_argument("--odds-file", help="CSV file with FanDuel odds")
    parser.add_argument("--manual-odds", action="store_true", help="Enter odds interactively")
    parser.add_argument("--template", help="Generate CSV template for odds input")
    parser.add_argument("--min-edge", type=float, default=0.02, help="Minimum edge threshold (default: 0.02)")
    parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    # If just generating template, run a quick scan first to get games
    if args.template:
        logger.info("Generating odds template...")
        model, feature_names = load_model_and_features()
        matchups = get_daily_matchups(args.date)

        if not matchups:
            print(f"No games found for {args.date}")
            return

        # Create minimal predictions just to get team names
        predictions = [
            GamePrediction(
                away_team=m.away_abbrev,
                home_team=m.home_abbrev,
                away_pitcher=m.away_pitcher,
                home_pitcher=m.home_pitcher,
                away_runs=0, home_runs=0, total=0,
                away_win_prob=0, home_win_prob=0, tie_prob=0,
                game_time=m.game_time
            )
            for m in matchups
        ]

        generate_odds_template(predictions, args.template)
        return

    predictions, recommendations = run_daily_scan(
        date=args.date,
        api_key=args.api_key,
        use_odds=not args.no_odds,
        min_edge=args.min_edge,
        odds_file=args.odds_file,
        manual_odds_input=args.manual_odds
    )

    output = format_output(predictions, recommendations, args.date)
    print(output)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"\nOutput saved to {args.output}")


if __name__ == "__main__":
    main()
