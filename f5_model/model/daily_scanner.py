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

    # With full manual input (pitcher IDs, lineup IDs, odds) from CSV
    python -m f5_model.model.daily_scanner --date 2026-04-04 --games-file games.csv

    # Interactive mode - prompt for odds
    python -m f5_model.model.daily_scanner --date 2026-04-04 --manual-odds

CSV Format for odds file (odds only, lineups from MLB API):
    away_team,home_team,away_ml,home_ml,total,over_odds,under_odds
    NYY,BOS,-150,+130,4.5,-110,-110
    LAD,SF,+120,-140,5.0,-105,-115

CSV Format for games file (full manual input with MLB IDs):
    away_team,home_team,away_pitcher_id,away_pitcher_hand,home_pitcher_id,home_pitcher_hand,away_lineup_ids,home_lineup_ids,away_ml,home_ml,total,over_odds,under_odds
    CIN,TEX,669270,R,666201,R,"669016,663460,683658,664702,672284,608369,673357,670868,687093","543760,608369,686780,683002,666971,683734,682998,681082,660670",-110,-110,4.5,-110,-110

Note: Lineup IDs should be comma-separated MLB player IDs in batting order, quoted if using commas.
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


@dataclass
class ManualGame:
    """Full game input with pitcher IDs, lineup IDs, and odds."""
    away_team: str
    home_team: str
    away_pitcher_id: int
    away_pitcher_hand: str
    home_pitcher_id: int
    home_pitcher_hand: str
    away_lineup_ids: List[int]
    home_lineup_ids: List[int]
    # Optional odds
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


def parse_games_csv(filepath: str) -> List[ManualGame]:
    """
    Parse full game inputs from a CSV file.

    Expected columns:
        away_team, home_team, away_pitcher_id, away_pitcher_hand,
        home_pitcher_id, home_pitcher_hand, away_lineup_ids, home_lineup_ids,
        away_ml, home_ml, total, over_odds, under_odds

    Lineup IDs should be comma-separated MLB player IDs.

    Returns:
        List of ManualGame objects
    """
    games = []

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                away = row['away_team'].strip().upper()
                home = row['home_team'].strip().upper()

                # Parse lineup IDs (comma-separated within the field)
                away_lineup_str = row.get('away_lineup_ids', '').strip()
                home_lineup_str = row.get('home_lineup_ids', '').strip()

                away_lineup_ids = [int(x.strip()) for x in away_lineup_str.split(',') if x.strip()]
                home_lineup_ids = [int(x.strip()) for x in home_lineup_str.split(',') if x.strip()]

                game = ManualGame(
                    away_team=away,
                    home_team=home,
                    away_pitcher_id=int(row['away_pitcher_id']),
                    away_pitcher_hand=row.get('away_pitcher_hand', 'R').strip().upper(),
                    home_pitcher_id=int(row['home_pitcher_id']),
                    home_pitcher_hand=row.get('home_pitcher_hand', 'R').strip().upper(),
                    away_lineup_ids=away_lineup_ids,
                    home_lineup_ids=home_lineup_ids,
                    away_ml=int(row['away_ml']) if row.get('away_ml') else None,
                    home_ml=int(row['home_ml']) if row.get('home_ml') else None,
                    total=float(row['total']) if row.get('total') else None,
                    over_odds=int(row['over_odds']) if row.get('over_odds') else None,
                    under_odds=int(row['under_odds']) if row.get('under_odds') else None
                )

                games.append(game)
                logger.info(f"  Loaded game: {away} @ {home}")

            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid row: {row} - {e}")
                continue

    return games


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
    # Data quality tracking
    confidence: float = 1.0  # 0-1, percentage of players with features
    players_found: int = 0
    players_total: int = 0


@dataclass
class BetRecommendation:
    """A single bet recommendation."""
    game: str
    market: str
    pick: str
    model_prob: float
    model_odds: int
    book_odds: Optional[int]
    raw_edge: float      # Edge before confidence adjustment
    adj_edge: float      # Edge after confidence discount
    ev: float            # Expected value per $100 bet
    confidence: float    # Data quality score (0-1)


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

        # Calculate confidence based on data quality
        # Total players: 2 pitchers + batters in both lineups
        players_found = (
            (1 if away_result.get('pitcher_found', False) else 0) +
            (1 if home_result.get('pitcher_found', False) else 0) +
            away_result.get('lineup_batters_found', 0) +
            home_result.get('lineup_batters_found', 0)
        )
        players_total = (
            2 +  # Both pitchers
            away_result.get('lineup_total', len(away_lineup_ids)) +
            home_result.get('lineup_total', len(home_lineup_ids))
        )
        confidence = players_found / players_total if players_total > 0 else 0

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
            edges={'probs': probs},
            confidence=confidence,
            players_found=players_found,
            players_total=players_total
        )

    except Exception as e:
        logger.error(f"Error predicting {matchup.away_abbrev} @ {matchup.home_abbrev}: {e}")
        return None


def predict_manual_game(
    model,
    feature_names: List[str],
    game: ManualGame,
    date: str,
    park: str = None
) -> Optional[GamePrediction]:
    """
    Run prediction for a manually specified game (with MLB IDs).

    Args:
        model: Trained model
        feature_names: Feature names
        game: ManualGame object with pitcher IDs and lineup IDs
        date: Game date
        park: Park code (defaults to home team)

    Returns:
        GamePrediction or None if can't predict
    """
    if len(game.away_lineup_ids) < 5 or len(game.home_lineup_ids) < 5:
        logger.warning(f"Skipping {game.away_team} @ {game.home_team}: not enough batters in lineup")
        return None

    park_code = park or game.home_team

    try:
        # Away team runs = Home pitcher vs Away lineup
        away_result = predict_f5_runs(
            model=model,
            feature_names=feature_names,
            pitcher_id=game.home_pitcher_id,
            pitcher_hand=game.home_pitcher_hand,
            lineup_ids=game.away_lineup_ids,
            date=date,
            starter_is_home=True,
            park=park_code
        )

        # Home team runs = Away pitcher vs Home lineup
        home_result = predict_f5_runs(
            model=model,
            feature_names=feature_names,
            pitcher_id=game.away_pitcher_id,
            pitcher_hand=game.away_pitcher_hand,
            lineup_ids=game.home_lineup_ids,
            date=date,
            starter_is_home=False,
            park=park_code
        )

        away_lambda = away_result['predicted_runs']
        home_lambda = home_result['predicted_runs']

        probs = compute_game_probs(away_lambda, home_lambda)

        # Calculate confidence based on data quality
        players_found = (
            (1 if away_result.get('pitcher_found', False) else 0) +
            (1 if home_result.get('pitcher_found', False) else 0) +
            away_result.get('lineup_batters_found', 0) +
            home_result.get('lineup_batters_found', 0)
        )
        players_total = (
            2 +  # Both pitchers
            len(game.away_lineup_ids) +
            len(game.home_lineup_ids)
        )
        confidence = players_found / players_total if players_total > 0 else 0

        return GamePrediction(
            away_team=game.away_team,
            home_team=game.home_team,
            away_pitcher=f"ID:{game.away_pitcher_id}",
            home_pitcher=f"ID:{game.home_pitcher_id}",
            away_runs=away_lambda,
            home_runs=home_lambda,
            total=away_lambda + home_lambda,
            away_win_prob=probs['away_ml'],
            home_win_prob=probs['home_ml'],
            tie_prob=probs['tie'],
            game_time="",
            edges={'probs': probs},
            confidence=confidence,
            players_found=players_found,
            players_total=players_total
        )

    except Exception as e:
        logger.error(f"Error predicting {game.away_team} @ {game.home_team}: {e}")
        return None


def find_edges(
    predictions: List[GamePrediction],
    odds: Dict[str, 'GameOdds'],
    min_edge: float = 0.02
) -> List[BetRecommendation]:
    """
    Find betting edges by comparing model to book odds.

    Edges are discounted by data quality (confidence) - a 10% raw edge
    with 60% data quality becomes a 6% adjusted edge.

    Args:
        predictions: List of game predictions
        odds: Dict of game odds from API
        min_edge: Minimum RAW edge to include (default 2%)

    Returns:
        List of bet recommendations sorted by ADJUSTED edge
    """
    recommendations = []

    for pred in predictions:
        game_key = f"{pred.away_team} @ {pred.home_team}"
        probs = pred.edges.get('probs', {})
        confidence = pred.confidence

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
            raw_edge = model_prob - book_prob
            adj_edge = raw_edge * confidence

            if raw_edge >= min_edge:
                recommendations.append(BetRecommendation(
                    game=game_key,
                    market="F5 ML",
                    pick=pred.away_team,
                    model_prob=model_prob,
                    model_odds=prob_to_american(model_prob),
                    book_odds=game_odds.fg_away_ml,
                    raw_edge=raw_edge,
                    adj_edge=adj_edge,
                    ev=calculate_ev(model_prob, game_odds.fg_away_ml),
                    confidence=confidence
                ))

        if game_odds.fg_home_ml:
            model_prob = probs.get('home_ml', 0)
            book_prob = american_to_prob(game_odds.fg_home_ml)
            raw_edge = model_prob - book_prob
            adj_edge = raw_edge * confidence

            if raw_edge >= min_edge:
                recommendations.append(BetRecommendation(
                    game=game_key,
                    market="F5 ML",
                    pick=pred.home_team,
                    model_prob=model_prob,
                    model_odds=prob_to_american(model_prob),
                    book_odds=game_odds.fg_home_ml,
                    raw_edge=raw_edge,
                    adj_edge=adj_edge,
                    ev=calculate_ev(model_prob, game_odds.fg_home_ml),
                    confidence=confidence
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
                    raw_edge = model_prob - book_prob
                    adj_edge = raw_edge * confidence

                    if raw_edge >= min_edge:
                        recommendations.append(BetRecommendation(
                            game=game_key,
                            market="F5 Total",
                            pick=f"Over {total_line}",
                            model_prob=model_prob,
                            model_odds=prob_to_american(model_prob),
                            book_odds=game_odds.fg_over_odds,
                            raw_edge=raw_edge,
                            adj_edge=adj_edge,
                            ev=calculate_ev(model_prob, game_odds.fg_over_odds),
                            confidence=confidence
                        ))

                    # Under
                    if game_odds.fg_under_odds:
                        model_prob = probs_dict['under']
                        book_prob = american_to_prob(game_odds.fg_under_odds)
                        raw_edge = model_prob - book_prob
                        adj_edge = raw_edge * confidence

                        if raw_edge >= min_edge:
                            recommendations.append(BetRecommendation(
                                game=game_key,
                                market="F5 Total",
                                pick=f"Under {total_line}",
                                model_prob=model_prob,
                                model_odds=prob_to_american(model_prob),
                                book_odds=game_odds.fg_under_odds,
                                raw_edge=raw_edge,
                                adj_edge=adj_edge,
                                ev=calculate_ev(model_prob, game_odds.fg_under_odds),
                                confidence=confidence
                            ))

    # Sort by ADJUSTED edge descending
    recommendations.sort(key=lambda x: x.adj_edge, reverse=True)

    return recommendations


def format_output(
    predictions: List[GamePrediction],
    recommendations: List[BetRecommendation],
    date: str,
    min_confidence: float = 0.75
) -> str:
    """Format the daily scanner output."""
    lines = []

    lines.append("=" * 80)
    lines.append(f"F5 DAILY SCANNER - {date}")
    lines.append("=" * 80)

    # Data Quality Summary - sorted by confidence
    lines.append(f"\n{'=' * 80}")
    lines.append("DATA QUALITY BY GAME")
    lines.append("=" * 80)

    sorted_by_conf = sorted(predictions, key=lambda x: x.confidence, reverse=True)
    for pred in sorted_by_conf:
        pct = pred.confidence * 100
        found = pred.players_found
        total = pred.players_total

        if pct >= 90:
            icon = "🟢"
        elif pct >= 75:
            icon = "🟡"
        else:
            icon = "🔴"

        status = ""
        if pct < min_confidence * 100:
            status = " ⚠️ LOW DATA"

        lines.append(f"  {icon} {pred.away_team} @ {pred.home_team}: {pct:.0f}% ({found}/{total}){status}")

    # Summary of all games - sorted by confidence
    lines.append(f"\n{'=' * 80}")
    lines.append("ALL GAMES - MODEL PREDICTIONS (sorted by data quality)")
    lines.append("=" * 80)
    lines.append(f"\n  {'Game':<15} {'Data':<10} {'Pitchers':<30} {'Score':<10} {'Total':<7} {'Fav':<10}")
    lines.append("  " + "-" * 90)

    for pred in sorted_by_conf:
        pitchers = f"{pred.away_pitcher[:12]} vs {pred.home_pitcher[:12]}"
        score = f"{pred.away_runs:.1f} - {pred.home_runs:.1f}"
        data_qual = f"{pred.confidence*100:.0f}%"

        if pred.confidence < min_confidence:
            data_qual += " ⚠️"

        if pred.away_win_prob > pred.home_win_prob:
            fav = f"{pred.away_team} {prob_to_american(pred.away_win_prob)}"
        else:
            fav = f"{pred.home_team} {prob_to_american(pred.home_win_prob)}"

        lines.append(f"  {pred.away_team} @ {pred.home_team:<8} {data_qual:<10} {pitchers:<30} {score:<10} {pred.total:<7.1f} {fav:<10}")

    # Top edges
    if recommendations:
        lines.append(f"\n{'=' * 80}")
        lines.append("TOP EDGES vs FANDUEL (sorted by adjusted edge)")
        lines.append("=" * 80)
        lines.append(f"\n  {'Game':<18} {'Market':<10} {'Pick':<12} {'Model':<7} {'Book':<7} {'Raw':<7} {'Adj':<7} {'Data':<6} {'EV':<8}")
        lines.append("  " + "-" * 95)

        for rec in recommendations[:15]:  # Top 15
            model_odds = f"{rec.model_odds:+d}" if rec.model_odds else "N/A"
            book_odds = f"{rec.book_odds:+d}" if rec.book_odds else "N/A"
            raw_str = f"{rec.raw_edge:.1%}"
            adj_str = f"{rec.adj_edge:.1%}"
            data_str = f"{rec.confidence*100:.0f}%"
            ev_str = f"${rec.ev:+.1f}"

            # Mark strong ADJUSTED edges
            if rec.adj_edge >= 0.05:
                adj_str += " **"
            elif rec.adj_edge >= 0.03:
                adj_str += " *"

            # Warn on low data
            if rec.confidence < min_confidence:
                data_str += "⚠️"

            lines.append(f"  {rec.game:<18} {rec.market:<10} {rec.pick:<12} {model_odds:<7} {book_odds:<7} {raw_str:<7} {adj_str:<7} {data_str:<6} {ev_str:<8}")

        lines.append("\n  Legend: ** = 5%+ adj edge (strong), * = 3-5% adj edge, ⚠️ = low data quality")
        lines.append("  Raw = edge before data discount, Adj = edge × data quality %")
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

    # Best bets summary - only show if ADJUSTED edge >= 5%
    if recommendations:
        strong_edges = [r for r in recommendations if r.adj_edge >= 0.05]
        if strong_edges:
            lines.append(f"\n{'=' * 80}")
            lines.append("BEST BETS (5%+ Adjusted Edge)")
            lines.append("=" * 80)
            for rec in strong_edges[:5]:
                data_warning = " ⚠️ LOW DATA" if rec.confidence < min_confidence else ""
                lines.append(f"\n  {rec.pick} ({rec.market}){data_warning}")
                lines.append(f"    Game: {rec.game}")
                lines.append(f"    Data Quality: {rec.confidence*100:.0f}%")
                lines.append(f"    Model: {rec.model_prob:.1%} ({rec.model_odds:+d})")
                lines.append(f"    Book:  {american_to_prob(rec.book_odds):.1%} ({rec.book_odds:+d})")
                lines.append(f"    Raw Edge:  {rec.raw_edge:.1%}")
                lines.append(f"    Adj Edge:  {rec.adj_edge:.1%}")
                lines.append(f"    EV:    ${rec.ev:+.1f} per $100")

    return "\n".join(lines)


def run_daily_scan(
    date: str,
    api_key: str = None,
    use_odds: bool = True,
    min_edge: float = 0.02,
    min_confidence: float = 0.75,
    odds_file: str = None,
    games_file: str = None,
    manual_odds_input: bool = False
) -> Tuple[List[GamePrediction], List[BetRecommendation]]:
    """
    Run the full daily scan.

    Args:
        date: Date to scan (YYYY-MM-DD)
        api_key: Odds API key
        use_odds: Whether to fetch odds from API
        min_edge: Minimum edge threshold
        min_confidence: Minimum data quality to include in edge finding (default 75%)
        odds_file: Path to CSV file with odds only
        games_file: Path to CSV file with full game input (IDs + odds)
        manual_odds_input: Whether to prompt for manual odds

    Returns:
        Tuple of (predictions, recommendations)
    """
    logger.info(f"Starting daily scan for {date}")

    # Load model
    logger.info("Loading model...")
    model, feature_names = load_model_and_features()

    # Check if using manual games file (full control with MLB IDs)
    if games_file:
        logger.info(f"Loading games from {games_file}...")
        manual_games = parse_games_csv(games_file)
        logger.info(f"Loaded {len(manual_games)} games from CSV")

        if not manual_games:
            logger.warning("No valid games in CSV file")
            return [], []

        # Run predictions from manual games
        logger.info("Running predictions...")
        predictions = []
        odds = {}

        for game in manual_games:
            logger.info(f"  Predicting {game.away_team} @ {game.home_team}...")
            pred = predict_manual_game(model, feature_names, game, date)
            if pred:
                predictions.append(pred)
                conf_pct = pred.confidence * 100
                conf_warn = " ⚠️ LOW DATA" if pred.confidence < min_confidence else ""
                logger.info(f"    Data quality: {conf_pct:.0f}% ({pred.players_found}/{pred.players_total}){conf_warn}")

            # Build odds dict from manual games
            if game.away_ml or game.home_ml or game.total:
                key = f"{game.away_team} @ {game.home_team}"
                odds[key] = type('GameOdds', (), {
                    'fg_away_ml': game.away_ml,
                    'fg_home_ml': game.home_ml,
                    'fg_total': game.total,
                    'fg_over_odds': game.over_odds,
                    'fg_under_odds': game.under_odds
                })()

        logger.info(f"Successfully predicted {len(predictions)} games")

        # Log data quality summary
        high_conf = [p for p in predictions if p.confidence >= min_confidence]
        low_conf = [p for p in predictions if p.confidence < min_confidence]
        logger.info(f"Data quality: {len(high_conf)} games >= {min_confidence*100:.0f}%, {len(low_conf)} games below threshold")

        # Find edges (only on games meeting minimum confidence for recommendations)
        recommendations = []
        if odds and predictions:
            # Filter to high-confidence games for edge finding
            high_conf_preds = [p for p in predictions if p.confidence >= min_confidence]
            recommendations = find_edges(high_conf_preds, odds, min_edge)
            logger.info(f"Found {len(recommendations)} edges >= {min_edge:.0%} (from {len(high_conf_preds)} high-data games)")

        return predictions, recommendations

    # Standard flow: fetch from MLB API
    logger.info("Fetching lineups from MLB API...")
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
            conf_pct = pred.confidence * 100
            conf_warn = " ⚠️ LOW DATA" if pred.confidence < min_confidence else ""
            logger.info(f"    Data quality: {conf_pct:.0f}% ({pred.players_found}/{pred.players_total}){conf_warn}")

    logger.info(f"Successfully predicted {len(predictions)} games")

    # Log data quality summary
    high_conf = [p for p in predictions if p.confidence >= min_confidence]
    low_conf = [p for p in predictions if p.confidence < min_confidence]
    logger.info(f"Data quality: {len(high_conf)} games >= {min_confidence*100:.0f}%, {len(low_conf)} games below threshold")

    # Find edges (only on games meeting minimum confidence for recommendations)
    recommendations = []
    if odds and predictions:
        # Filter to high-confidence games for edge finding
        high_conf_preds = [p for p in predictions if p.confidence >= min_confidence]
        recommendations = find_edges(high_conf_preds, odds, min_edge)
        logger.info(f"Found {len(recommendations)} edges >= {min_edge:.0%} (from {len(high_conf_preds)} high-data games)")

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


def generate_games_template(filepath: str):
    """Generate a CSV template for full manual game input."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'away_team', 'home_team',
            'away_pitcher_id', 'away_pitcher_hand',
            'home_pitcher_id', 'home_pitcher_hand',
            'away_lineup_ids', 'home_lineup_ids',
            'away_ml', 'home_ml', 'total', 'over_odds', 'under_odds'
        ])
        # Example row
        writer.writerow([
            'CIN', 'TEX',
            '669270', 'R',
            '666201', 'R',
            '669016,663460,683658,664702,672284,608369,673357,670868,687093',
            '543760,608369,686780,683002,666971,683734,682998,681082,660670',
            '-110', '-110', '4.5', '-110', '-110'
        ])

    print(f"Games template saved to {filepath}")
    print("\nColumns:")
    print("  away_team, home_team: Team abbreviations (CIN, TEX, etc.)")
    print("  away_pitcher_id, home_pitcher_id: MLB player IDs")
    print("  away_pitcher_hand, home_pitcher_hand: L or R")
    print("  away_lineup_ids, home_lineup_ids: Comma-separated MLB IDs in batting order")
    print("  away_ml, home_ml, total, over_odds, under_odds: FanDuel F5 odds (optional)")
    print("\nLook up MLB IDs at: https://baseballsavant.mlb.com/")


def main():
    parser = argparse.ArgumentParser(
        description="Daily F5 Scanner - Find betting edges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-fetch lineups, predictions only (no odds)
    python -m f5_model.model.daily_scanner --date 2026-04-04 --no-odds

    # Auto-fetch lineups, with manual odds from CSV
    python -m f5_model.model.daily_scanner --date 2026-04-04 --odds-file odds.csv

    # RECOMMENDED: Full manual input with MLB IDs (most reliable)
    python -m f5_model.model.daily_scanner --date 2026-04-04 --games-file games.csv

    # Generate templates for manual input
    python -m f5_model.model.daily_scanner --template odds.csv
    python -m f5_model.model.daily_scanner --games-template games.csv
        """
    )
    parser.add_argument("--date", "-d", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date to scan (YYYY-MM-DD)")
    parser.add_argument("--api-key", help="Odds API key (or set ODDS_API_KEY env var)")
    parser.add_argument("--no-odds", action="store_true", help="Skip fetching odds (model only)")
    parser.add_argument("--odds-file", help="CSV file with FanDuel odds (uses MLB API for lineups)")
    parser.add_argument("--games-file", help="CSV file with full game input (pitcher IDs, lineup IDs, odds)")
    parser.add_argument("--manual-odds", action="store_true", help="Enter odds interactively")
    parser.add_argument("--template", help="Generate CSV template for odds-only input")
    parser.add_argument("--games-template", help="Generate CSV template for full game input with IDs")
    parser.add_argument("--min-edge", type=float, default=0.02, help="Minimum edge threshold (default: 0.02)")
    parser.add_argument("--min-confidence", type=float, default=0.75,
                        help="Minimum data quality to include in edges (default: 0.75 = 75%%)")
    parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    # Generate full games template
    if args.games_template:
        generate_games_template(args.games_template)
        return

    # Generate odds-only template
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
        min_confidence=args.min_confidence,
        odds_file=args.odds_file,
        games_file=args.games_file,
        manual_odds_input=args.manual_odds
    )

    output = format_output(predictions, recommendations, args.date, min_confidence=args.min_confidence)
    print(output)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"\nOutput saved to {args.output}")


if __name__ == "__main__":
    main()
