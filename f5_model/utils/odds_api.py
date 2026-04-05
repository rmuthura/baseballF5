"""
The Odds API Integration.

Fetches betting odds from The Odds API (https://the-odds-api.com).
Free tier: 500 requests/month.

API Key required - set via environment variable ODDS_API_KEY or pass directly.

Note: The Odds API may not have F5-specific markets for all books.
This fetches available markets and we'll use what's available.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# API Configuration
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
MLB_SPORT = "baseball_mlb"

# Market keys for different bet types
MARKETS = {
    'h2h': 'h2h',           # Moneyline
    'spreads': 'spreads',    # Run line
    'totals': 'totals',      # Over/under
    # F5 specific (if available)
    'h2h_1st_5_innings': 'h2h_1st_5_innings',
    'spreads_1st_5_innings': 'spreads_1st_5_innings',
    'totals_1st_5_innings': 'totals_1st_5_innings',
}


@dataclass
class GameOdds:
    """Odds for a single game."""
    away_team: str
    home_team: str
    commence_time: str
    bookmaker: str
    # F5 markets (if available)
    f5_away_ml: Optional[int] = None
    f5_home_ml: Optional[int] = None
    f5_away_spread: Optional[float] = None
    f5_away_spread_odds: Optional[int] = None
    f5_home_spread: Optional[float] = None
    f5_home_spread_odds: Optional[int] = None
    f5_total: Optional[float] = None
    f5_over_odds: Optional[int] = None
    f5_under_odds: Optional[int] = None
    # Full game markets (fallback)
    fg_away_ml: Optional[int] = None
    fg_home_ml: Optional[int] = None
    fg_total: Optional[float] = None
    fg_over_odds: Optional[int] = None
    fg_under_odds: Optional[int] = None


def get_api_key() -> str:
    """Get API key from environment or raise error."""
    key = os.environ.get('ODDS_API_KEY')
    if not key:
        raise ValueError(
            "ODDS_API_KEY environment variable not set. "
            "Get a free API key at https://the-odds-api.com"
        )
    return key


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds."""
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1) * 100))
    else:
        return int(round(-100 / (decimal_odds - 1)))


def fetch_odds(
    api_key: str = None,
    bookmakers: List[str] = None,
    markets: List[str] = None,
    date: str = None
) -> List[Dict]:
    """
    Fetch MLB odds from The Odds API.

    Args:
        api_key: API key (uses env var if not provided)
        bookmakers: List of bookmaker keys (default: fanduel)
        markets: List of market keys to fetch
        date: Optional date filter (YYYY-MM-DD)

    Returns:
        List of game odds dictionaries
    """
    if api_key is None:
        api_key = get_api_key()

    if bookmakers is None:
        bookmakers = ['fanduel']

    if markets is None:
        # Try F5 markets first, fall back to full game
        markets = ['h2h', 'spreads', 'totals']

    url = f"{ODDS_API_BASE}/sports/{MLB_SPORT}/odds"

    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': ','.join(markets),
        'bookmakers': ','.join(bookmakers),
        'oddsFormat': 'american',
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()

        # Log remaining requests
        remaining = response.headers.get('x-requests-remaining', 'unknown')
        logger.info(f"Fetched odds. API requests remaining: {remaining}")

        return data

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise ValueError("Invalid API key")
        elif e.response.status_code == 429:
            raise ValueError("API rate limit exceeded")
        raise

    except Exception as e:
        logger.error(f"Error fetching odds: {e}")
        return []


def fetch_f5_odds(api_key: str = None) -> List[Dict]:
    """
    Attempt to fetch F5-specific odds.

    The Odds API may have F5 markets under alternate market keys.
    """
    if api_key is None:
        api_key = get_api_key()

    # Try different market endpoints
    f5_markets = [
        'h2h_1st_5_innings',
        'spreads_1st_5_innings',
        'totals_1st_5_innings',
        'alternate_spreads',
        'alternate_totals',
    ]

    all_odds = []

    for market in f5_markets:
        try:
            url = f"{ODDS_API_BASE}/sports/{MLB_SPORT}/odds"
            params = {
                'apiKey': api_key,
                'regions': 'us',
                'markets': market,
                'bookmakers': 'fanduel',
                'oddsFormat': 'american',
            }

            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data:
                    logger.info(f"Found {len(data)} games with {market} market")
                    all_odds.extend(data)

        except Exception as e:
            logger.debug(f"Market {market} not available: {e}")
            continue

    return all_odds


def parse_game_odds(game_data: Dict, bookmaker: str = 'fanduel') -> Optional[GameOdds]:
    """
    Parse odds data for a single game.

    Args:
        game_data: Raw game data from API
        bookmaker: Bookmaker to extract odds from

    Returns:
        GameOdds object or None
    """
    away_team = game_data.get('away_team', '')
    home_team = game_data.get('home_team', '')
    commence_time = game_data.get('commence_time', '')

    odds = GameOdds(
        away_team=away_team,
        home_team=home_team,
        commence_time=commence_time,
        bookmaker=bookmaker
    )

    # Find the specified bookmaker
    for bm in game_data.get('bookmakers', []):
        if bm.get('key') != bookmaker:
            continue

        for market in bm.get('markets', []):
            market_key = market.get('key', '')
            outcomes = market.get('outcomes', [])

            # Parse based on market type
            if market_key == 'h2h':
                for outcome in outcomes:
                    if outcome.get('name') == away_team:
                        odds.fg_away_ml = outcome.get('price')
                    elif outcome.get('name') == home_team:
                        odds.fg_home_ml = outcome.get('price')

            elif market_key == 'totals':
                for outcome in outcomes:
                    if outcome.get('name') == 'Over':
                        odds.fg_total = outcome.get('point')
                        odds.fg_over_odds = outcome.get('price')
                    elif outcome.get('name') == 'Under':
                        odds.fg_under_odds = outcome.get('price')

            elif market_key == 'spreads':
                for outcome in outcomes:
                    if outcome.get('name') == away_team:
                        odds.f5_away_spread = outcome.get('point')
                        odds.f5_away_spread_odds = outcome.get('price')
                    elif outcome.get('name') == home_team:
                        odds.f5_home_spread = outcome.get('point')
                        odds.f5_home_spread_odds = outcome.get('price')

            # F5 specific markets (if available)
            elif 'h2h_1st_5' in market_key or 'h2h_first_5' in market_key:
                for outcome in outcomes:
                    if outcome.get('name') == away_team:
                        odds.f5_away_ml = outcome.get('price')
                    elif outcome.get('name') == home_team:
                        odds.f5_home_ml = outcome.get('price')

            elif 'totals_1st_5' in market_key or 'totals_first_5' in market_key:
                for outcome in outcomes:
                    if outcome.get('name') == 'Over':
                        odds.f5_total = outcome.get('point')
                        odds.f5_over_odds = outcome.get('price')
                    elif outcome.get('name') == 'Under':
                        odds.f5_under_odds = outcome.get('price')

    return odds


def get_daily_odds(api_key: str = None, bookmaker: str = 'fanduel') -> Dict[str, GameOdds]:
    """
    Get all MLB odds for today, organized by matchup.

    Args:
        api_key: API key
        bookmaker: Bookmaker to use

    Returns:
        Dict mapping "AWAY @ HOME" to GameOdds
    """
    raw_odds = fetch_odds(api_key, bookmakers=[bookmaker])

    odds_by_game = {}

    for game in raw_odds:
        parsed = parse_game_odds(game, bookmaker)
        if parsed:
            key = f"{parsed.away_team} @ {parsed.home_team}"
            odds_by_game[key] = parsed

    logger.info(f"Parsed odds for {len(odds_by_game)} games")
    return odds_by_game


def check_api_status(api_key: str = None) -> Dict:
    """
    Check API key status and remaining requests.

    Returns:
        Dict with status info
    """
    if api_key is None:
        api_key = get_api_key()

    url = f"{ODDS_API_BASE}/sports"
    params = {'apiKey': api_key}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        return {
            'status': 'ok',
            'requests_remaining': response.headers.get('x-requests-remaining'),
            'requests_used': response.headers.get('x-requests-used'),
        }

    except Exception as e:
        return {'status': 'error', 'message': str(e)}


if __name__ == "__main__":
    import sys

    # Check API status
    try:
        status = check_api_status()
        print(f"API Status: {status}")

        if status['status'] == 'ok':
            print(f"\nFetching today's MLB odds...")
            odds = get_daily_odds()

            for matchup, game_odds in odds.items():
                print(f"\n{matchup}")
                if game_odds.fg_away_ml:
                    print(f"  ML: {game_odds.fg_away_ml} / {game_odds.fg_home_ml}")
                if game_odds.fg_total:
                    print(f"  Total: {game_odds.fg_total} (O: {game_odds.fg_over_odds} / U: {game_odds.fg_under_odds})")
                if game_odds.f5_away_ml:
                    print(f"  F5 ML: {game_odds.f5_away_ml} / {game_odds.f5_home_ml}")

    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo use The Odds API:")
        print("1. Get a free API key at https://the-odds-api.com")
        print("2. Set it: export ODDS_API_KEY='your-key-here'")
