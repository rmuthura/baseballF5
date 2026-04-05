"""
MLB Starting Lineup Scraper.

Scrapes starting lineups from MLB.com for a given date.
URL format: https://www.mlb.com/{team}/roster/starting-lineups/{date}

Also fetches the daily schedule to know which teams are playing.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# MLB team codes for URL construction
MLB_TEAMS = [
    'angels', 'astros', 'athletics', 'bluejays', 'braves', 'brewers',
    'cardinals', 'cubs', 'dbacks', 'dodgers', 'giants', 'guardians',
    'mariners', 'marlins', 'mets', 'nationals', 'orioles', 'padres',
    'phillies', 'pirates', 'rangers', 'rays', 'reds', 'redsox',
    'rockies', 'royals', 'tigers', 'twins', 'whitesox', 'yankees'
]

# Mapping from team slug to abbreviation
TEAM_ABBREV = {
    'angels': 'LAA', 'astros': 'HOU', 'athletics': 'OAK', 'bluejays': 'TOR',
    'braves': 'ATL', 'brewers': 'MIL', 'cardinals': 'STL', 'cubs': 'CHC',
    'dbacks': 'ARI', 'dodgers': 'LAD', 'giants': 'SF', 'guardians': 'CLE',
    'mariners': 'SEA', 'marlins': 'MIA', 'mets': 'NYM', 'nationals': 'WSH',
    'orioles': 'BAL', 'padres': 'SD', 'phillies': 'PHI', 'pirates': 'PIT',
    'rangers': 'TEX', 'rays': 'TB', 'reds': 'CIN', 'redsox': 'BOS',
    'rockies': 'COL', 'royals': 'KC', 'tigers': 'DET', 'twins': 'MIN',
    'whitesox': 'CWS', 'yankees': 'NYY'
}

ABBREV_TO_SLUG = {v: k for k, v in TEAM_ABBREV.items()}


@dataclass
class GameLineup:
    """Represents a single team's lineup for a game."""
    team: str
    team_abbrev: str
    pitcher: str
    pitcher_hand: str
    lineup: List[str]
    opponent: str
    opponent_abbrev: str
    is_home: bool
    game_time: str


@dataclass
class Matchup:
    """Represents a full game matchup."""
    away_team: str
    home_team: str
    away_abbrev: str
    home_abbrev: str
    away_pitcher: str
    home_pitcher: str
    away_pitcher_hand: str
    home_pitcher_hand: str
    away_lineup: List[str]
    home_lineup: List[str]
    game_time: str
    park: str


def get_schedule(date: str) -> List[Dict]:
    """
    Get MLB schedule for a given date using MLB Stats API.

    Args:
        date: Date string in YYYY-MM-DD format

    Returns:
        List of game dictionaries with team info
    """
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        games = []
        if data.get('dates') and len(data['dates']) > 0:
            for game in data['dates'][0].get('games', []):
                away = game['teams']['away']['team']
                home = game['teams']['home']['team']

                games.append({
                    'game_pk': game['gamePk'],
                    'game_time': game.get('gameDate', ''),
                    'away_team': away['name'],
                    'away_id': away['id'],
                    'home_team': home['name'],
                    'home_id': home['id'],
                    'venue': game.get('venue', {}).get('name', ''),
                    'status': game.get('status', {}).get('detailedState', '')
                })

        logger.info(f"Found {len(games)} games scheduled for {date}")
        return games

    except Exception as e:
        logger.error(f"Error fetching schedule: {e}")
        return []


def get_team_slug(team_name: str) -> Optional[str]:
    """Convert team name to URL slug."""
    name_lower = team_name.lower()

    # Direct mapping for common variations
    mappings = {
        'arizona diamondbacks': 'dbacks',
        'd-backs': 'dbacks',
        'los angeles angels': 'angels',
        'la angels': 'angels',
        'houston astros': 'astros',
        'oakland athletics': 'athletics',
        'toronto blue jays': 'bluejays',
        'atlanta braves': 'braves',
        'milwaukee brewers': 'brewers',
        'st. louis cardinals': 'cardinals',
        'chicago cubs': 'cubs',
        'los angeles dodgers': 'dodgers',
        'la dodgers': 'dodgers',
        'san francisco giants': 'giants',
        'cleveland guardians': 'guardians',
        'seattle mariners': 'mariners',
        'miami marlins': 'marlins',
        'new york mets': 'mets',
        'washington nationals': 'nationals',
        'baltimore orioles': 'orioles',
        'san diego padres': 'padres',
        'philadelphia phillies': 'phillies',
        'pittsburgh pirates': 'pirates',
        'texas rangers': 'rangers',
        'tampa bay rays': 'rays',
        'cincinnati reds': 'reds',
        'boston red sox': 'redsox',
        'colorado rockies': 'rockies',
        'kansas city royals': 'royals',
        'detroit tigers': 'tigers',
        'minnesota twins': 'twins',
        'chicago white sox': 'whitesox',
        'new york yankees': 'yankees',
    }

    if name_lower in mappings:
        return mappings[name_lower]

    # Try to find partial match
    for full_name, slug in mappings.items():
        if slug in name_lower or name_lower in full_name:
            return slug

    return None


def scrape_lineup_page(team_slug: str, date: str) -> Optional[GameLineup]:
    """
    Scrape a team's starting lineup page.

    Args:
        team_slug: Team URL slug (e.g., 'yankees')
        date: Date string in YYYY-MM-DD format

    Returns:
        GameLineup object or None if not available
    """
    url = f"https://www.mlb.com/{team_slug}/roster/starting-lineups/{date}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            logger.debug(f"No lineup page for {team_slug}: HTTP {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find lineup data - MLB.com structure varies
        # Look for player names in lineup cards
        lineup = []
        pitcher = None
        pitcher_hand = 'R'

        # Find lineup container
        lineup_card = soup.find('div', class_=re.compile(r'lineup', re.I))
        if not lineup_card:
            lineup_card = soup.find('section', class_=re.compile(r'lineup', re.I))

        if lineup_card:
            # Find player links/names
            player_links = lineup_card.find_all('a', href=re.compile(r'/player/'))
            for link in player_links[:9]:  # Max 9 batters
                name = link.get_text(strip=True)
                if name:
                    lineup.append(name)

            # Find pitcher
            pitcher_section = soup.find(text=re.compile(r'Starting Pitcher|SP:', re.I))
            if pitcher_section:
                parent = pitcher_section.find_parent()
                if parent:
                    pitcher_link = parent.find('a', href=re.compile(r'/player/'))
                    if pitcher_link:
                        pitcher = pitcher_link.get_text(strip=True)

        if not lineup:
            logger.debug(f"Could not parse lineup for {team_slug}")
            return None

        return GameLineup(
            team=team_slug,
            team_abbrev=TEAM_ABBREV.get(team_slug, team_slug.upper()),
            pitcher=pitcher or "TBD",
            pitcher_hand=pitcher_hand,
            lineup=lineup,
            opponent="",
            opponent_abbrev="",
            is_home=False,
            game_time=""
        )

    except Exception as e:
        logger.error(f"Error scraping {team_slug}: {e}")
        return None


def get_lineups_from_statsapi(date: str) -> List[Matchup]:
    """
    Get lineups using MLB Stats API (more reliable than scraping).

    Args:
        date: Date string in YYYY-MM-DD format

    Returns:
        List of Matchup objects
    """
    # First get schedule
    schedule = get_schedule(date)
    matchups = []

    for game in schedule:
        game_pk = game['game_pk']

        # Get game feed for lineups
        feed_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

        try:
            response = requests.get(feed_url, timeout=10)
            if response.status_code != 200:
                continue

            data = response.json()
            game_data = data.get('gameData', {})
            live_data = data.get('liveData', {})

            # Get probable pitchers
            probables = game_data.get('probablePitchers', {})
            away_pitcher = probables.get('away', {}).get('fullName', 'TBD')
            home_pitcher = probables.get('home', {}).get('fullName', 'TBD')

            # Get pitcher handedness
            players = game_data.get('players', {})
            away_hand = 'R'
            home_hand = 'R'

            away_pitcher_id = probables.get('away', {}).get('id')
            home_pitcher_id = probables.get('home', {}).get('id')

            if away_pitcher_id:
                player_key = f"ID{away_pitcher_id}"
                if player_key in players:
                    away_hand = players[player_key].get('pitchHand', {}).get('code', 'R')

            if home_pitcher_id:
                player_key = f"ID{home_pitcher_id}"
                if player_key in players:
                    home_hand = players[player_key].get('pitchHand', {}).get('code', 'R')

            # Get lineups from boxscore
            boxscore = live_data.get('boxscore', {})
            teams_box = boxscore.get('teams', {})

            away_lineup = []
            home_lineup = []

            # Away batting order
            away_batters = teams_box.get('away', {}).get('battingOrder', [])
            for batter_id in away_batters[:9]:
                player_key = f"ID{batter_id}"
                if player_key in players:
                    away_lineup.append(players[player_key].get('fullName', ''))

            # Home batting order
            home_batters = teams_box.get('home', {}).get('battingOrder', [])
            for batter_id in home_batters[:9]:
                player_key = f"ID{batter_id}"
                if player_key in players:
                    home_lineup.append(players[player_key].get('fullName', ''))

            # Get team abbreviations
            teams = game_data.get('teams', {})
            away_abbrev = teams.get('away', {}).get('abbreviation', '')
            home_abbrev = teams.get('home', {}).get('abbreviation', '')

            # Get venue for park factor
            venue = game_data.get('venue', {}).get('name', '')

            matchup = Matchup(
                away_team=game['away_team'],
                home_team=game['home_team'],
                away_abbrev=away_abbrev,
                home_abbrev=home_abbrev,
                away_pitcher=away_pitcher,
                home_pitcher=home_pitcher,
                away_pitcher_hand=away_hand,
                home_pitcher_hand=home_hand,
                away_lineup=away_lineup,
                home_lineup=home_lineup,
                game_time=game['game_time'],
                park=home_abbrev  # Use home team as park code
            )

            matchups.append(matchup)
            logger.info(f"  {away_abbrev} @ {home_abbrev}: {away_pitcher} vs {home_pitcher}")

        except Exception as e:
            logger.error(f"Error fetching game {game_pk}: {e}")
            continue

    return matchups


def get_daily_matchups(date: str) -> List[Matchup]:
    """
    Get all matchups for a given date with lineups.

    This is the main function to call.

    Args:
        date: Date string in YYYY-MM-DD format

    Returns:
        List of Matchup objects
    """
    logger.info(f"Fetching matchups for {date}...")
    return get_lineups_from_statsapi(date)


if __name__ == "__main__":
    import sys

    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")

    print(f"\nFetching lineups for {date}...\n")
    matchups = get_daily_matchups(date)

    print(f"\nFound {len(matchups)} matchups:\n")
    for m in matchups:
        print(f"{m.away_abbrev} @ {m.home_abbrev}")
        print(f"  Pitchers: {m.away_pitcher} ({m.away_pitcher_hand}) vs {m.home_pitcher} ({m.home_pitcher_hand})")
        if m.away_lineup:
            print(f"  Away lineup: {', '.join(m.away_lineup[:3])}...")
        if m.home_lineup:
            print(f"  Home lineup: {', '.join(m.home_lineup[:3])}...")
        print()
