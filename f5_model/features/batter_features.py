"""
Batter feature engineering for F5 runs prediction model.

Computes season-to-date and rolling window features for each batter
on each game date, split by pitcher handedness (platoon splits).
Uses only data available BEFORE that game (no leakage).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from f5_model.utils.statcast_pull import load_all_raw_data, get_processed_data_dir
from f5_model.utils.f5_processor import filter_f5

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_season(date: str) -> int:
    """Extract season year from date string."""
    return int(str(date)[:4])


def identify_batter_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns identifying batter event types from the 'events' column.
    """
    df = df.copy()

    # Plate appearances (events that end an at-bat)
    df['is_pa'] = df['events'].notna()

    # At bats (PA minus walks, HBP, sac flies, sac bunts)
    non_ab_events = ['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt', 'sac_fly_double_play',
                     'catcher_interf', 'intent_walk']
    df['is_ab'] = df['events'].notna() & ~df['events'].isin(non_ab_events)

    # Hits
    df['is_single'] = df['events'] == 'single'
    df['is_double'] = df['events'] == 'double'
    df['is_triple'] = df['events'] == 'triple'
    df['is_hr'] = df['events'] == 'home_run'
    df['is_hit'] = df['events'].isin(['single', 'double', 'triple', 'home_run'])

    # Strikeouts and walks
    df['is_strikeout'] = df['events'].isin(['strikeout', 'strikeout_double_play'])
    df['is_walk'] = df['events'].isin(['walk', 'intent_walk'])
    df['is_hbp'] = df['events'] == 'hit_by_pitch'

    # Balls in play
    df['is_bip'] = df['launch_speed'].notna()

    # Ground balls (launch angle < 10)
    df['is_gb'] = df['is_bip'] & (df['launch_angle'] < 10)

    # Barrels (high exit velo + optimal launch angle)
    df['is_barrel'] = (df['launch_speed'] >= 98) & (df['launch_angle'] >= 26) & (df['launch_angle'] <= 30)

    return df


def compute_batter_game_stats(game_pitches: pd.DataFrame) -> Dict:
    """
    Compute stats for a single game's worth of plate appearances for one batter.

    Args:
        game_pitches: DataFrame of pitches for one batter in one game

    Returns:
        Dictionary of game-level stats
    """
    if len(game_pitches) == 0:
        return {}

    stats = {}

    # Basic counts
    stats['pa'] = game_pitches['is_pa'].sum()
    stats['ab'] = game_pitches['is_ab'].sum()

    # Hits
    stats['singles'] = game_pitches['is_single'].sum()
    stats['doubles'] = game_pitches['is_double'].sum()
    stats['triples'] = game_pitches['is_triple'].sum()
    stats['hrs'] = game_pitches['is_hr'].sum()
    stats['hits'] = game_pitches['is_hit'].sum()

    # Outs and other
    stats['strikeouts'] = game_pitches['is_strikeout'].sum()
    stats['walks'] = game_pitches['is_walk'].sum()
    stats['hbp'] = game_pitches['is_hbp'].sum()

    # Batted ball stats
    stats['bip'] = game_pitches['is_bip'].sum()
    stats['gb'] = game_pitches['is_gb'].sum()
    stats['barrels'] = game_pitches['is_barrel'].sum()

    # Exit velocity (on contact)
    contact = game_pitches[game_pitches['launch_speed'].notna()]
    stats['total_exit_velo'] = contact['launch_speed'].sum() if len(contact) > 0 else 0
    stats['exit_velo_count'] = len(contact)

    # wOBA value (from Statcast)
    woba_pitches = game_pitches[game_pitches['woba_value'].notna()]
    stats['woba_value_sum'] = woba_pitches['woba_value'].sum()
    stats['woba_denom_sum'] = game_pitches['woba_denom'].sum()

    # xwOBA (expected wOBA from Statcast)
    xwoba_pitches = game_pitches[game_pitches['estimated_woba_using_speedangle'].notna()]
    stats['xwoba_sum'] = xwoba_pitches['estimated_woba_using_speedangle'].sum()
    stats['xwoba_count'] = len(xwoba_pitches)

    return stats


def compute_season_batter_stats(games_df: pd.DataFrame) -> Dict:
    """
    Aggregate game-level stats into season-level batter features.

    Args:
        games_df: DataFrame with one row per game, containing game-level stats

    Returns:
        Dictionary of season-level features
    """
    if len(games_df) == 0:
        return {}

    features = {}

    # Totals
    total_pa = games_df['pa'].sum()
    total_ab = games_df['ab'].sum()
    total_hits = games_df['hits'].sum()
    total_singles = games_df['singles'].sum()
    total_doubles = games_df['doubles'].sum()
    total_triples = games_df['triples'].sum()
    total_hrs = games_df['hrs'].sum()
    total_walks = games_df['walks'].sum()
    total_hbp = games_df['hbp'].sum()
    total_strikeouts = games_df['strikeouts'].sum()
    total_bip = games_df['bip'].sum()

    # Batting average
    features['avg'] = total_hits / total_ab if total_ab > 0 else np.nan

    # On-base percentage
    # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
    # Simplified: (H + BB + HBP) / PA
    features['obp'] = (total_hits + total_walks + total_hbp) / total_pa if total_pa > 0 else np.nan

    # Slugging percentage
    total_bases = total_singles + 2*total_doubles + 3*total_triples + 4*total_hrs
    features['slg'] = total_bases / total_ab if total_ab > 0 else np.nan

    # OPS
    features['ops'] = (features['obp'] or 0) + (features['slg'] or 0)

    # ISO (Isolated Power = SLG - AVG)
    features['iso'] = (features['slg'] or 0) - (features['avg'] or 0)

    # K rate and BB rate
    features['k_rate'] = total_strikeouts / total_pa if total_pa > 0 else np.nan
    features['bb_rate'] = total_walks / total_pa if total_pa > 0 else np.nan

    # Ground ball rate
    features['gb_rate'] = games_df['gb'].sum() / total_bip if total_bip > 0 else np.nan

    # Barrel rate
    features['barrel_rate'] = games_df['barrels'].sum() / total_bip if total_bip > 0 else np.nan

    # Average exit velocity
    total_ev = games_df['total_exit_velo'].sum()
    total_ev_count = games_df['exit_velo_count'].sum()
    features['avg_exit_velo'] = total_ev / total_ev_count if total_ev_count > 0 else np.nan

    # wOBA
    total_woba_value = games_df['woba_value_sum'].sum()
    total_woba_denom = games_df['woba_denom_sum'].sum()
    features['woba'] = total_woba_value / total_woba_denom if total_woba_denom > 0 else np.nan

    # xwOBA
    total_xwoba = games_df['xwoba_sum'].sum()
    total_xwoba_count = games_df['xwoba_count'].sum()
    features['xwoba'] = total_xwoba / total_xwoba_count if total_xwoba_count > 0 else np.nan

    # Sample size
    features['games'] = len(games_df)
    features['pa_total'] = total_pa

    return features


def compute_rolling_batter_stats(games_df: pd.DataFrame, n_games: int) -> Dict:
    """
    Compute rolling stats over the last N games.

    Args:
        games_df: DataFrame with one row per game, sorted by date
        n_games: Number of games to include

    Returns:
        Dictionary of rolling features
    """
    prefix = f'roll_{n_games}g'

    if len(games_df) < n_games:
        return {
            f'{prefix}_woba': np.nan,
            f'{prefix}_xwoba': np.nan,
            f'{prefix}_k_rate': np.nan,
        }

    recent = games_df.tail(n_games)

    features = {}

    # wOBA
    total_woba_value = recent['woba_value_sum'].sum()
    total_woba_denom = recent['woba_denom_sum'].sum()
    features[f'{prefix}_woba'] = total_woba_value / total_woba_denom if total_woba_denom > 0 else np.nan

    # xwOBA
    total_xwoba = recent['xwoba_sum'].sum()
    total_xwoba_count = recent['xwoba_count'].sum()
    features[f'{prefix}_xwoba'] = total_xwoba / total_xwoba_count if total_xwoba_count > 0 else np.nan

    # K rate
    total_k = recent['strikeouts'].sum()
    total_pa = recent['pa'].sum()
    features[f'{prefix}_k_rate'] = total_k / total_pa if total_pa > 0 else np.nan

    return features


def build_batter_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build batter feature table with platoon splits.

    Creates one row per (batter, game_date, pitcher_hand) combination,
    using only data available BEFORE that game date.

    Args:
        raw_df: Raw Statcast pitch-level data

    Returns:
        DataFrame with batter features split by pitcher handedness
    """
    logger.info("Building batter features...")

    # Filter to F5 innings and add event columns
    logger.info("  Filtering to F5 and adding event columns...")
    df = filter_f5(raw_df)
    df = identify_batter_events(df)

    # Sort by date
    df = df.sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number'])

    # Pre-compute game-level stats for all batters (much faster)
    logger.info("  Pre-computing game-level stats...")

    game_stats_all = []
    for (batter, game_pk, p_throws), group in df.groupby(['batter', 'game_pk', 'p_throws']):
        stats = compute_batter_game_stats(group)
        if stats:
            stats['batter'] = batter
            stats['game_pk'] = game_pk
            stats['p_throws'] = p_throws
            stats['game_date'] = group['game_date'].iloc[0]
            game_stats_all.append(stats)

    game_stats_df = pd.DataFrame(game_stats_all)
    game_stats_df['season'] = game_stats_df['game_date'].apply(get_season)
    game_stats_df = game_stats_df.sort_values(['batter', 'game_date'])

    logger.info(f"  Pre-computed {len(game_stats_df):,} batter-game-hand stat rows")

    # Get unique batters
    batters = game_stats_df['batter'].unique()
    logger.info(f"  Processing {len(batters):,} unique batters...")

    feature_rows = []

    for idx, batter in enumerate(batters):
        if (idx + 1) % 100 == 0:
            logger.info(f"    Progress: {idx + 1:,} / {len(batters):,} batters")

        batter_stats = game_stats_df[game_stats_df['batter'] == batter].copy()

        # Get unique game dates for this batter
        game_dates = batter_stats['game_date'].unique()

        for game_date in game_dates:
            season = get_season(game_date)

            # Get prior games for this batter in same season
            prior_stats = batter_stats[
                (batter_stats['game_date'] < game_date) &
                (batter_stats['season'] == season)
            ]

            # Compute features for each pitcher handedness
            for hand in ['L', 'R']:
                hand_stats = prior_stats[prior_stats['p_throws'] == hand]

                if len(hand_stats) == 0:
                    feature_rows.append({
                        'batter': batter,
                        'game_date': game_date,
                        'vs_hand': hand,
                    })
                    continue

                # Compute season stats from pre-computed game stats
                season_features = compute_season_batter_stats(hand_stats)

                # Compute rolling stats (15 games)
                roll_15g = compute_rolling_batter_stats(hand_stats, 15)

                features = {
                    'batter': batter,
                    'game_date': game_date,
                    'vs_hand': hand,
                    **season_features,
                    **roll_15g
                }

                feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)
    logger.info(f"  Built features for {len(features_df):,} batter-game-hand combinations")

    return features_df


def process_batter_features() -> pd.DataFrame:
    """
    Main function to process batter features and save to parquet.
    """
    logger.info("=" * 60)
    logger.info("Processing batter features")
    logger.info("=" * 60)

    # Load data
    logger.info("\n1. Loading raw data...")
    raw_df = load_all_raw_data()

    # Build features
    logger.info("\n2. Building batter features...")
    features_df = build_batter_features(raw_df)

    # Save
    logger.info("\n3. Saving...")
    output_path = get_processed_data_dir() / "batter_features.parquet"
    features_df.to_parquet(output_path, index=False)
    logger.info(f"   Saved to {output_path}")

    # Summary stats
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total rows: {len(features_df):,}")
    logger.info(f"Unique batters: {features_df['batter'].nunique():,}")
    logger.info(f"Columns: {len(features_df.columns)}")

    # Check for missing values in key columns
    key_cols = ['woba', 'xwoba', 'k_rate', 'avg_exit_velo']
    for col in key_cols:
        if col in features_df.columns:
            missing = features_df[col].isna().sum()
            pct = missing / len(features_df) * 100
            logger.info(f"  {col}: {missing:,} missing ({pct:.1f}%)")

    # Stats by platoon split
    logger.info("\nBy platoon split:")
    for hand in ['L', 'R']:
        hand_df = features_df[features_df['vs_hand'] == hand]
        valid_woba = hand_df['woba'].dropna()
        logger.info(f"  vs {hand}HP: {len(hand_df):,} rows, mean wOBA={valid_woba.mean():.3f}")

    return features_df


def verify_batter_features() -> None:
    """Verify the batter features file."""
    output_path = get_processed_data_dir() / "batter_features.parquet"

    if not output_path.exists():
        print(f"ERROR: {output_path} not found")
        return

    df = pd.read_parquet(output_path)

    print("\n" + "=" * 60)
    print("BATTER FEATURES VERIFICATION")
    print("=" * 60)

    print(f"\nShape: {df.shape}")
    print(f"Unique batters: {df['batter'].nunique():,}")

    print(f"\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print(f"\nSample stats by platoon split:")
    for hand in ['L', 'R']:
        hand_df = df[df['vs_hand'] == hand]
        print(f"\n  vs {hand}HP:")
        for col in ['woba', 'xwoba', 'k_rate', 'iso', 'avg_exit_velo']:
            if col in hand_df.columns:
                valid = hand_df[col].dropna()
                if len(valid) > 0:
                    print(f"    {col}: mean={valid.mean():.3f}, std={valid.std():.3f}")


if __name__ == "__main__":
    process_batter_features()
