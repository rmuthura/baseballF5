"""
Daily Data Update Script.

Updates Statcast data and feature tables for ongoing model use.

Usage:
    # Update yesterday's data
    python -m f5_model.scripts.daily_update

    # Update specific date
    python -m f5_model.scripts.daily_update --date 2026-04-03

    # Update date range
    python -m f5_model.scripts.daily_update --start 2026-04-01 --end 2026-04-03

    # Force rebuild all features
    python -m f5_model.scripts.daily_update --rebuild-features
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent / "data"


def get_raw_dir() -> Path:
    """Get raw data directory."""
    return get_data_dir() / "raw"


def get_processed_dir() -> Path:
    """Get processed data directory."""
    return get_data_dir() / "processed"


def pull_statcast_day(date: str) -> Optional[pd.DataFrame]:
    """
    Pull Statcast data for a single day.

    Args:
        date: Date string YYYY-MM-DD

    Returns:
        DataFrame with pitch data or None if no games
    """
    try:
        from pybaseball import statcast
    except ImportError:
        logger.error("pybaseball not installed. Run: pip install pybaseball")
        return None

    logger.info(f"Pulling Statcast data for {date}...")

    try:
        df = statcast(start_dt=date, end_dt=date)

        if df is None or len(df) == 0:
            logger.info(f"  No data for {date} (likely no games)")
            return None

        logger.info(f"  Pulled {len(df):,} pitches")
        return df

    except Exception as e:
        logger.error(f"  Error pulling {date}: {e}")
        return None


def append_to_raw_data(df: pd.DataFrame, date: str):
    """
    Append new data to raw statcast files.

    Saves to monthly parquet files.
    """
    raw_dir = get_raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Determine month for this data
    month = date[:7]  # YYYY-MM
    filepath = raw_dir / f"statcast_{month.replace('-', '_')}.parquet"

    if filepath.exists():
        # Append to existing
        existing = pd.read_parquet(filepath)
        # Remove any existing data for this date (avoid duplicates)
        existing = existing[existing['game_date'] != date]
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_parquet(filepath, index=False)
        logger.info(f"  Appended to {filepath.name} ({len(combined):,} total pitches)")
    else:
        # New file
        df.to_parquet(filepath, index=False)
        logger.info(f"  Created {filepath.name} ({len(df):,} pitches)")


def update_pitcher_features(date: str):
    """
    Update pitcher feature table with new game data.

    Args:
        date: Date of new data
    """
    from f5_model.features.pitcher_features import build_pitcher_features

    logger.info(f"Updating pitcher features for {date}...")

    try:
        processed_dir = get_processed_dir()
        features_path = processed_dir / "pitcher_features.parquet"

        # Load existing features if available
        if features_path.exists():
            existing = pd.read_parquet(features_path)
            # Remove any features for this date (will recalculate)
            existing = existing[existing['game_date'] < date]
        else:
            existing = None

        # Build new features for this date
        new_features = build_pitcher_features(end_date=date)

        if new_features is None or len(new_features) == 0:
            logger.warning("  No new pitcher features generated")
            return

        # Filter to just the new date
        new_features = new_features[new_features['game_date'] == date]

        if len(new_features) == 0:
            logger.info("  No pitcher starts on this date")
            return

        # Combine
        if existing is not None and len(existing) > 0:
            combined = pd.concat([existing, new_features], ignore_index=True)
        else:
            combined = new_features

        combined.to_parquet(features_path, index=False)
        logger.info(f"  Updated pitcher features ({len(combined):,} total rows)")

    except Exception as e:
        logger.error(f"  Error updating pitcher features: {e}")


def update_batter_features(date: str):
    """
    Update batter feature table with new game data.

    Args:
        date: Date of new data
    """
    from f5_model.features.batter_features import build_batter_features

    logger.info(f"Updating batter features for {date}...")

    try:
        processed_dir = get_processed_dir()
        features_path = processed_dir / "batter_features.parquet"

        # Load existing features if available
        if features_path.exists():
            existing = pd.read_parquet(features_path)
            # Remove features after this date
            existing = existing[existing['game_date'] < date]
        else:
            existing = None

        # Build new features
        new_features = build_batter_features(end_date=date)

        if new_features is None or len(new_features) == 0:
            logger.warning("  No new batter features generated")
            return

        # Filter to just the new date
        new_features = new_features[new_features['game_date'] == date]

        if len(new_features) == 0:
            logger.info("  No batter appearances on this date")
            return

        # Combine
        if existing is not None and len(existing) > 0:
            combined = pd.concat([existing, new_features], ignore_index=True)
        else:
            combined = new_features

        combined.to_parquet(features_path, index=False)
        logger.info(f"  Updated batter features ({len(combined):,} total rows)")

    except Exception as e:
        logger.error(f"  Error updating batter features: {e}")


def rebuild_all_features():
    """Rebuild all feature tables from raw data."""
    from f5_model.features.pitcher_features import build_pitcher_features
    from f5_model.features.batter_features import build_batter_features

    logger.info("Rebuilding all features from raw data...")

    processed_dir = get_processed_dir()
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Pitcher features
    logger.info("Building pitcher features...")
    pitcher_features = build_pitcher_features()
    if pitcher_features is not None:
        pitcher_features.to_parquet(processed_dir / "pitcher_features.parquet", index=False)
        logger.info(f"  Saved {len(pitcher_features):,} pitcher feature rows")

    # Batter features
    logger.info("Building batter features...")
    batter_features = build_batter_features()
    if batter_features is not None:
        batter_features.to_parquet(processed_dir / "batter_features.parquet", index=False)
        logger.info(f"  Saved {len(batter_features):,} batter feature rows")


def run_daily_update(
    date: str = None,
    start_date: str = None,
    end_date: str = None,
    rebuild: bool = False
):
    """
    Run the daily update pipeline.

    Args:
        date: Single date to update
        start_date: Start of date range
        end_date: End of date range
        rebuild: Whether to rebuild all features
    """
    # Determine dates to process
    if date:
        dates = [date]
    elif start_date and end_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
    else:
        # Default to yesterday
        yesterday = datetime.now() - timedelta(days=1)
        dates = [yesterday.strftime("%Y-%m-%d")]

    logger.info(f"Processing dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")

    # Pull new data
    for d in dates:
        df = pull_statcast_day(d)
        if df is not None and len(df) > 0:
            append_to_raw_data(df, d)

    # Update features
    if rebuild:
        rebuild_all_features()
    else:
        for d in dates:
            update_pitcher_features(d)
            update_batter_features(d)

    logger.info("Daily update complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Daily data update for F5 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Update yesterday's data
    python -m f5_model.scripts.daily_update

    # Update specific date
    python -m f5_model.scripts.daily_update --date 2026-04-03

    # Update date range
    python -m f5_model.scripts.daily_update --start 2026-04-01 --end 2026-04-03

    # Rebuild all features from raw data
    python -m f5_model.scripts.daily_update --rebuild-features
        """
    )
    parser.add_argument("--date", "-d", help="Single date to update (YYYY-MM-DD)")
    parser.add_argument("--start", help="Start date for range (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date for range (YYYY-MM-DD)")
    parser.add_argument("--rebuild-features", action="store_true",
                        help="Rebuild all features from raw data")

    args = parser.parse_args()

    if args.start and not args.end:
        args.end = datetime.now().strftime("%Y-%m-%d")

    run_daily_update(
        date=args.date,
        start_date=args.start,
        end_date=args.end,
        rebuild=args.rebuild_features
    )


if __name__ == "__main__":
    main()
