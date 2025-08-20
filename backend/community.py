# backend/community.py
"""
Simple local community/crowdsourcing backend.
Stores submissions to backend/data/community_submissions.csv

UPDATED: Includes support for unique IDs and up/down voting.
- Automatically migrates existing CSV to the new format.

Functions:
  - add_submission(...)
  - list_submissions() -> pandas.DataFrame
  - upvote_submission(submission_id)
  - downvote_submission(submission_id)
"""

from pathlib import Path
import pandas as pd
from datetime import datetime
import uuid
import os

# --- Configuration ---
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "community_submissions.csv"

# The full schema we want for our CSV file
EXPECTED_COLUMNS = [
    "submission_id", "timestamp", "name", "email", "role",
    "submission_type", "content", "attached_filename",
    "upvotes", "downvotes"
]

# --- Helper Functions ---

def _load_and_migrate_df() -> pd.DataFrame:
    """
    Loads the submissions DataFrame. If the CSV doesn't exist or is empty,
    it creates one with the correct schema. It also handles migrating
    older CSVs by adding missing columns.
    """
    if not DB_PATH.exists() or os.path.getsize(DB_PATH) == 0:
        # If file doesn't exist or is empty, create a new one
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        df.to_csv(DB_PATH, index=False)
        return df

    # Load the existing CSV
    df = pd.read_csv(DB_PATH, dtype={'submission_id': str})

    # --- Migration Logic: Check for and add missing columns ---
    needs_saving = False
    if 'submission_id' not in df.columns:
        # Add unique IDs to existing rows
        df['submission_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        needs_saving = True

    if 'upvotes' not in df.columns:
        df['upvotes'] = 0
        needs_saving = True

    if 'downvotes' not in df.columns:
        df['downvotes'] = 0
        needs_saving = True
    
    # Ensure vote columns are numeric integers
    df['upvotes'] = pd.to_numeric(df['upvotes'], errors='coerce').fillna(0).astype(int)
    df['downvotes'] = pd.to_numeric(df['downvotes'], errors='coerce').fillna(0).astype(int)

    if needs_saving:
        # Reorder columns to the expected format and save the migrated file
        df = df.reindex(columns=EXPECTED_COLUMNS)
        df.to_csv(DB_PATH, index=False)

    return df

def _save_df(df: pd.DataFrame):
    """Saves the entire DataFrame to the CSV file."""
    df.to_csv(DB_PATH, index=False)

# --- Public API Functions ---

def add_submission(name: str, email: str, role: str, submission_type: str, content: str, attached_filename: str = None):
    """
    Append a submission row to the CSV DB. Includes a unique ID and vote counts.
    """
    df = _load_and_migrate_df()

    new_submission = {
        "submission_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "name": name,
        "email": email,
        "role": role,
        "submission_type": submission_type,
        "content": content,
        "attached_filename": attached_filename or "",
        "upvotes": 0,
        "downvotes": 0
    }

    new_row_df = pd.DataFrame([new_submission])
    df = pd.concat([df, new_row_df], ignore_index=True)
    _save_df(df)
    return True

def list_submissions(limit: int = 200) -> pd.DataFrame:
    """
    Read CSV and return DataFrame, sorted by timestamp.
    """
    df = _load_and_migrate_df()
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        # Sort by timestamp to show newest first, as in the original code
        df = df.sort_values("timestamp", ascending=False)
        
    if limit:
        df = df.head(limit)
        
    return df.reset_index(drop=True)

def upvote_submission(submission_id: str):
    """
    Increments the upvote count for a given submission ID.
    """
    df = _load_and_migrate_df()
    mask = df["submission_id"] == submission_id
    
    if mask.any():
        df.loc[mask, "upvotes"] += 1
        _save_df(df)

def downvote_submission(submission_id: str):
    """
    Increments the downvote count for a given submission ID.
    """
    df = _load_and_migrate_df()
    mask = df["submission_id"] == submission_id

    if mask.any():
        df.loc[mask, "downvotes"] += 1
        _save_df(df)