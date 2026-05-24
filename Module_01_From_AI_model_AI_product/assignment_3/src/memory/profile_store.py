import sqlite3
import json
from typing import Optional


DB_PATH = "agent_memory.db"


def init_profile_table(conn: sqlite3.Connection) -> None:
    """Creates the user_profiles table if it doesn't already exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            session_id      TEXT PRIMARY KEY,
            name            TEXT,
            preferences     TEXT DEFAULT '[]',
            frequent_topics TEXT DEFAULT '[]'
        )
    """)
    conn.commit()


def get_profile(conn: sqlite3.Connection, session_id: str) -> Optional[dict]:
    """Reads the profile for a given session. Returns None if not found."""
    cursor = conn.execute(
        "SELECT name, preferences, frequent_topics FROM user_profiles WHERE session_id = ?",
        (session_id,)
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "name":            row[0],
        "preferences":     json.loads(row[1] or "[]"),
        "frequent_topics": json.loads(row[2] or "[]")
    }


def upsert_profile(conn: sqlite3.Connection, session_id: str, profile: dict) -> None:
    """Inserts or updates the user profile for a given session."""
    conn.execute("""
        INSERT INTO user_profiles (session_id, name, preferences, frequent_topics)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            name            = excluded.name,
            preferences     = excluded.preferences,
            frequent_topics = excluded.frequent_topics
    """, (
        session_id,
        profile.get("name"),
        json.dumps(profile.get("preferences", [])),
        json.dumps(profile.get("frequent_topics", []))
    ))
    conn.commit()


def format_profile_for_prompt(profile: Optional[dict]) -> str:
    """Formats a profile dict into a system prompt string. Returns empty string if no profile."""
    if not profile:
        return ""
    parts = []
    if profile.get("name"):
        parts.append(f"User's name: {profile['name']}.")
    if profile.get("preferences"):
        parts.append(f"Known preferences: {', '.join(profile['preferences'])}.")
    if profile.get("frequent_topics"):
        parts.append(f"Frequent topics of interest: {', '.join(profile['frequent_topics'])}.")
    if not parts:
        return ""
    return "\n\nKnown user context (from previous sessions):\n" + "\n".join(parts)