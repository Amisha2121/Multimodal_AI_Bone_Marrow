# -*- coding: utf-8 -*-
"""
Database persistence layer for Bone Marrow Cytomorphology Analysis cases.
Provides SQLite storage at DATA_DIR/cases.db.
"""

import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(REPO_ROOT / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "cases.db"


def get_db_connection():
    """Create and return a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialise SQLite database schema with additive migrations only."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                analysis_id TEXT UNIQUE,
                timestamp TEXT,
                image_path TEXT,
                first_image TEXT,
                age_years REAL,
                sex TEXT,
                sample_site TEXT,
                cells_count INTEGER,
                blast_pct REAL,
                morphology_counts_json TEXT,
                morphology_percentages_json TEXT,
                detection_images_json TEXT,
                cell_results_json TEXT,
                report_text TEXT,
                report_json TEXT,
                chat_messages_json TEXT,
                created_at TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT,
                timestamp TEXT,
                agent TEXT,
                tool TEXT,
                input_summary TEXT,
                output_summary TEXT,
                status TEXT
            );
        """)
        # Phase 9 – additive migrations
        try:
            conn.execute("ALTER TABLE cases ADD COLUMN annotations_json TEXT;")
        except Exception:
            pass  # column already exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS report_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                analysis_id TEXT,
                version_num INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                trigger TEXT,
                generator TEXT,
                revision_count INTEGER DEFAULT 0,
                report_json TEXT,
                section_title TEXT
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rv_case
            ON report_versions(case_id, version_num);
        """)
        conn.commit()


def _serialize_json(val, default=""):
    """Helper to safely serialize data to JSON string."""
    if val is None:
        return default
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return default


def _extract_case_fields(analysis_id, case_dict, chat_messages=None):
    """Extract standard dictionary of columns from an analysis dictionary."""
    case_id = case_dict.get('case_id') or f"CASE-{analysis_id}"
    timestamp = case_dict.get('timestamp') or datetime.now().isoformat()
    result = case_dict.get('analysis')

    image_path = None
    first_image = case_dict.get('first_image')
    age_years = None
    sex = None
    sample_site = None

    if result is not None:
        age_years = getattr(result, 'age_years', None)
        sex = getattr(result, 'sex', None)
        sample_site = getattr(result, 'sample_site', None)
        input_files = getattr(result, 'input_files', [])
        if input_files:
            image_path = str(input_files[0])
    
    if not image_path and case_dict.get('images'):
        image_path = str(case_dict['images'][0])
    if not image_path and first_image:
        image_path = str(first_image)

    cells_count = case_dict.get('cells_count', 0)
    blast_pct = float(case_dict.get('blast_pct', 0.0) or 0.0)

    morphology_counts = case_dict.get('morphology') or (
        getattr(result, 'morphology_counts', {}) if result else {}
    )
    morphology_percentages = case_dict.get('morphology_percentages') or (
        getattr(result, 'morphology_percentages', {}) if result else {}
    )
    detection_images = case_dict.get('detection_images') or {}
    cell_images = case_dict.get('cell_images') or []

    cached_report = case_dict.get('cached_report') or {}
    report_text = ""
    if isinstance(cached_report, dict) and 'sections' in cached_report:
        report_text = "\n\n".join(
            f"## {s.get('title', '')}\n{s.get('content', '')}"
            for s in cached_report.get('sections', [])
        )

    chats = chat_messages if chat_messages is not None else []

    return {
        'case_id': case_id,
        'analysis_id': analysis_id,
        'timestamp': timestamp,
        'image_path': str(image_path) if image_path else None,
        'first_image': str(first_image) if first_image else None,
        'age_years': age_years,
        'sex': sex,
        'sample_site': sample_site,
        'cells_count': cells_count,
        'blast_pct': blast_pct,
        'morphology_counts_json': _serialize_json(morphology_counts, "{}"),
        'morphology_percentages_json': _serialize_json(morphology_percentages, "{}"),
        'detection_images_json': _serialize_json(detection_images, "{}"),
        'cell_results_json': _serialize_json(cell_images, "[]"),
        'report_text': report_text,
        'report_json': _serialize_json(cached_report, "{}"),
        'chat_messages_json': _serialize_json(chats, "[]"),
        'created_at': datetime.now().isoformat(),
    }


def import_existing_case(analysis_id, case_dict, chat_messages=None):
    """
    Idempotent insert-only import. Never overwrites or deletes existing rows.
    """
    if not case_dict or not analysis_id:
        return False

    case_id = case_dict.get('case_id')
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM cases WHERE analysis_id = ? OR case_id = ?",
            (analysis_id, case_id)
        )
        if cursor.fetchone():
            return False  # Already exists, do not overwrite

        fields = _extract_case_fields(analysis_id, case_dict, chat_messages)
        cursor.execute("""
            INSERT INTO cases (
                case_id, analysis_id, timestamp, image_path, first_image,
                age_years, sex, sample_site, cells_count, blast_pct,
                morphology_counts_json, morphology_percentages_json,
                detection_images_json, cell_results_json, report_text,
                report_json, chat_messages_json, created_at
            ) VALUES (
                :case_id, :analysis_id, :timestamp, :image_path, :first_image,
                :age_years, :sex, :sample_site, :cells_count, :blast_pct,
                :morphology_counts_json, :morphology_percentages_json,
                :detection_images_json, :cell_results_json, :report_text,
                :report_json, :chat_messages_json, :created_at
            )
        """, fields)
        conn.commit()
        return True


def save_case_to_db(analysis_id, case_dict, chat_messages=None):
    """
    Insert or update a case in SQLite for dual-write during live operation.
    """
    if not case_dict or not analysis_id:
        return False

    fields = _extract_case_fields(analysis_id, case_dict, chat_messages)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO cases (
                case_id, analysis_id, timestamp, image_path, first_image,
                age_years, sex, sample_site, cells_count, blast_pct,
                morphology_counts_json, morphology_percentages_json,
                detection_images_json, cell_results_json, report_text,
                report_json, chat_messages_json, created_at
            ) VALUES (
                :case_id, :analysis_id, :timestamp, :image_path, :first_image,
                :age_years, :sex, :sample_site, :cells_count, :blast_pct,
                :morphology_counts_json, :morphology_percentages_json,
                :detection_images_json, :cell_results_json, :report_text,
                :report_json, :chat_messages_json, :created_at
            )
            ON CONFLICT(case_id) DO UPDATE SET
                analysis_id = excluded.analysis_id,
                timestamp = excluded.timestamp,
                image_path = COALESCE(excluded.image_path, cases.image_path),
                first_image = COALESCE(excluded.first_image, cases.first_image),
                age_years = COALESCE(excluded.age_years, cases.age_years),
                sex = COALESCE(excluded.sex, cases.sex),
                sample_site = COALESCE(excluded.sample_site, cases.sample_site),
                cells_count = excluded.cells_count,
                blast_pct = excluded.blast_pct,
                morphology_counts_json = excluded.morphology_counts_json,
                morphology_percentages_json = excluded.morphology_percentages_json,
                detection_images_json = excluded.detection_images_json,
                cell_results_json = excluded.cell_results_json,
                report_text = CASE WHEN length(excluded.report_text) > 0 THEN excluded.report_text ELSE cases.report_text END,
                report_json = CASE WHEN excluded.report_json != '{}' THEN excluded.report_json ELSE cases.report_json END
        """, fields)
        conn.commit()
        return True


def update_case_chat_in_db(analysis_id, chat_messages):
    """Update chat messages for a case in SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE cases
            SET chat_messages_json = ?
            WHERE analysis_id = ? OR case_id = ?
        """, (_serialize_json(chat_messages, "[]"), analysis_id, analysis_id))
        conn.commit()


def update_case_report_in_db(analysis_id, report_data):
    """Update cached report and report text for a case in SQLite."""
    report_text = ""
    if isinstance(report_data, dict) and 'sections' in report_data:
        report_text = "\n\n".join(
            f"## {s.get('title', '')}\n{s.get('content', '')}"
            for s in report_data.get('sections', [])
        )
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE cases
            SET report_json = ?, report_text = ?
            WHERE analysis_id = ? OR case_id = ?
        """, (_serialize_json(report_data, "{}"), report_text, analysis_id, analysis_id))
        conn.commit()


def get_history_from_db():
    """
    Fetch all saved cases formatted for the Analysis History sidebar.
    Returns list of case summary dictionaries.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT analysis_id, case_id, timestamp, cells_count, blast_pct, first_image, image_path
            FROM cases
            ORDER BY timestamp ASC
        """)
        rows = cursor.fetchall()

    history = []
    for r in rows:
        history.append({
            'id': r['analysis_id'],
            'case_id': r['case_id'],
            'timestamp': r['timestamp'],
            'cells': r['cells_count'],
            'blasts': float(r['blast_pct'] or 0.0),
            'thumbnail': r['first_image'] or r['image_path'],
        })
    return history


def get_case_from_db(case_or_analysis_id):
    """
    Fetch full case details by analysis_id or case_id.
    Returns a dictionary or None if not found.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM cases
            WHERE analysis_id = ? OR case_id = ?
            LIMIT 1
        """, (case_or_analysis_id, case_or_analysis_id))
        row = cursor.fetchone()

    if not row:
        return None

    def _parse_json(val, default):
        if not val:
            return default
        try:
            return json.loads(val)
        except Exception:
            return default

    morphology_counts = _parse_json(row['morphology_counts_json'], {})
    morphology_percentages = _parse_json(row['morphology_percentages_json'], {})
    detection_images = _parse_json(row['detection_images_json'], {})
    cell_images = _parse_json(row['cell_results_json'], [])
    cached_report = _parse_json(row['report_json'], {})
    chat_messages = _parse_json(row['chat_messages_json'], [])

    from pipeline.uncertainty import wilson_interval
    blast_cnt = morphology_counts.get("Blast", int(round(row['cells_count'] * float(row['blast_pct'] or 0.0)))) if row['cells_count'] else 0
    ci = wilson_interval(blast_cnt, row['cells_count'])

    return {
        'id': row['analysis_id'],
        'case_id': row['case_id'],
        'timestamp': row['timestamp'],
        'image_path': row['image_path'],
        'first_image': row['first_image'],
        'age_years': row['age_years'],
        'sex': row['sex'],
        'sample_site': row['sample_site'],
        'metrics': {
            'images_processed': 1,
            'cells_detected': row['cells_count'],
            'cells_classified': row['cells_count'],
            'blast_fraction': float(row['blast_pct'] or 0.0),
            'blast_ci': list(ci) if ci else None,
        },
        'morphology': [
            {
                'type': ct,
                'count': cnt,
                'percentage': float(morphology_percentages.get(ct, 0.0)),
            }
            for ct, cnt in morphology_counts.items()
        ],
        'morphology_counts': morphology_counts,
        'morphology_percentages': morphology_percentages,
        'detection_images': detection_images,
        'cell_images': cell_images,
        'report': cached_report,
        'report_text': row['report_text'],
        'chat_history': chat_messages,
    }


def log_agent_event(case_id, agent, tool, input_summary, output_summary, status="success"):
    """
    Append-only logger for agent events.
    """
    now = datetime.now().isoformat()
    try:
        init_db()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO agent_events (case_id, timestamp, agent, tool, input_summary, output_summary, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (case_id or "", now, str(agent), str(tool), str(input_summary)[:500], str(output_summary)[:500], str(status)))
            conn.commit()
    except Exception as e:
        print(f"Failed to log agent event: {e}")


def get_agent_events_from_db(case_id=None, limit=100):
    """
    Retrieve agent events, newest first.
    """
    try:
        init_db()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if case_id:
                cursor.execute("""
                    SELECT id, case_id, timestamp, agent, tool, input_summary, output_summary, status
                    FROM agent_events
                    WHERE case_id = ? OR case_id = ''
                    ORDER BY id DESC
                    LIMIT ?
                """, (case_id, limit))
            else:
                cursor.execute("""
                    SELECT id, case_id, timestamp, agent, tool, input_summary, output_summary, status
                    FROM agent_events
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"Failed to fetch agent events: {e}")
        return []


def update_cell_reviewer_label_in_db(case_or_analysis_id, cell_id, reviewer_label):
    """
    Update reviewer_label for a specific cell in cell_results_json.
    Does not touch original AI class or confidence!
    """
    try:
        init_db()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT cell_results_json FROM cases WHERE case_id = ? OR analysis_id = ?", (case_or_analysis_id, case_or_analysis_id))
            row = cursor.fetchone()
            if not row:
                return False
            cells = json.loads(row['cell_results_json'] or "[]")
            updated = False
            for c in cells:
                cid = c.get('id') or c.get('cell_id')
                if cid == cell_id:
                    if reviewer_label is None or reviewer_label == '':
                        c.pop('reviewer_label', None)
                    else:
                        c['reviewer_label'] = reviewer_label
                    updated = True
                    break
            if updated:
                cursor.execute("UPDATE cases SET cell_results_json = ? WHERE case_id = ? OR analysis_id = ?",
                               (json.dumps(cells, ensure_ascii=False), case_or_analysis_id, case_or_analysis_id))
                conn.commit()
                return True
    except Exception as e:
        print(f"Failed to update cell reviewer label in DB: {e}")
    return False




# ---------------------------------------------------------------------------
# Phase 9 – Report versions
# ---------------------------------------------------------------------------

def save_report_version(case_id, analysis_id, report_data, trigger='initial',
                        generator='LLM', revision_count=0, section_title=None):
    """
    Append a new version row to report_versions.
    Returns the new version_num (1-based).
    """
    try:
        init_db()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(version_num) FROM report_versions WHERE case_id = ?",
                (case_id,)
            )
            row = cursor.fetchone()
            prev_max = row[0] if row and row[0] is not None else 0
            new_version = prev_max + 1
            cursor.execute("""
                INSERT INTO report_versions
                    (case_id, analysis_id, version_num, timestamp, trigger, generator,
                     revision_count, report_json, section_title)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                case_id,
                analysis_id,
                new_version,
                datetime.now().isoformat(),
                trigger,
                generator,
                revision_count,
                _serialize_json(report_data),
                section_title,
            ))
            conn.commit()
            return new_version
    except Exception as e:
        print(f"Failed to save report version: {e}")
        return None


def get_report_versions(case_or_analysis_id):
    """Return list of version metadata dicts (without the full report_json)."""
    try:
        init_db()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, case_id, analysis_id, version_num, timestamp, trigger,
                       generator, revision_count, section_title
                FROM report_versions
                WHERE case_id = ? OR analysis_id = ?
                ORDER BY version_num ASC
            """, (case_or_analysis_id, case_or_analysis_id))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"Failed to get report versions: {e}")
        return []


def get_report_version(case_or_analysis_id, version_num):
    """Return full report_json for a specific version."""
    try:
        init_db()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT report_json FROM report_versions
                WHERE (case_id = ? OR analysis_id = ?) AND version_num = ?
            """, (case_or_analysis_id, case_or_analysis_id, version_num))
            row = cursor.fetchone()
            if row:
                return json.loads(row['report_json'] or '{}')
    except Exception as e:
        print(f"Failed to get report version {version_num}: {e}")
    return None


# ---------------------------------------------------------------------------
# Phase 9 – Annotations (per-claim, never modifies report_json text)
# ---------------------------------------------------------------------------

def save_annotations(case_or_analysis_id, annotations_dict):
    """
    Persist annotations_dict (claim_id -> annotation) for the case.
    Does NOT modify report_json -- stored in separate column.
    """
    try:
        init_db()
        with get_db_connection() as conn:
            conn.execute("""
                UPDATE cases SET annotations_json = ?
                WHERE case_id = ? OR analysis_id = ?
            """, (
                _serialize_json(annotations_dict),
                case_or_analysis_id,
                case_or_analysis_id,
            ))
            conn.commit()
            return True
    except Exception as e:
        print(f"Failed to save annotations: {e}")
        return False


def get_annotations(case_or_analysis_id):
    """Return annotations dict for the case (empty dict if none)."""
    try:
        init_db()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT annotations_json FROM cases WHERE case_id = ? OR analysis_id = ?",
                (case_or_analysis_id, case_or_analysis_id)
            )
            row = cursor.fetchone()
            if row and row['annotations_json']:
                return json.loads(row['annotations_json'])
    except Exception as e:
        print(f"Failed to get annotations: {e}")
    return {}
