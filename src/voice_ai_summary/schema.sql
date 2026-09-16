-- Voice AI Summary schema. Applied idempotently by db.py.
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

-- Immutable capture units (one audio file each).
CREATE TABLE IF NOT EXISTS recordings (
    id              INTEGER PRIMARY KEY,
    source          TEXT NOT NULL,              -- mac_mic | mac_system | file
    device_id       TEXT NOT NULL DEFAULT '',
    started_at_utc  TEXT NOT NULL,              -- ISO 8601, UTC, e.g. 2026-09-15T01:02:03Z
    tz_offset       TEXT NOT NULL DEFAULT '+00:00',
    duration_ms     INTEGER,
    sha256          TEXT NOT NULL UNIQUE,
    storage_path    TEXT NOT NULL,              -- relative to <data_dir>/store
    original_name   TEXT NOT NULL DEFAULT '',
    ingested_at     TEXT NOT NULL,
    processed_at    TEXT,
    error           TEXT,
    claimed_at      TEXT,                       -- set while a caller owns this row for processing (pipeline.py)
    processing_ms   INTEGER                     -- wall time for VAD+ASR in process_recording (pipeline.py)
);
CREATE INDEX IF NOT EXISTS idx_recordings_started ON recordings(started_at_utc);
CREATE INDEX IF NOT EXISTS idx_recordings_unprocessed ON recordings(processed_at) WHERE processed_at IS NULL;

-- VAD output: speech regions within a recording.
CREATE TABLE IF NOT EXISTS segments (
    id            INTEGER PRIMARY KEY,
    recording_id  INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    start_ms      INTEGER NOT NULL,
    end_ms        INTEGER NOT NULL,
    speech_prob   REAL
);
CREATE INDEX IF NOT EXISTS idx_segments_recording ON segments(recording_id);

-- Episodes: contiguous stretches of activity (a call, a solo session, ambient chatter).
CREATE TABLE IF NOT EXISTS episodes (
    id              INTEGER PRIMARY KEY,
    started_at_utc  TEXT NOT NULL,
    ended_at_utc    TEXT NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'ambient',   -- call | solo | ambient
    title           TEXT,
    source_mix      TEXT NOT NULL DEFAULT ''           -- comma-joined sources seen, e.g. "mac_mic,mac_system"
);
CREATE INDEX IF NOT EXISTS idx_episodes_started ON episodes(started_at_utc);

-- ASR output: one row per utterance. Absolute time = recording.started_at_utc + t_start_ms.
CREATE TABLE IF NOT EXISTS utterances (
    id             INTEGER PRIMARY KEY,
    recording_id   INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    segment_id     INTEGER REFERENCES segments(id) ON DELETE SET NULL,
    episode_id     INTEGER REFERENCES episodes(id) ON DELETE SET NULL,
    t_start_ms     INTEGER NOT NULL,
    t_end_ms       INTEGER NOT NULL,
    abs_start_utc  TEXT NOT NULL,             -- denormalised for time-range queries
    text           TEXT NOT NULL,
    lang           TEXT,
    asr_model      TEXT,
    avg_logprob    REAL,
    speaker        TEXT NOT NULL DEFAULT 'unknown',  -- me | other | unknown (derived from track)
    raw_text       TEXT,                      -- pre-correction ASR text (set once corrected)
    corrected_at   TEXT,                       -- ISO 8601 UTC, when the correction pass last ran
    correction_model TEXT                      -- model id used for the correction pass
);
CREATE INDEX IF NOT EXISTS idx_utterances_recording ON utterances(recording_id);
CREATE INDEX IF NOT EXISTS idx_utterances_abs_start ON utterances(abs_start_utc);
CREATE INDEX IF NOT EXISTS idx_utterances_episode ON utterances(episode_id);

-- Full-text search. `trigram` is required for Japanese substring queries
-- (the default unicode61 tokenizer returns nothing for most Japanese searches).
CREATE VIRTUAL TABLE IF NOT EXISTS utterances_fts USING fts5(
    text,
    content='utterances',
    content_rowid='id',
    tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS utterances_ai AFTER INSERT ON utterances BEGIN
    INSERT INTO utterances_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS utterances_ad AFTER DELETE ON utterances BEGIN
    INSERT INTO utterances_fts(utterances_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS utterances_au AFTER UPDATE OF text ON utterances BEGIN
    INSERT INTO utterances_fts(utterances_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO utterances_fts(rowid, text) VALUES (new.id, new.text);
END;

-- Summaries at episode or day scope. scope_key: episode id, or YYYY-MM-DD (local date).
CREATE TABLE IF NOT EXISTS summaries (
    id              INTEGER PRIMARY KEY,
    scope           TEXT NOT NULL,             -- episode | day
    scope_key       TEXT NOT NULL,
    model           TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    json            TEXT NOT NULL,
    markdown        TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE(scope, scope_key, prompt_version)
);

-- Delivery log so a digest is never sent twice for the same day/channel.
CREATE TABLE IF NOT EXISTS deliveries (
    id          INTEGER PRIMARY KEY,
    scope_key   TEXT NOT NULL,                 -- YYYY-MM-DD
    channel     TEXT NOT NULL,                 -- slack | email
    sent_at     TEXT NOT NULL,
    status      TEXT NOT NULL,                 -- ok | error
    detail      TEXT
);
