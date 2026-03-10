
import sqlite3
import os
import logging
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DB_NAME = config.DB_NAME

def create_connection():
    """Create a database connection to the SQLite database specified by DB_NAME"""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        logger.info("Connected to %s", DB_NAME)
    except sqlite3.Error as e:
        logger.error("Database connection error: %s", e)
    return conn

def create_table(conn, create_table_sql):
    """Create a table from the create_table_sql statement"""
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        logger.error("SQL execution error: %s", e)

def main():
    database = DB_NAME

    # SQL Statements
    sql_create_subjects_table = """
    CREATE TABLE IF NOT EXISTS subjects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE
    );"""

    sql_create_grades_table = """
    CREATE TABLE IF NOT EXISTS grades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        level_name TEXT NOT NULL,
        subject_id INTEGER NOT NULL,
        FOREIGN KEY (subject_id) REFERENCES subjects (id),
        UNIQUE(level_name, subject_id)
    );"""

    sql_create_units_table = """
    CREATE TABLE IF NOT EXISTS units (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        unit_no INTEGER,
        grade_id INTEGER NOT NULL,
        FOREIGN KEY (grade_id) REFERENCES grades (id)
    );"""

    sql_create_outcomes_table = """
    CREATE TABLE IF NOT EXISTS outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT NOT NULL,
        description TEXT,
        implementation_guide TEXT,
        unit_id INTEGER NOT NULL,
        FOREIGN KEY (unit_id) REFERENCES units (id),
        UNIQUE(code, unit_id)
    );"""

    sql_create_components_table = """
    CREATE TABLE IF NOT EXISTS components (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT NOT NULL,
        description TEXT NOT NULL,
        outcome_id INTEGER NOT NULL,
        FOREIGN KEY (outcome_id) REFERENCES outcomes (id)
    );"""

    sql_create_questions_table = """
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id INTEGER NOT NULL,
        grade_id INTEGER NOT NULL,
        unit_id INTEGER NOT NULL,
        outcome_id INTEGER NOT NULL,
        component_id INTEGER,
        context TEXT,
        question_text TEXT NOT NULL,
        rubric TEXT, -- Store as JSON string
        correct_answer_summary TEXT,
        cognitive_level TEXT,
        elapsed_time REAL,
        context_elapsed_time REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (subject_id) REFERENCES subjects (id),
        FOREIGN KEY (grade_id) REFERENCES grades (id),
        FOREIGN KEY (unit_id) REFERENCES units (id),
        FOREIGN KEY (outcome_id) REFERENCES outcomes (id),
        FOREIGN KEY (component_id) REFERENCES components (id)
    );"""

    # Indexes for performance (covering common JOIN columns)
    sql_create_indexes = [
        "CREATE INDEX IF NOT EXISTS idx_outcomes_unit ON outcomes(unit_id);",
        "CREATE INDEX IF NOT EXISTS idx_components_outcome ON components(outcome_id);",
        "CREATE INDEX IF NOT EXISTS idx_grades_subject ON grades(subject_id);",
        "CREATE INDEX IF NOT EXISTS idx_units_grade ON units(grade_id);",
        "CREATE INDEX IF NOT EXISTS idx_questions_outcome ON questions(outcome_id);",
    ]

    # Create DB Connection
    conn = create_connection()

    if conn is not None:
        logger.info("Creating tables...")
        create_table(conn, sql_create_subjects_table)
        create_table(conn, sql_create_grades_table)
        create_table(conn, sql_create_units_table)
        create_table(conn, sql_create_outcomes_table)
        create_table(conn, sql_create_components_table)
        create_table(conn, sql_create_questions_table)

        for idx_sql in sql_create_indexes:
            create_table(conn, idx_sql)

        logger.info("Tables and indexes created successfully.")

        # Close connection
        conn.close()
    else:
        logger.error("Cannot create the database connection.")

if __name__ == '__main__':
    main()
