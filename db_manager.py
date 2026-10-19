import sqlite3
import os
from datetime import datetime
import pandas as pd

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "healthcare_records.db")

def get_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Patients Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        contact TEXT,
        medical_notes TEXT,
        created_at TEXT
    )
    """)
    
    # 2. Screening History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS screening_records (
        record_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        disease TEXT NOT NULL,
        probability REAL NOT NULL,
        status TEXT NOT NULL,
        vitals TEXT,
        notes TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
    )
    """)
    
    # Check if default demo patient exists
    cursor.execute("SELECT patient_id FROM patients WHERE patient_id = 'P-1001'")
    if not cursor.fetchone():
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
        INSERT INTO patients (patient_id, name, age, gender, contact, medical_notes, created_at)
        VALUES ('P-1001', 'Default Demo Patient (Self-Screening)', 45, 'Male', '+91 98765 43210', 'Baseline multi-system screening profile', ?)
        """, (now_str,))
        
    conn.commit()
    conn.close()

def add_patient(patient_id, name, age, gender, contact="", notes=""):
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO patients (patient_id, name, age, gender, contact, medical_notes, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (patient_id, name, age, gender, contact, notes, now_str))
    conn.commit()
    conn.close()

def get_all_patients():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_patient(patient_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def save_screening_record(patient_id, disease, probability, status, vitals="", notes=""):
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO screening_records (patient_id, timestamp, disease, probability, status, vitals, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (patient_id, now_str, disease, round(float(probability), 1), status, vitals, notes))
    conn.commit()
    conn.close()

def get_screening_records(patient_id=None):
    conn = get_connection()
    cursor = conn.cursor()
    if patient_id:
        cursor.execute("""
        SELECT r.*, p.name as patient_name 
        FROM screening_records r
        LEFT JOIN patients p ON r.patient_id = p.patient_id
        WHERE r.patient_id = ?
        ORDER BY r.record_id DESC
        """, (patient_id,))
    else:
        cursor.execute("""
        SELECT r.*, p.name as patient_name 
        FROM screening_records r
        LEFT JOIN patients p ON r.patient_id = p.patient_id
        ORDER BY r.record_id DESC
        """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_screening_record(record_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM screening_records WHERE record_id = ?", (record_id,))
    conn.commit()
    conn.close()

# Initialize DB when module loads
init_db()
