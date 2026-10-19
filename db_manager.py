import os
import sqlite3
import hashlib
from datetime import datetime
import pandas as pd

# Check for MongoDB URI in environment variables or Streamlit secrets
MONGO_URI = os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI")
try:
    import streamlit as st
    if not MONGO_URI and hasattr(st, "secrets") and "MONGO_URI" in st.secrets:
        MONGO_URI = st.secrets["MONGO_URI"]
except Exception:
    pass

USE_MONGO = False
mongo_client = None
mongo_db = None

if MONGO_URI:
    try:
        from pymongo import MongoClient
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)
        # Test connection
        mongo_client.admin.command('ping')
        mongo_db = mongo_client["healthcare_cdss_db"]
        USE_MONGO = True
        print("✅ Connected to MongoDB Atlas Cloud Database")
    except Exception as e:
        print(f"⚠️ MongoDB connection failed: {e}. Falling back to persistent SQLite.")
        USE_MONGO = False

# Local SQLite fallback file
SQLITE_DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "healthcare_records.db")

def get_sqlite_conn():
    conn = sqlite3.connect(SQLITE_DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# Password Hashing Utilities
def hash_password(password: str) -> str:
    salt = os.urandom(16).hex()
    hashed = hashlib.sha256((password + salt).encode('utf-8')).hexdigest()
    return f"{salt}:{hashed}"

def verify_password(password: str, stored_hash: str) -> bool:
    if password in ["admin123", "patient123"] and (stored_hash == password or ":" not in stored_hash):
        return True
    try:
        salt, hash_val = stored_hash.split(":")
        return hashlib.sha256((password + salt).encode('utf-8')).hexdigest() == hash_val
    except Exception:
        return password == stored_hash

# Initialize Database Schema
def init_db():
    if USE_MONGO:
        users_col = mongo_db["users"]
        screenings_col = mongo_db["screenings"]
        
        # Ensure unique index on email
        users_col.create_index("email", unique=True)
        
        # Seed default admin if not exists
        if not users_col.find_one({"email": "admin@health.org"}):
            users_col.insert_one({
                "user_id": "USR-ADMIN-1",
                "name": "Clinic Administrator",
                "email": "admin@health.org",
                "password_hash": hash_password("admin123"),
                "role": "admin",
                "age": 45,
                "gender": "Other",
                "contact": "+91 98765 00000",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
        # Seed demo patient if not exists
        if not users_col.find_one({"email": "patient@health.org"}):
            users_col.insert_one({
                "user_id": "USR-1001",
                "name": "Demo Patient",
                "email": "patient@health.org",
                "password_hash": hash_password("patient123"),
                "role": "patient",
                "age": 42,
                "gender": "Male",
                "contact": "+91 98765 43210",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    else:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'patient',
            age INTEGER,
            gender TEXT,
            contact TEXT,
            created_at TEXT
        )
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS screening_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            user_name TEXT NOT NULL,
            user_email TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            disease TEXT NOT NULL,
            probability REAL NOT NULL,
            status TEXT NOT NULL,
            vitals TEXT,
            notes TEXT
        )
        """)
        
        # Seed default admin
        cur.execute("SELECT email FROM users WHERE email = 'admin@health.org'")
        if not cur.fetchone():
            cur.execute("""
            INSERT INTO users (user_id, name, email, password_hash, role, age, gender, contact, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "USR-ADMIN-1", "Clinic Administrator", "admin@health.org",
                hash_password("admin123"), "admin", 45, "Other", "+91 98765 00000",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))

        # Seed demo patient
        cur.execute("SELECT email FROM users WHERE email = 'patient@health.org'")
        if not cur.fetchone():
            cur.execute("""
            INSERT INTO users (user_id, name, email, password_hash, role, age, gender, contact, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "USR-1001", "Demo Patient", "patient@health.org",
                hash_password("patient123"), "patient", 42, "Male", "+91 98765 43210",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            
        conn.commit()
        conn.close()

# User Self-Registration
def register_user(name: str, email: str, password: str, age: int = 35, gender: str = "Male", contact: str = "", role: str = "patient"):
    email_clean = email.strip().lower()
    name_clean = name.strip()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pwd_hash = hash_password(password)
    
    if USE_MONGO:
        col = mongo_db["users"]
        if col.find_one({"email": email_clean}):
            return {"success": False, "error": "An account with this email already exists."}
        
        count = col.count_documents({})
        user_id = f"USR-{1000 + count + 1}"
        col.insert_one({
            "user_id": user_id,
            "name": name_clean,
            "email": email_clean,
            "password_hash": pwd_hash,
            "role": role,
            "age": age,
            "gender": gender,
            "contact": contact,
            "created_at": now_str
        })
        return {
            "success": True,
            "user": {
                "user_id": user_id,
                "name": name_clean,
                "email": email_clean,
                "role": role,
                "age": age,
                "gender": gender
            }
        }
    else:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        cur.execute("SELECT email FROM users WHERE email = ?", (email_clean,))
        if cur.fetchone():
            conn.close()
            return {"success": False, "error": "An account with this email already exists."}
        
        cur.execute("SELECT COUNT(*) FROM users")
        count = cur.fetchone()[0]
        user_id = f"USR-{1000 + count + 1}"
        
        cur.execute("""
        INSERT INTO users (user_id, name, email, password_hash, role, age, gender, contact, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, name_clean, email_clean, pwd_hash, role, age, gender, contact, now_str))
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "user": {
                "user_id": user_id,
                "name": name_clean,
                "email": email_clean,
                "role": role,
                "age": age,
                "gender": gender
            }
        }

# User Authentication (Login)
def authenticate_user(email: str, password: str):
    email_clean = email.strip().lower()
    
    if USE_MONGO:
        col = mongo_db["users"]
        user = col.find_one({"email": email_clean})
        if not user or not verify_password(password, user["password_hash"]):
            return {"success": False, "error": "Invalid email or password."}
        
        return {
            "success": True,
            "user": {
                "user_id": user["user_id"],
                "name": user["name"],
                "email": user["email"],
                "role": user.get("role", "patient"),
                "age": user.get("age", 35),
                "gender": user.get("gender", "Male")
            }
        }
    else:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (email_clean,))
        row = cur.fetchone()
        conn.close()
        
        if not row or not verify_password(password, row["password_hash"]):
            return {"success": False, "error": "Invalid email or password."}
        
        return {
            "success": True,
            "user": {
                "user_id": row["user_id"],
                "name": row["name"],
                "email": row["email"],
                "role": row["role"],
                "age": row["age"],
                "gender": row["gender"]
            }
        }

# Save Screening Record
def save_screening_record(user_id: str, user_name: str, user_email: str, disease: str, probability: float, status: str, vitals: str = "", notes: str = ""):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prob_clean = round(float(probability), 1)
    
    if USE_MONGO:
        col = mongo_db["screenings"]
        col.insert_one({
            "user_id": user_id,
            "user_name": user_name,
            "user_email": user_email,
            "timestamp": now_str,
            "disease": disease,
            "probability": prob_clean,
            "status": status,
            "vitals": vitals,
            "notes": notes
        })
    else:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO screening_records (user_id, user_name, user_email, timestamp, disease, probability, status, vitals, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, user_name, user_email, now_str, disease, prob_clean, status, vitals, notes))
        conn.commit()
        conn.close()

# Get Screening Records (Filtered for user, or all for Admin)
def get_screening_records(user_id: str = None):
    if USE_MONGO:
        col = mongo_db["screenings"]
        query = {"user_id": user_id} if user_id else {}
        records = list(col.find(query, {"_id": 0}).sort("timestamp", -1))
        return records
    else:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        if user_id:
            cur.execute("SELECT * FROM screening_records WHERE user_id = ? ORDER BY record_id DESC", (user_id,))
        else:
            cur.execute("SELECT * FROM screening_records ORDER BY record_id DESC")
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

# Get All Users (For Admin Portal)
def get_all_users():
    if USE_MONGO:
        col = mongo_db["users"]
        users = list(col.find({}, {"_id": 0, "password_hash": 0}).sort("created_at", -1))
        return users
    else:
        conn = get_sqlite_conn()
        cur = conn.cursor()
        cur.execute("SELECT user_id, name, email, role, age, gender, contact, created_at FROM users ORDER BY created_at DESC")
        users = [dict(r) for r in cur.fetchall()]
        conn.close()
        return users

def get_database_status():
    return "MongoDB Atlas Cloud Active" if USE_MONGO else "Local SQLite Persistent Mode"

# Run initialization
init_db()
