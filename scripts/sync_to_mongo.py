"""
Helper script to sync all existing local SQLite database records
to MongoDB Atlas once IP whitelist is enabled.
"""

import sqlite3
import os
import certifi
from pymongo import MongoClient

def sync_data():
    mongo_uri = "mongodb+srv://luckybiswal98210_db_user:Rasmi420410@cluster0.osbg15e.mongodb.net/?retryWrites=true&w=majority"
    db_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "healthcare_records.db")

    if not os.path.exists(db_file):
        print(f"❌ SQLite database file not found at: {db_file}")
        return

    print("🔄 Attempting to connect to MongoDB Atlas Cloud Database...")
    try:
        client = MongoClient(mongo_uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✅ Successfully connected to MongoDB Atlas!")
    except Exception as e:
        print(f"❌ Could not connect to MongoDB Atlas: {e}")
        print("\n👉 To fix this:")
        print("1. Go to https://cloud.mongodb.com/")
        print("2. Navigate to 'Network Access' under Security")
        print("3. Click 'Add IP Address' -> Select 'Allow Access from Anywhere' (0.0.0.0/0)")
        print("4. Wait 1 minute and run this script again.")
        return

    mongo_db = client["healthcare_cdss_db"]
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    tables = ["users", "screening_records", "clinical_uploads", "patient_complaints"]

    for table in tables:
        try:
            cur.execute(f"SELECT * FROM {table}")
            rows = [dict(r) for r in cur.fetchall()]
            if not rows:
                print(f"ℹ️ Table '{table}' has 0 rows, skipping.")
                continue

            col = mongo_db[table]
            upserted = 0
            for row in rows:
                if table == "users":
                    col.update_one({"user_id": row["user_id"]}, {"$set": row}, upsert=True)
                elif table == "screening_records":
                    col.update_one({"user_id": row["user_id"], "timestamp": row["timestamp"], "disease": row["disease"]}, {"$set": row}, upsert=True)
                    mongo_db["screenings"].update_one({"user_id": row["user_id"], "timestamp": row["timestamp"], "disease": row["disease"]}, {"$set": row}, upsert=True)
                elif table == "clinical_uploads":
                    col.update_one({"user_id": row["user_id"], "timestamp": row["timestamp"], "file_name": row.get("file_name")}, {"$set": row}, upsert=True)
                elif table == "patient_complaints":
                    col.update_one({"complaint_id": row.get("complaint_id", 0)}, {"$set": row}, upsert=True)
                upserted += 1

            print(f"✅ Synced {upserted} records into MongoDB collection '{table}'")
        except Exception as e:
            print(f"⚠️ Error syncing table {table}: {e}")

    conn.close()
    print("\n🎉 All local SQLite records successfully synced to MongoDB Atlas!")

if __name__ == "__main__":
    sync_data()
