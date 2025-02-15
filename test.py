import sqlite3

try:
    print("🔄 Connecting to SQLite...")

    # Connect to SQLite (creates 'test_db.sqlite' if it doesn't exist)
    db = sqlite3.connect("test_db.sqlite")
    cursor = db.cursor()

    # Create a test table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL
    )
    """)
    db.commit()
    
    # Insert test data
    cursor.execute("INSERT INTO users (name, email) VALUES ('John Doe', 'john@example.com')")
    db.commit()

    # Fetch and print data
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()

    print("✅ SQLite Works! Here’s the data:")
    for user in users:
        print(user)

except Exception as e:
    print("❌ SQLite Error:", e)

finally:
    db.close()


