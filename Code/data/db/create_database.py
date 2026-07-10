import sqlite3
import pandas as pd
import os

def create_db(conn):

    # Map CSV filename to table name
    csv_files = {
        "sets.csv": "sets",
        "themes.csv": "themes",
        "parts.csv": "parts",
        "part_categories.csv": "part_categories",
        "part_relationships.csv": "part_relationships",
        "colors.csv": "colors",
        "elements.csv": "elements",
        "inventories.csv": "inventories",
        "inventory_parts.csv": "inventory_parts",
        "inventory_minifigs.csv": "inventory_minifigs",
        "inventory_sets.csv": "inventory_sets",
        "minifigs.csv": "minifigs",
    }

    csv_dir = "./datafiles" 

    for filename, table_name in csv_files.items():
        path = os.path.join(csv_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="utf-8")
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"✓ Loaded {table_name} ({len(df)} rows)")
        else:
            print(f"✗ Missing: {filename}")


def validate_db(conn):
    cursor = conn.cursor()

    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    print([row[0] for row in cursor.fetchall()])

    # Sanity check a join
    df = pd.read_sql_query("""
        SELECT s.set_num, s.name, s.year, t.name as theme
        FROM sets s
        JOIN themes t ON s.theme_id = t.id
        LIMIT 5
    """, conn)
    return df

if __name__ == "__main__":
    connection = sqlite3.connect("lego.db")

    # create_db(connection)

    print(validate_db(connection))

    connection.close()