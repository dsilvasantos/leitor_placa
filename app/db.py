import psycopg2
from contextlib import contextmanager

@contextmanager
def get_conn():
    conn = psycopg2.connect(
        dbname="placas",
        user="placa",
        password="placa",
        host="localhost",
        port="5432"
    )
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS placas (
                    id SERIAL PRIMARY KEY,
                    placa TEXT NOT NULL,
                    data TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()
