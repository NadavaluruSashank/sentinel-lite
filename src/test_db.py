from sqlalchemy import create_engine, text

DATABASE_URL = (
    "postgresql+psycopg2://sentinel:sentinel@localhost:5432/olist"
)

engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database(), version()")
        )
        database, version = result.fetchone()

        print(f"Connected successfully to: {database}")
        print(f"Database version: {version}")

except Exception as error:
    print(f"Database connection failed: {error}")
    raise

finally:
    engine.dispose()
