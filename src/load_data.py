import glob
import os

import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql+psycopg2://sentinel:sentinel@localhost:5432/olist"
)

try:
    paths = glob.glob("data/raw/olist_*_dataset.csv")

    if not paths:
        print("No CSV files found in data/raw/")
    else:
        for path in paths:
            table = (
                os.path.basename(path)
                .replace("olist_", "")
                .replace("_dataset.csv", "")
            )

            df = pd.read_csv(path)
            df.to_sql(
                table,
                engine,
                if_exists="replace",
                index=False,
                chunksize=1000,
                method="multi",
            )

            print(f"{table}: {len(df)} rows loaded")

finally:
    engine.dispose()
