import json
import os
from typing import List

import pandas as pd


def read_schema_csv(path: str) -> pd.DataFrame:
    """Load a schema CSV, trying multiple encodings to avoid decode errors."""
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 0, f"Failed to decode {path} with tried encodings")


def extract_original_column_names(df: pd.DataFrame) -> List[str]:
    """Return cleaned original_column_name values, or raise if column missing."""
    normalized_columns = [str(col).strip().lstrip("\ufeff") for col in df.columns]
    df.columns = normalized_columns

    if "original_column_name" not in df.columns:
        raise KeyError("original_column_name")

    return df["original_column_name"].dropna().astype(str).tolist()


SOURCE_JSON = "/root/autodl-tmp/comp/LLaMA-Factory/datasets/minidev/MINIDEV/mini_dev_mysql.json"
COLUMN_DIR = "datasets/minidev/MINIDEV/dev_databases"
OUTPUT_JSONL = "datasets/minidev/mini_dev_mysql_with_schema.jsonl"
OUTPUT_ALPACA = "datasets/minidev/mini_dev_mysql_alpaca.json"
PROMPT = "You are an expert data analyst. Please write a MySQL query that satisfies the requirement below."

df_mini = pd.read_json(SOURCE_JSON)

records = []
missing_schema_dbs = []
failed_tables = []

for _, row in df_mini.iterrows():
    question_id = row["question_id"]
    question = str(row["question"]).strip()
    sql_text = str(row["SQL"]).strip()
    db_id = row["db_id"]

    column_mini = os.path.join(COLUMN_DIR, db_id, "database_description")
    if not os.path.isdir(column_mini):
        missing_schema_dbs.append(db_id)
        column_content: List[str] = []
    else:
        column_content = []
        for item in sorted(os.listdir(column_mini)):
            if not item.lower().endswith(".csv"):
                continue

            column_path = os.path.join(column_mini, item)
            try:
                content = read_schema_csv(column_path)
                names = extract_original_column_names(content)
            except UnicodeDecodeError as err:
                failed_tables.append((column_path, f"decode_error: {err}"))
                continue
            except KeyError:
                failed_tables.append((column_path, "missing_original_column_name"))
                continue

            table_name = os.path.splitext(item)[0]
            columns_str = f"Table {table_name} column names: {', '.join(names)}"
            column_content.append(columns_str)

    if column_content:
        schema_snippet = "\n".join(column_content)
        enriched_question = f"{question}\n{schema_snippet}"
    else:
        enriched_question = question

    records.append(
        {
            "instance_id": f"mini_{question_id}",
            "question_id": question_id,
            "db_id": db_id,
            "question": enriched_question,
            "original_question": question,
            "output": sql_text,
        }
    )

new_df = pd.DataFrame(records)
pd.set_option("display.max_colwidth", None)

os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
new_df.to_json(OUTPUT_JSONL, lines=True, orient="records", force_ascii=False)

alpaca_records = [
    {
        "instruction": PROMPT,
        "input": row["question"],
        "output": row["output"],
    }
    for _, row in new_df.iterrows()
]

with open(OUTPUT_ALPACA, "w", encoding="utf-8") as f:
    json.dump(alpaca_records, f, ensure_ascii=False, indent=2)

print(f"成功样本数：{len(new_df)}，已保存至 {OUTPUT_JSONL}")
print(f"Alpaca 格式样本已保存至 {OUTPUT_ALPACA}")

if missing_schema_dbs:
    unique_missing = sorted(set(missing_schema_dbs))
    print(f"缺少 schema 描述的数据库：{unique_missing}")

if failed_tables:
    print("以下表在解析时出错（仅显示前 10 项）：")
    for info in failed_tables[:10]:
        print(f"  {info[0]} -> {info[1]}")


