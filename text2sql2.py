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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip BOM/空格，标准化列名。"""
    normalized_columns = [str(col).strip().lstrip("\ufeff") for col in df.columns]
    df.columns = normalized_columns
    return df


def build_create_table_stmt(table_name: str, df: pd.DataFrame) -> str:
    """基于 schema CSV 构造 CREATE TABLE 语句，包含列名、类型和注释。"""
    df = normalize_columns(df.copy())

    def get_col(name: str) -> pd.Series | None:
        return df[name] if name in df.columns else None

    col_name_col = get_col("original_column_name")
    col_desc_col = get_col("column_description")
    data_fmt_col = get_col("data_format")
    value_desc_col = get_col("value_description")

    if col_name_col is None:
        raise KeyError("original_column_name")

    col_defs = []
    for idx, raw_name in col_name_col.dropna().astype(str).items():
        name = raw_name.strip()
        if not name:
            continue

        # 类型处理
        dtype = "TEXT"
        if data_fmt_col is not None:
            fmt = str(data_fmt_col.iloc[idx]).strip()
            if fmt and fmt.lower() != "nan":
                fmt_lower = fmt.lower()
                if "int" in fmt_lower:
                    dtype = "BIGINT"
                elif "real" in fmt_lower or "double" in fmt_lower or "float" in fmt_lower:
                    dtype = "DOUBLE"
                else:
                    dtype = "TEXT"

        # 注释处理
        comments = []
        if col_desc_col is not None:
            desc = str(col_desc_col.iloc[idx]).strip()
            if desc and desc.lower() != "nan":
                comments.append(desc)

        if value_desc_col is not None:
            vdesc = str(value_desc_col.iloc[idx]).strip()
            if vdesc and vdesc.lower() != "nan":
                comments.append(vdesc)

        comment_part = ""
        if comments:
            # 简单的转义单引号
            joined_comment = "; ".join(comments).replace("'", "")
            comment_part = f" COMMENT '{joined_comment}'"

        col_defs.append(f"  `{name}` {dtype}{comment_part}")

    body = ",\n".join(col_defs)
    return f"CREATE TABLE `{table_name}` (\n{body}\n);"


SOURCE_JSON = "/root/autodl-tmp/comp/LLaMA-Factory/datasets/minidev/MINIDEV/mini_dev_mysql.json"
COLUMN_DIR = "datasets/minidev/MINIDEV/dev_databases"
OUTPUT_JSONL = "datasets/minidev/mini_dev_mysql_with_schema.jsonl"
OUTPUT_ALPACA = "datasets/minidev/mini_dev_mysql_alpaca.json"
PROMPT = "请你接下来一步步思考，写出正确的SQL查询语句以满足用户的需求。"

# nl2sqlite 中文模板（MySQL 方言）
TEMPLATE = """你是一名{dialect}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用{dialect}知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```sql"""

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
            table_name = os.path.splitext(item)[0]
            try:
                content = read_schema_csv(column_path)
                create_table_sql = build_create_table_stmt(table_name, content)
            except UnicodeDecodeError as err:
                failed_tables.append((column_path, f"decode_error: {err}"))
                continue
            except KeyError:
                failed_tables.append((column_path, "missing_original_column_name"))
                continue

            column_content.append(create_table_sql)

    schema_snippet = "\n".join(column_content) if column_content else ""
    evidence = ""
    # 使用中文模板构造最终的 Prompt（要求输出 MySQL 查询）
    prompt_text = TEMPLATE.format(
        dialect="MySQL",
        question=question,
        db_schema=schema_snippet,
        evidence=evidence,
    )

    records.append(
        {
            "instance_id": f"mini_{question_id}",
            "question_id": question_id,
            "db_id": db_id,
            "question": prompt_text,
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


