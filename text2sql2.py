import json
import os
import http.client
from typing import List, Dict, Tuple

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
TRANSLATION_CACHE = "datasets/minidev/translation_cache.json"

# 通过环境变量配置 API
API_HOST = os.getenv("GPTBEST_API_HOST", "hk-api.gptbest.vip")
API_KEY = os.getenv("GPTBEST_API_KEY","sk-GMYNUCidV96DStXskUpPqgemoaDur0alDXZkeyiq5E3mXGZn")  # 若为空则跳过翻译
FORCE_RETRANSLATE = os.getenv("FORCE_RETRANSLATE", "0") == "1"


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

# 加载翻译缓存（question_id -> {question_cn, evidence_cn}）
if os.path.isfile(TRANSLATION_CACHE) and not FORCE_RETRANSLATE:
    try:
        with open(TRANSLATION_CACHE, "r", encoding="utf-8") as cf:
            translation_cache: Dict[str, Dict[str, str]] = json.load(cf)
    except Exception:
        translation_cache = {}
else:
    translation_cache = {}


def save_cache() -> None:
    os.makedirs(os.path.dirname(TRANSLATION_CACHE), exist_ok=True)
    with open(TRANSLATION_CACHE, "w", encoding="utf-8") as cf:
        json.dump(translation_cache, cf, ensure_ascii=False, indent=2)


def extract_identifiers(schema_sql: str) -> List[str]:
    """从CREATE TABLE片段中提取表名与列名集合，用于提示模型不要翻译这些标识符。"""
    import re
    identifiers = set()
    # 表名
    for m in re.finditer(r"CREATE TABLE\s+`([^`]+)`", schema_sql):
        identifiers.add(m.group(1))
    # 列名（位于反引号内）
    for m in re.finditer(r"\n\s*`([^`]+)`\s+[A-Z]+", schema_sql):
        identifiers.add(m.group(1))
    # 限制数量避免提示过长
    return sorted(list(identifiers))[:120]

def chinese_ratio(text: str) -> float:
    total = len(text)
    if total == 0:
        return 0.0
    cn = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    return cn / total

UNTRANSLATED_THRESHOLD = 0.3  # 中文占比低于此认为未译


def translate_question_and_evidence(qid: str, question: str, evidence: str, identifiers: List[str]) -> Tuple[str, str]:
    """始终尝试翻译(有API_KEY)，使用缓存；保留schema中出现的表/列名不翻译。"""
    if qid in translation_cache and not FORCE_RETRANSLATE:
        cached = translation_cache[qid]
        qc = cached.get("question_cn", question)
        ec = cached.get("evidence_cn", evidence)
        # 若疑似未翻译则继续翻译
        if chinese_ratio(qc) < UNTRANSLATED_THRESHOLD or chinese_ratio(ec) < UNTRANSLATED_THRESHOLD:
            pass
        else:
            return qc, ec

    if not API_KEY:  # 无key直接返回
        translation_cache[qid] = {"question_cn": question, "evidence_cn": evidence}
        save_cache()
        return question, evidence

    # 在 system 提示中明确列出不应翻译的标识符
    id_hint = ", ".join(identifiers) if identifiers else ""
    # 为了让模型更好识别不要翻译的部分，将截断schema一起喂给system提示
    schema_preview = schema_sql if (schema_sql := "\n".join([s for s in identifiers])) else ""
    sys_prompt = (
        "You are a professional translator into Simplified Chinese. Translate ONLY natural language parts. "
        "KEEP ALL technical identifiers (table names, column names, column definitions, SQL keywords, numbers, date strings) EXACTLY AS ORIGINAL. "
        "Identifiers are enclosed by backticks or appear as SQL types. Do NOT translate them. "
        "Return STRICT JSON with keys question_cn and evidence_cn."
    )
    if id_hint:
        sys_prompt += f" Do NOT translate these identifiers: {id_hint}."

    payload_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps({"question": question, "evidence": evidence}, ensure_ascii=False)},
    ]

    req_body = {"model": "gpt-5-mini", "stream": False, "messages": payload_messages}

    try:
        conn = http.client.HTTPSConnection(API_HOST, timeout=2000)
        headers = {"Accept": "application/json", "Authorization": API_KEY, "Content-Type": "application/json"}
        conn.request("POST", "/v1/chat/completions", json.dumps(req_body, ensure_ascii=False).encode("utf-8"), headers)
        res = conn.getresponse()
        raw = res.read().decode("utf-8", errors="ignore")
        data = json.loads(raw)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        try:
            parsed = json.loads(content)
            question_cn = parsed.get("question_cn") or question
            evidence_cn = parsed.get("evidence_cn") or evidence
        except Exception:
            question_cn = question  # 若格式异常则保留原文
            evidence_cn = evidence
            print(f"警告：翻译结果非JSON格式，qid={qid}，内容：{content}")
    except Exception:
        question_cn, evidence_cn = question, evidence
        print(f"警告：翻译请求失败，qid={qid}，保留原文。")

    translation_cache[qid] = {"question_cn": question_cn, "evidence_cn": evidence_cn}
    save_cache()
    return question_cn, evidence_cn

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
    # 从原数据中提取 evidence 字段（若存在），否则为空
    evidence = str(row.get("evidence", "")).strip()

    identifiers = extract_identifiers(schema_snippet)
    # 翻译 question 与 evidence (强制翻译, 保留标识符)
    question_cn, evidence_cn = translate_question_and_evidence(str(question_id), question, evidence, identifiers)

    # 使用中文模板构造最终的 Prompt（要求输出 MySQL 查询）
    prompt_text = TEMPLATE.format(
        dialect="MySQL",
        question=question_cn,
        db_schema=schema_snippet,
        evidence=evidence_cn,
    )

    records.append(
        {
            "instance_id": f"mini_{question_id}",
            "question_id": question_id,
            "db_id": db_id,
            "question": prompt_text,
            "original_question": question,
            "original_evidence": evidence,
            "question_cn": question_cn,
            "evidence_cn": evidence_cn,
            "output": sql_text,
        }
    )

new_df = pd.DataFrame(records)
pd.set_option("display.max_colwidth", None)

os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
new_df.to_json(OUTPUT_JSONL, lines=True, orient="records", force_ascii=False)

alpaca_records = []
for _, row in new_df.iterrows():
    alpaca_records.append(
        {
            "instruction": PROMPT,
            "input": row["question"],  # 已是中文模板成品
            "output": row["output"],
            "meta": {
                "question_id": row["question_id"],
                "db_id": row["db_id"],
                "original_question": row["original_question"],
                "original_evidence": row["original_evidence"],
                "question_cn": row["question_cn"],
                "evidence_cn": row["evidence_cn"],
            },
        }
    )

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


