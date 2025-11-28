"""Dataset augmentation script: question & SQL multi-step decomposition.

功能概述:
1. 读取基础数据集(假设字段包含 question, sql 或 output / table_list / knowledge 等)。
2. 为每条样本随机生成若干不同的“分解版本”(variant)，每个版本随机选择步数范围内的 n_steps。
3. 使用本地 LLM (call_openai) 分解自然语言问题 -> steps JSON。
4. 基于 steps 和最终 SQL, 由 LLM 生成对应的每步 SQL 片段(step_sqls)。
5. 对每个 step SQL 进行执行验证; 若失败, 携带错误信息多轮对话请求 LLM 修订(最多重试 retry_limit)。
6. 记录多轮对话日志, 以及最终验证结果(全部成功/部分失败)。
7. 一个原始样本可扩展出多个 variant, 形成增广数据集。

输出数据结构(保存为 JSON list): 每条增强记录示例:
{
  "original_id": <原问题索引或question_id>,
  "variant_id": "<original_id>-v<k>",
  "question": "...",
  "final_sql": "...",  # 原始最终 SQL
  "steps": ["step1 描述", "step2 描述", ...],
  "step_sqls": ["SELECT ...", "SELECT ...", ...],
  "step_exec_status": ["success", "error"...],
  "step_exec_errors": [null, "错误信息"...],
  "dialogues": [
      {"role": "system", "content": "初始分解指令"},
      {"role": "assistant", "content": "返回的JSON"},
      {"role": "user", "content": "请根据这些步骤生成SQL分段..."},
      ...
  ],
  "verification": {"all_steps_success": true, "failed_steps": []},
  "n_steps": 3
}

运行示例:
python augment_decomposition.py \
  --input_file mini_dev_mysql_alpaca_filtered.json \
  --output_file data/augmented_decomposition.json \
  --db_id mysql_db \
  --min_steps 2 --max_steps 4 \
  --variants_per_question 3 \
  --max_step_retries 2 \
  --limit 10

注意:
- 依赖已有的环境变量: OPENAI_BASE_URL, OPENAI_API_KEY, SR_HOST, SR_PORT, SR_USER, SR_DB。
- 远程执行验证使用 MySQL; 若步骤 SQL 需要依赖临时结构, 请在生成 prompt 中让模型使用可执行的独立查询(必要时使用CTE)。
- 生成的 step SQL 必须是独立可执行的 MySQL 查询, 否则验证失败并触发修订重试。

未来可扩展:
- 增加“部分依赖链”验证: 允许前一步生成临时视图 (当前简化为独立查询)。
- 引入 EXPLAIN 验证与语义覆盖度评估。
"""

from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
from urllib.parse import urlparse
import os
import re
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv(override=True)

from alphasql.llm_call.openai_llm import call_openai, DEFAULT_COST_RECORDER
from alphasql.database.sql_execution import execute_sql_with_pymysql

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

QUESTION_DECOMPOSE_PROMPT = """你是资深SQL分析助手。以下是用户问题与数据库上下文，请把用户问题拆解为 {n_steps} 个循序渐进的子问题。要求:
1. 覆盖原问题的全部意图。
2. 允许“存在依赖”或“互相独立”的混合；对于存在依赖的步骤，标注依赖的上一步索引。
3. 每步描述清晰、可映射到独立的SQL逻辑；若存在依赖，也仅为“逻辑依赖”(每步SQL仍需可单独执行)。
4. 输出严格 JSON，仅包含：{{"steps": [{{"desc": "...", "independent": true/false, "depends_on": [<int>, ...]}}, ...]}}，不要额外文本。
5. 步骤描述用简洁中文，不含具体SQL。

【用户问题】
{user_question}

【数据库schema】
{schema_text}

【参考信息】
{ref_text}
"""

META_PLAN_PROMPT = """你是资深数据分析教练。请根据用户问题与数据库上下文，进行元规划第0步：决定后续子问题的数量与主题。
要求：
1) 总步数为 {n_steps}，其中第0步为元规划，需生成后续 {sub_steps} 个具体子问题标题，按顺序排列；
2) 每个子问题应可映射到独立的SQL逻辑（过滤、聚合、连接等）；
3) 仅输出严格JSON：{{"sub_questions": ["子问题1", "子问题2", ...]}}，数组长度必须为 {sub_steps}，不要额外文本。

【用户问题】
{user_question}

【数据库schema】
{schema_text}

【参考信息】
{ref_text}
"""

STEP_SQL_GENERATE_PROMPT = """你是资深MySQL工程师。根据给定的最终目标SQL与步骤分解, 为每个步骤生成一个可独立执行的 MySQL 查询语句(允许使用CTE)。要求:
1. 每个步骤SQL仅实现该步骤描述对应的中间查询逻辑。
2. 最后一个步骤的SQL必须等价于最终完整SQL。
3. 所有SQL必须可单独执行；若步骤标注有依赖，仅为“逻辑依赖”，请在该步SQL中重写必要条件。
4. 输出严格 JSON: {{"step_sqls": ["SQL1", "SQL2", ...]}} 数量需与 steps 一致。
5. 避免使用非标准MySQL语法, 不要使用临时文件、存储过程。可使用WITH(CTE)。

步骤列表:
{steps_json}

最终目标SQL:
{final_sql}
"""

STEP_SQL_ONE_PROMPT = """你是资深MySQL工程师。请仅针对以下单个步骤，生成一个可独立执行的 MySQL 查询语句（允许使用CTE）。
严格遵守：
1) 该SQL只实现该步骤描述的中间逻辑，不要提前输出最终答案；
2) is_final_step={is_final_step}。当 is_final_step=false 时，禁止使用 COUNT/聚合或完整跨窗口连接；优先返回中间结果集合（例如 SELECT DISTINCT vplayerid ...）；
3) 使用标准MySQL语法；
4) 仅输出严格JSON：{{\"sql\": \"...\"}}，不要额外文本。

【步骤描述】
{step_desc}

【用户问题】
{user_question}

【数据库schema】
{schema_text}

【参考信息】
{ref_text}
"""

SCOPE_REVISION_PROMPT = """你给出的SQL范围过大或等价于最终答案。请仅返回该步骤的中间查询SQL：
约束：
1) 当该步骤不是最后一步时，不要使用 COUNT/聚合；不要进行完整跨窗口连接。
2) 仅根据该步骤的描述生成最小可执行的查询（例如返回对应窗口的 SELECT DISTINCT vplayerid，或在必要时使用 NOT EXISTS 排除另一窗口）。
返回严格JSON：{{\"sql\": \"...\"}}，不要额外文本。

【步骤描述】
{step_desc}

【当前SQL（过度范围）】
{current_sql}
"""

REVISION_PROMPT = """你生成的某步SQL执行失败。请参考:
步骤描述: {step_desc}
原始失败SQL: {failed_sql}
错误信息: {error_msg}
请返回一个新的可执行修正版 SQL (单条查询)，仅返回 JSON: {{"revised_sql": "..."}} 不要解释。"""

FINAL_SYNTHESIS_PROMPT = """你是资深MySQL工程师。请根据以下信息一次性生成满足用户需求的最终SQL：
1) 用户问题与上下文(数据库schema与参考信息)
2) 分解后的步骤与对应的(已验证/修订后)SQL片段
要求：
- 最终SQL可在当前数据库上独立执行；
- 语义与用户原始需求一致；
- 使用标准MySQL语法；
- 仅输出 JSON: {{"final_sql": "..."}} 不要额外文本。

【用户问题】
{user_question}

【数据库schema】
{schema_text}

【参考信息】
{ref_text}

【步骤与SQL片段】
{steps_and_sqls}
"""

def safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


def parse_cn_prompt_blocks(text: str) -> Dict[str, str]:
    """Robust parser for Chinese labeled sections.
    Extract content between explicit markers:
    - 【用户问题】 ... 【数据库schema】
    - 【数据库schema】 ... 【参考信息】
    - 【参考信息】 ... (to end)
    Falls back gracefully if markers missing.
    """
    if not text:
        return {"user_question": "", "schema_text": "", "ref_text": ""}
    # Normalize line endings
    t = text.replace('\r\n', '\n').replace('\r', '\n')
    def _segment(start_marker: str, end_marker: str | None) -> str:
        s_idx = t.find(start_marker)
        if s_idx == -1:
            return ""
        s_idx += len(start_marker)
        if end_marker:
            e_idx = t.find(end_marker, s_idx)
            content = t[s_idx:e_idx if e_idx != -1 else len(t)]
        else:
            content = t[s_idx:]
        return content.strip()

    user_q = _segment("【用户问题】\n", "【数据库schema】")
    if not user_q:
        # try without newline
        user_q = _segment("【用户问题】", "【数据库schema】")
    schema_text = _segment("【数据库schema】\n", "【参考信息】")
    if not schema_text:
        schema_text = _segment("【数据库schema】", "【参考信息】")
    ref_text = _segment("【参考信息】\n", None)
    if not ref_text:
        ref_text = _segment("【参考信息】", None)

    # Clean trailing hints like "输出：玩家数" from user question if present
    # Split by lines and stop at first empty line
    if user_q:
        lines = [ln.strip() for ln in user_q.split('\n')]
        # remove label lines like "输出：..."
        filtered = []
        for ln in lines:
            if ln.startswith("输出："):
                continue
            filtered.append(ln)
        # keep contiguous non-empty lines at start
        user_q = "\n".join([ln for ln in filtered if ln])

    return {"user_question": user_q.strip(), "schema_text": schema_text.strip(), "ref_text": ref_text.strip()}

def offline_plan_steps(user_q: str, n_steps: int) -> List[Dict[str, Any]]:
    # 显式加入第0步为元规划(meta)，不生成SQL，其余为具体子问题
    plan: List[Dict[str, Any]] = []
    # 第0步(meta)
    plan.append({
        "desc": f"第0步：决定子问题数量与主题，围绕原问题进行分解：{user_q}",
        "independent": True,
        "depends_on": [],
        "meta": True
    })
    # 后续具体步骤数量为 n_steps-1，至少为1
    sub_steps = max(1, n_steps - 1)
    for i in range(sub_steps):
        dep = random.random() < 0.5
        plan.append({
            "desc": f"具体子问题(第{i+1}步)：细化一个可执行的查询条件/聚合逻辑",
            "independent": not dep,
            "depends_on": [i] if dep else [],
            "meta": False
        })
    return plan

def decompose_question(context: Dict[str, str], n_steps: int, llm_model: str, temperature: float, offline: bool = False) -> List[Dict[str, Any]]:
    user_q = context.get("user_question", "")
    if offline:
        return offline_plan_steps(user_q, n_steps)
    # 在线模式：先进行元规划，得到具体子问题标题，再形成steps_plan
    sub_steps = max(1, n_steps - 1)
    meta_prompt = META_PLAN_PROMPT.format(
        n_steps=n_steps,
        sub_steps=sub_steps,
        user_question=user_q,
        schema_text=context.get("schema_text", ""),
        ref_text=context.get("ref_text", "")
    )
    try:
        meta_resp = call_openai(prompt=meta_prompt, model=llm_model, temperature=temperature, n=1, max_tokens=512)
        meta_data = safe_json_loads(meta_resp[0]) or {}
        titles = meta_data.get("sub_questions", [])
    except Exception:
        titles = []
    if not isinstance(titles, list) or len(titles) != sub_steps:
        # 兜底：使用离线规则生成
        return offline_plan_steps(user_q, n_steps)
    steps_plan: List[Dict[str, Any]] = []
    steps_plan.append({
        "desc": f"第0步：决定子问题数量与主题，围绕原问题进行分解：{user_q}",
        "independent": True,
        "depends_on": [],
        "meta": True
    })
    for i, title in enumerate(titles):
        dep = random.random() < 0.5
        steps_plan.append({
            "desc": f"{title}",
            "independent": not dep,
            "depends_on": [i] if dep else [],
            "meta": False
        })
    return steps_plan

def generate_single_step_sql(step: Dict[str, Any], is_final_step: bool, final_sql: str, llm_model: str, temperature: float,
                              offline: bool, context: Dict[str, str]) -> str:
    """生成单步SQL：离线直接返回final或空；在线使用单步提示并做范围约束。"""
    if step.get("meta"):
        return ""
    if offline:
        return final_sql if is_final_step else f"SELECT DISTINCT vplayerid FROM dws_jordass_matchlog_stat_di LIMIT 50"  # 简单占位中间结果
    one_prompt = STEP_SQL_ONE_PROMPT.format(
        step_desc=step.get("desc", ""),
        user_question=context.get("user_question", ""),
        schema_text=context.get("schema_text", ""),
        ref_text=context.get("ref_text", ""),
        is_final_step=str(is_final_step).lower()
    )
    try:
        resp = call_openai(prompt=one_prompt, model=llm_model, temperature=temperature, n=1, max_tokens=1024)
        data = safe_json_loads(resp[0]) or {}
        sql = data.get("sql")
    except Exception:
        sql = None
    if not sql:
        sql = final_sql if is_final_step else "SELECT DISTINCT vplayerid FROM dws_jordass_matchlog_stat_di WHERE dtstatdate >= '20240101' LIMIT 100"
    else:
        # 过度范围约束：非最终步不允许聚合或等价最终SQL
        if not is_final_step:
            norm_final = re.sub(r"\s+", " ", final_sql.strip()).lower()
            norm_sql = re.sub(r"\s+", " ", sql.strip()).lower()
            if norm_sql == norm_final or "count(" in norm_sql or "group by" in norm_sql:
                try:
                    scope_prompt = SCOPE_REVISION_PROMPT.format(step_desc=step.get("desc", ""), current_sql=sql)
                    resp2 = call_openai(prompt=scope_prompt, model=llm_model, temperature=temperature, n=1, max_tokens=512)
                    data2 = safe_json_loads(resp2[0]) or {}
                    sql2 = data2.get("sql")
                    if sql2:
                        sql = sql2
                except Exception:
                    pass
    return sql

def validate_step_sql(sql_text: str, timeout: int = 60) -> Dict[str, Any]:
    result = execute_sql_with_pymysql(sql_text, timeout=timeout)
    return {
        "status": result.result_type.value,
        "error": result.error_message,
        "rows_sample": result.result[:3] if result.result else []
    }

def revise_step_sql(step_desc: str, failed_sql: str, error_msg: str, llm_model: str, temperature: float, offline: bool = False) -> str:
    if offline:
        return failed_sql
    prompt = REVISION_PROMPT.format(step_desc=step_desc, failed_sql=failed_sql, error_msg=error_msg)
    responses = call_openai(prompt=prompt, model=llm_model, temperature=temperature, n=1, max_tokens=512)
    data = safe_json_loads(responses[0]) or {}
    revised = data.get("revised_sql") or failed_sql
    return revised

def synthesize_final_sql(context: Dict[str, str], steps: List[Dict[str, Any]], validated_sqls: List[str], fallback_final: str, llm_model: str, temperature: float, offline: bool = False) -> str:
    if offline:
        return fallback_final
    bundle = json.dumps({"steps": steps, "step_sqls": validated_sqls}, ensure_ascii=False)
    prompt = FINAL_SYNTHESIS_PROMPT.format(
        user_question=context.get("user_question", ""),
        schema_text=context.get("schema_text", ""),
        ref_text=context.get("ref_text", ""),
        steps_and_sqls=bundle
    )
    responses = call_openai(prompt=prompt, model=llm_model, temperature=temperature, n=1, max_tokens=2048)
    data = safe_json_loads(responses[0]) or {}
    return data.get("final_sql") or fallback_final

def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", (sql or "").strip()).lower()

def process_one_record(record: Dict[str, Any], args) -> List[Dict[str, Any]]:
    # 解析输入结构：优先从 input 中提取中文分段
    instruction = record.get("instruction") or record.get("question") or ""
    input_text = record.get("input") or ""
    ctx = parse_cn_prompt_blocks(input_text)
    if not ctx.get("user_question"):
        ctx["user_question"] = instruction
    final_sql = record.get("sql") or record.get("output") or ""
    if not final_sql or not ctx.get("user_question"):
        return []
    variants = []
    for vidx in range(args.variants_per_question):
        n_steps = random.randint(args.min_steps, args.max_steps)
        steps_plan = decompose_question(ctx, n_steps, args.llm_model, args.decompose_temperature, offline=args.offline)
        # 找到最后一个非meta步索引用于标记最终步
        last_non_meta_idx = max(i for i, st in enumerate(steps_plan) if not st.get("meta")) if any(not s.get("meta") for s in steps_plan) else -1
        step_sqls: List[str] = []

        dialogues = []
        dialogues.append({"role": "system", "content": f"运行上下文: endpoint_type={args.endpoint_type_resolved}, model={args.model_used}, base_url={args.openai_base_url}"})
        dialogues.append({"role": "user", "content": f"请依据用户问题与数据库上下文，将问题拆解为 {n_steps} 步并标注依赖。"})
        dialogues.append({"role": "assistant", "content": json.dumps({"steps": steps_plan}, ensure_ascii=False)})
        # 逐步对话 & 工具调用
        for s_idx, s_plan in enumerate(steps_plan):
            if s_plan.get("meta"):
                step_sqls.append("")
                dialogues.append({"role": "user", "content": f"第{s_idx}步(meta)规划完成，继续下一步。"})
                dialogues.append({"role": "assistant", "content": json.dumps({"ack_meta": True}, ensure_ascii=False)})
                continue
            is_final_step = (s_idx == last_non_meta_idx)
            dialogues.append({"role": "user", "content": f"请生成第{s_idx}步SQL (is_final_step={is_final_step})：{s_plan.get('desc','')}"})
            sql_one = generate_single_step_sql(s_plan, is_final_step, final_sql, args.llm_model, args.sqlgen_temperature, args.offline, ctx)
            step_sqls.append(sql_one)
            dialogues.append({"role": "assistant", "content": json.dumps({"sql": sql_one}, ensure_ascii=False)})
            # 执行工具调用（延后统一在验证循环内记录 tool 输出）

        step_exec_status = []
        step_exec_errors = []
        validated_sqls = []
        for s_idx, (s_plan, s_sql) in enumerate(zip(steps_plan, step_sqls)):
            # 跳过meta步骤（不执行SQL验证）
            if s_plan.get("meta"):
                step_exec_status.append("meta")
                step_exec_errors.append(None)
                validated_sqls.append("")
                continue
            current_sql = s_sql
            for attempt in range(args.max_step_retries + 1):
                exec_result = validate_step_sql(current_sql)
                # 记录工具调用日志
                dialogues.append({"role": "tool", "content": json.dumps({"step_index": s_idx, "attempt": attempt+1, "executed_sql": current_sql, "status": exec_result['status'], "error": exec_result['error'], "rows_sample": exec_result['rows_sample']}, ensure_ascii=False)})
                if exec_result["status"] == "success":
                    step_exec_status.append("success")
                    step_exec_errors.append(None)
                    validated_sqls.append(current_sql)
                    break
                else:
                    if attempt < args.max_step_retries:
                        dialogues.append({"role": "user", "content": f"第{s_idx}步SQL执行失败,错误:{exec_result['error']} 请修订仅返回JSON"})
                        revised_sql = revise_step_sql(s_plan.get('desc',''), current_sql, exec_result["error"] or "Execution error", args.llm_model, args.revise_temperature, offline=args.offline)
                        dialogues.append({"role": "assistant", "content": json.dumps({"revised_sql": revised_sql}, ensure_ascii=False)})
                        # 记录纠错到独立文件（若提供）
                        try:
                            if getattr(args, 'corrections_file', None):
                                corr_entry = {
                                    "original_id": record.get("question_id"),
                                    "variant_id": f"{record.get('question_id','unknown')}-v{vidx}",
                                    "step_index": s_idx,
                                    "attempt": attempt + 1,
                                    "desc": s_plan.get('desc',''),
                                    "prev_sql": current_sql,
                                    "error": exec_result.get("error"),
                                    "revised_sql": revised_sql,
                                    "llm_model": args.model_used,
                                    "endpoint_type": args.endpoint_type_resolved,
                                    "timestamp": int(time.time())
                                }
                                # 以追加的方式写入JSON Lines
                                Path(args.corrections_file).parent.mkdir(parents=True, exist_ok=True)
                                with open(args.corrections_file, 'a', encoding='utf-8') as fw:
                                    fw.write(json.dumps(corr_entry, ensure_ascii=False) + "\n")
                        except Exception:
                            pass
                        current_sql = revised_sql
                        time.sleep(0.5)
                    else:
                        step_exec_status.append("error")
                        step_exec_errors.append(exec_result["error"])
                        validated_sqls.append(current_sql)
                        break

        # 最终汇合生成SQL
        dialogues.append({"role": "user", "content": "请基于步骤与其验证后的SQL片段生成最终SQL，仅返回JSON。"})
        final_sql_generated = synthesize_final_sql(ctx, steps_plan, validated_sqls, final_sql, args.llm_model, args.sqlgen_temperature, offline=args.offline)
        dialogues.append({"role": "assistant", "content": json.dumps({"final_sql": final_sql_generated}, ensure_ascii=False)})

        all_success = all(st == "success" for st in step_exec_status)
        match_flag = normalize_sql(final_sql_generated) == normalize_sql(final_sql)
        variant = {
            "original_id": record.get("question_id"),
            "variant_id": f"{record.get('question_id','unknown')}-v{vidx}",
            "question": ctx.get("user_question", ""),
            "schema": ctx.get("schema_text", ""),
            "reference": ctx.get("ref_text", ""),
            "final_sql": final_sql,
            "steps": [s.get("desc", "") for s in steps_plan],
            "step_plan": steps_plan,
            "step_sqls": validated_sqls,
            "step_exec_status": step_exec_status,
            "step_exec_errors": step_exec_errors,
            "dialogues": dialogues,
            "verification": {"all_steps_success": all_success, "failed_steps": [i for i,s in enumerate(step_exec_status) if s != 'success']},
            "n_steps": n_steps,
            "final_sql_generated": final_sql_generated,
            "final_sql_match": match_flag,
            "llm_model": args.model_used,
            "endpoint_type": args.endpoint_type_resolved,
            "openai_base_url": args.openai_base_url
        }
        variants.append(variant)
    return variants

def load_input_dataset(path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(data, dict):
        # 如果是 {id: sql} 结构，转换为列表
        new = []
        for k,v in data.items():
            new.append({"question_id": int(k), "question": "", "sql": v})
        return new
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--db_id', type=str, default='mysql_db')
    parser.add_argument('--min_steps', type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=4)
    parser.add_argument('--variants_per_question', type=int, default=2)
    parser.add_argument('--max_step_retries', type=int, default=2)
    parser.add_argument('--limit', type=int, default=-1, help='只处理前N条，-1为全部')
    parser.add_argument('--llm_model', type=str, default='Qwen3-Coder-30B-A3B-Instruct')
    parser.add_argument('--decompose_temperature', type=float, default=0.2)
    parser.add_argument('--sqlgen_temperature', type=float, default=0.3)
    parser.add_argument('--revise_temperature', type=float, default=0.1)
    parser.add_argument('--offline', action='store_true', help='离线模式: 跳过LLM调用, 使用规则兜底生成')
    parser.add_argument('--endpoint_type', type=str, default='auto', choices=['auto','online','vllm'], help='显式标注端点类型; auto会基于OPENAI_BASE_URL推断')
    parser.add_argument('--corrections_file', type=str, default='', help='将修订尝试记录为JSON Lines到该文件(可选)')
    parser.add_argument('--openai_base_url', type=str, default='', help='覆盖环境变量OPENAI_BASE_URL (可选)')
    parser.add_argument('--openai_api_key', type=str, default='', help='覆盖环境变量OPENAI_API_KEY (可选)')
    args = parser.parse_args()

    # 覆盖环境变量（若提供）
    if args.openai_base_url:
        os.environ['OPENAI_BASE_URL'] = args.openai_base_url
    if args.openai_api_key:
        os.environ['OPENAI_API_KEY'] = args.openai_api_key

    # 端点类型与模型标注
    base_url = os.getenv('OPENAI_BASE_URL', '')
    parsed = urlparse(base_url) if base_url else None
    inferred = 'unknown'
    if parsed and parsed.hostname:
        host = parsed.hostname
        if host in ('127.0.0.1','localhost'):
            inferred = 'vllm'
        else:
            inferred = 'online'
    endpoint_resolved = 'offline' if args.offline else (args.endpoint_type if args.endpoint_type != 'auto' else inferred)
    model_used = 'offline-rules' if args.offline else args.llm_model

    # 挂到args用于下游写入与日志（优先使用命令行覆盖值）
    args.openai_base_url = args.openai_base_url or base_url
    args.endpoint_type_resolved = endpoint_resolved
    args.model_used = model_used

    print(f"[augment] endpoint_type={endpoint_resolved}, model={model_used}, base_url={base_url}")

    data = load_input_dataset(args.input_file)
    if args.limit != -1:
        data = data[:args.limit]

    augmented: List[Dict[str, Any]] = []
    for rec in data:
        augmented.extend(process_one_record(rec, args))

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_file).write_text(json.dumps(augmented, ensure_ascii=False, indent=2))
    print(f"Augmented records: {len(augmented)} saved -> {args.output_file}")
    DEFAULT_COST_RECORDER.print_profile()

if __name__ == '__main__':
    main()
