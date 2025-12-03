# -*- coding: utf-8 -*-

"""Agent-style question decomposition and SQL augmentation.

This refactor mirrors the ``spider-agent-tc`` structure by splitting the logic
into reusable components:

* ``LLMClient`` centralises OpenAI-compatible calls with retries.
* ``StepPlanner`` emits a meta planning step plus concrete sub-steps.
* ``StepExecutor`` treats SQL execution as a tool call and performs sequential
  retries with LLM-based revisions.
* ``AugmentationAgent`` streams dataset variants to disk with a JSON writer,
  collecting the full agent dialogue for later inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

from alphasql.database.sql_execution import execute_sql_with_pymysql
from alphasql.llm_call.openai_llm import DEFAULT_COST_RECORDER, call_openai

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Prompts (concise, JSON enforced to match spider-agent conventions)
# ---------------------------------------------------------------------------

META_PLAN_PROMPT = """你是资深数据分析教练。请依据下面的【用户问题】完成第0步的元规划：
1) 总步数为 {n_steps}，第0步固定为元规划；输出 {sub_steps} 个按顺序排列的子问题标题；
2) 子问题需覆盖过滤、关联、集合构建、聚合/排序等关键逻辑，并引用上下文实体；
3) 严格输出 JSON：{{"sub_questions": ["子问题1", "子问题2", ...]}}，长度必须为 {sub_steps}。

【用户问题】
{user_question}

【数据库schema】
{schema_text}

【参考信息】
{ref_text}
"""

STEP_SQL_ONE_PROMPT = """你是资深MySQL工程师。仅针对以下单步，生成可独立执行的查询（允许CTE）。
约束：
1) is_final_step={is_final_step}。若为 false，禁止 COUNT/聚合/完整答案，仅返回中间集合；
2) 若为 true，输出必须等价于最终MySQL；
3) 严格输出 JSON：{{"sql": "..."}}。

【步骤描述】
{step_desc}

【用户问题】
{user_question}

【数据库schema】
{schema_text}

【上下文提示】
{ref_text}
"""

SCOPE_REVISION_PROMPT = """当前MySQL覆盖范围过大或提前给出最终答案。请仅返回该步骤需要的最小中间查询，禁止聚合。
输出 JSON：{{"sql": "..."}}。

【步骤描述】
{step_desc}

【过度SQL】
{current_sql}
"""

REVISION_PROMPT = """你生成的MySQL是错误的。请根据错误信息修订MySQL并输出 JSON：{{"revised_sql": "..."}}。

【步骤描述】
{step_desc}

【用户问题】
{user_question}

【数据库schema】
{schema_text}

【参考信息】
{ref_text}

【失败SQL】
{failed_sql}

【错误信息】
{error_msg}
"""

FINAL_SYNTHESIS_PROMPT = """你是资深MySQL工程师。请结合已验证的步骤SQL生成满足原始需求的最终SQL，仅输出 JSON：{{"final_sql": "..."}}。

【用户问题】
{user_question}

【数据库schema】
{schema_text}

【参考信息】
{ref_text}

【步骤与SQL】
{steps_and_sqls}
"""

REFLECTION_STEP_PROMPT = """你是资深MySQL专家。
【用户问题】
{user_question}

【数据库schema】
{schema_text}

【标准答案SQL】
{final_sql}

【所有步骤及其SQL（供参考，避免重复劳动）】
{all_steps_and_sqls}

【当前步骤】
步骤{step_idx}: {step_desc}

【当前生成的MySQL】
{current_sql}

要求：
1) 当前步骤SQL应对应本步骤的意图，尽量不要与其它步骤已经在做的事情完全重复；
2) 如果不可避免有重合，也要保证本步骤仍然有独立价值。

请判断当前SQL是否正确（逻辑上是否是标准答案的一个中间步骤，或者与标准答案在当前步骤的意图一致）。
如果正确，请输出 JSON：{{"is_correct": true}}
如果不正确，请输出修正后的SQL JSON：{{"is_correct": false, "revised_sql": "..."}}
"""

EXECUTOR_SYSTEM_PROMPT = "你是资深MySQL工程师，会结合全部历史对话逐步生成和修正各子问题的SQL，所有回答必须是JSON格式。"
REFLECTION_SYSTEM_PROMPT = "你是资深MySQL专家，会参考历史对话逐轮反思并修正每个步骤的SQL。"

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StepPlan:
    desc: str
    independent: bool
    depends_on: List[int]
    meta: bool = False


@dataclass
class StepSQLResult:
    plan: StepPlan
    sql: str
    status: str
    error: Optional[str] = None
    rows_sample: List[Any] = field(default_factory=list)
    attempts: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    # Attempt to extract JSON from markdown code blocks
    pattern = r"```(?:json)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict):
                return data
        except Exception:
            continue

    # Attempt to find the first JSON object in the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Fallback: try parsing the entire text
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return None


def parse_cn_prompt_blocks(text: str) -> Dict[str, str]:
    if not text:
        return {"user_question": "", "schema_text": "", "ref_text": ""}
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    def _segment(start: str, end: Optional[str]) -> str:
        s_idx = normalized.find(start)
        if s_idx == -1:
            return ""
        s_idx += len(start)
        e_idx = normalized.find(end, s_idx) if end else -1
        return normalized[s_idx:e_idx if e_idx != -1 else len(normalized)].strip()

    user_q = _segment("【用户问题】\n", "【数据库schema】") or _segment("【用户问题】", "【数据库schema】")
    schema_text = _segment("【数据库schema】\n", "【参考信息】") or _segment("【数据库schema】", "【参考信息】")
    ref_text = _segment("【参考信息】\n", None) or _segment("【参考信息】", None)

    if user_q:
        filtered = [ln.strip() for ln in user_q.split('\n') if not ln.startswith("输出：")]
        user_q = "\n".join([ln for ln in filtered if ln])

    return {
        "user_question": user_q.strip(),
        "schema_text": schema_text.strip(),
        "ref_text": ref_text.strip(),
    }


# ---------------------------------------------------------------------------
# LLM + tool wrappers (spider-agent style abstractions)
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, model: str, default_temperature: float, offline: bool = False, max_retries: int = 6):
        self.model = model
        self.default_temperature = default_temperature
        self.offline = offline
        self.max_retries = max_retries

    def invoke(
        self,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        if self.offline:
            raise RuntimeError("LLM not available in offline mode")
        if messages is None and not prompt:
            raise ValueError("LLMClient.invoke requires either prompt or messages")
        temp = temperature if temperature is not None else self.default_temperature
        for attempt in range(1, self.max_retries + 1):
            try:
                responses = call_openai(
                    prompt=prompt or (messages[-1]["content"] if messages else ""),
                    model=self.model,
                    temperature=temp,
                    n=1,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                if responses:
                    return responses[0]
            except Exception as exc:  # pragma: no cover
                print(f"[LLM] call failed ({attempt}/{self.max_retries}): {exc}")
                time.sleep(0.4 * attempt)
        raise RuntimeError("LLM invocation exceeded retry limit")


class SQLExecutionTool:
    def run(self, sql_text: str, timeout: int = 60) -> Dict[str, Any]:
        try:
            result = execute_sql_with_pymysql(sql_text, timeout=timeout)
            return {
                "status": result.result_type.value,
                "error": result.error_message,
                "rows_sample": result.result[:3] if result.result else [],
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Tool execution exception: {str(e)}",
                "rows_sample": [],
            }


# ---------------------------------------------------------------------------
# Planning logic
# ---------------------------------------------------------------------------

class StepPlanner:
    def __init__(self, llm_client: Optional[LLMClient]):
        self.llm_client = llm_client

    def plan(self, context: Dict[str, str], n_steps: int, final_sql: str, offline: bool) -> List[StepPlan]:
        print("[planner] planning steps...")
        if offline or not self.llm_client:
            print("[planner] offline mode or no LLM client, using heuristic planning")
            return self._offline_plan(context, n_steps, final_sql)
        sub_steps = max(1, n_steps - 1)
        prompt = META_PLAN_PROMPT.format(
            n_steps=n_steps,
            sub_steps=sub_steps,
            user_question=context.get("user_question", ""),
            schema_text=context.get("schema_text", ""),
            ref_text=context.get("ref_text", ""),
        )
        print("[planner] prompt {}...".format(prompt))
        try:
            response = self.llm_client.invoke(prompt, max_tokens=1024)
            print("[planner] meta planning response: {}".format(response))
            titles = (safe_json_loads(response) or {}).get("sub_questions", [])
            if isinstance(titles, list) and len(titles) == sub_steps:
                return self._wrap_titles(context, titles)
        except Exception as exc:
            print(f"[planner] meta planning failed, fallback to heuristics: {exc}")
        return self._offline_plan(context, n_steps, final_sql)

    def _wrap_titles(self, context: Dict[str, str], titles: List[str]) -> List[StepPlan]:
        plans = [self._meta_step(context.get("user_question", ""))]
        for idx, title in enumerate(titles):
            dep = random.random() < 0.4
            plans.append(
                StepPlan(
                    desc=title,
                    independent=not dep,
                    depends_on=[idx] if dep else [],
                )
            )
        return plans

    def _offline_plan(self, context: Dict[str, str], n_steps: int, final_sql: str) -> List[StepPlan]:
        user_q = context.get("user_question", "")
        titles = self._heuristic_titles(final_sql, n_steps - 1)
        plans = [self._meta_step(user_q)]
        for idx, title in enumerate(titles):
            dep = random.random() < 0.4
            plans.append(
                StepPlan(
                    desc=title,
                    independent=not dep,
                    depends_on=[idx] if dep else [],
                )
            )
        return plans

    @staticmethod
    def _meta_step(user_q: str) -> StepPlan:
        return StepPlan(
            desc=f"第0步：决定子问题数量与主题，围绕原问题进行分解：{user_q}",
            independent=True,
            depends_on=[],
            meta=True,
        )

    @staticmethod
    def _heuristic_titles(final_sql: str, need: int) -> List[str]:
        sql = (final_sql or "").lower()
        buckets: List[str] = []
        if re.search(r"dtstatdate|date|statis_date", sql):
            buckets.append("按时间窗口过滤记录")
        if re.search(r"sgamecode|splattype|saccounttype|suseridtype", sql):
            buckets.append("筛选指定游戏/平台/账号类型")
        if " join " in sql:
            buckets.append("关联用户/维度表，补充字段")
        if "distinct" in sql:
            buckets.append("形成去重后的候选集合")
        if re.search(r"group by|count\(|sum\(|avg\(", sql):
            buckets.append("对候选集合进行聚合统计")
        if "order by" in sql:
            buckets.append("按指定字段排序并输出")
        if not buckets:
            buckets = ["识别核心过滤条件", "构建候选集合", "整合并输出结果"]
        deduped: List[str] = []
        for item in buckets:
            if item not in deduped:
                deduped.append(item)
        while len(deduped) < max(1, need):
            deduped.append(f"补充条件细化（步骤{len(deduped)}）")
        return deduped[: max(1, need)]


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------

class StepExecutor:
    def __init__(self, llm_client: Optional[LLMClient], sql_tool: SQLExecutionTool, args):
        self.llm_client = llm_client
        self.sql_tool = sql_tool
        self.args = args
        self.chat_history: List[Dict[str, str]] = []

    def reset_history(self, system_prompt: Optional[str] = None):
        self.chat_history = []
        base_prompt = system_prompt or EXECUTOR_SYSTEM_PROMPT
        if base_prompt:
            self.chat_history.append({"role": "system", "content": base_prompt})

    def _history_snapshot(self) -> List[Dict[str, str]]:
        return [msg.copy() for msg in self.chat_history]

    def _chat_completion(self, user_prompt: str, temperature: float, max_tokens: int) -> str:
        if self.llm_client is None:
            raise RuntimeError("LLM client unavailable for chat completion")
        self.chat_history.append({"role": "user", "content": user_prompt})
        response = self.llm_client.invoke(
            temperature=temperature,
            max_tokens=max_tokens,
            messages=self._history_snapshot(),
            prompt=user_prompt,
        )
        self.chat_history.append({"role": "assistant", "content": response})
        return response

    def run_step(
        self,
        step: StepPlan,
        context: Dict[str, str],
        final_sql: str,
        is_final_step: bool,
        validated_sqls: List[str],
        dialogues: List[Dict[str, Any]],
        variant_id: str,
    ) -> StepSQLResult:
        if step.meta:
            dialogues.append({"role": "user", "content": "[META] 第0步：高层规划与子问题分解。"})
            dialogues.append({"role": "assistant", "content": json.dumps({"ack_meta": True}, ensure_ascii=False)})
            return StepSQLResult(step, sql="", status="meta", rows_sample=[])

        enriched_context = context.copy()
        if validated_sqls:
            enriched_context["ref_text"] = (
                context.get("ref_text", "")
                + "\n"
                + json.dumps({"validated_prev_sqls": [sql for sql in validated_sqls if sql]}, ensure_ascii=False)
            ).strip()

        # 第1轮：生成并内部自修正，直到能正常执行或达到最大重试
        sql_candidate = self._generate_sql(step, enriched_context, final_sql, is_final_step, dialogues)
        attempts: List[Dict[str, Any]] = []
        current_sql = sql_candidate
        success = False
        last_error: Optional[str] = None
        last_rows_sample: List[Any] = []

        # 在单轮对话中，允许多次“执行->报错->自修正”尝试
        for attempt in range(1, self.args.max_step_retries + 1):
            exec_result = self.sql_tool.run(current_sql)
            last_rows_sample = exec_result.get("rows_sample", [])
            attempts.append(
                {
                    "attempt": attempt,
                    "sql": current_sql,
                    "status": exec_result["status"],
                    "error": exec_result.get("error"),
                    "rows_sample": exec_result.get("rows_sample"),
                }
            )
            dialogues.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "step_desc": step.desc,
                            "attempt": attempt,
                            **exec_result,
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                }
            )

            if exec_result["status"] == "success":
                success = True
                break

            last_error = exec_result.get("error")
            if attempt >= self.args.max_step_retries:
                break
            # 自修正阶段：不依赖最终标准答案，只根据执行错误修正当前子问题 SQL
            current_sql = self._revise_sql(step, current_sql, last_error, dialogues, enriched_context)

        status = "success" if success else "error"
        return StepSQLResult(step, sql=current_sql, status=status, error=last_error, rows_sample=last_rows_sample, attempts=attempts)

    def _generate_sql(
        self,
        step: StepPlan,
        context: Dict[str, str],
        final_sql: str,
        is_final_step: bool,
        dialogues: List[Dict[str, Any]],
    ) -> str:
        if self.llm_client is None:
            return final_sql if is_final_step else "SELECT DISTINCT vplayerid FROM dws_jordass_matchlog_stat_di LIMIT 50"

        prompt = STEP_SQL_ONE_PROMPT.format(
            step_desc=step.desc,
            user_question=context.get("user_question", ""),
            schema_text=context.get("schema_text", ""),
            ref_text=context.get("ref_text", ""),
            is_final_step=str(is_final_step).lower(),
        )
        # 记录完整的 prompt 方便离线分析
        dialogues.append({
            "role": "user",
            "content": json.dumps(
                {
                    "prompt_type": "STEP_SQL_ONE_PROMPT",
                    "is_final_step": is_final_step,
                    "raw_prompt": prompt,
                },
                ensure_ascii=False,
            ),
        })
        response = self._chat_completion(prompt, self.args.sqlgen_temperature, 4096) if self.llm_client else ""
        parsed = safe_json_loads(response) or {}
        sql = parsed.get("sql")

        if not sql:
            sql = final_sql if is_final_step else "SELECT DISTINCT vplayerid FROM dws_jordass_matchlog_stat_di WHERE dtstatdate >= '20240101' LIMIT 100"
        elif not is_final_step:
            norm_final = re.sub(r"\s+", " ", final_sql.strip()).lower()
            norm_sql = re.sub(r"\s+", " ", sql.strip()).lower()
            if norm_sql == norm_final or "count(" in norm_sql or "group by" in norm_sql:
                scope_prompt = SCOPE_REVISION_PROMPT.format(step_desc=step.desc, current_sql=sql)
                dialogues.append({
                    "role": "user",
                    "content": json.dumps(
                        {
                            "prompt_type": "SCOPE_REVISION_PROMPT",
                            "raw_prompt": scope_prompt,
                        },
                        ensure_ascii=False,
                    ),
                })
                correction = self._chat_completion(scope_prompt, self.args.sqlgen_temperature, 8192) if self.llm_client else ""
                dialogues.append({"role": "assistant", "content": correction})
                sql = (safe_json_loads(correction) or {}).get("sql", sql)

        dialogues.append({
            "role": "assistant",
            "content": json.dumps({"sql": sql}, ensure_ascii=False),
        })
        return sql

    def _revise_sql(
        self,
        step: StepPlan,
        failed_sql: str,
        error_msg: Optional[str],
        dialogues: List[Dict[str, Any]],
        context: Dict[str, str],
    ) -> str:
        if self.llm_client is None:
            return failed_sql
        prompt = REVISION_PROMPT.format(
            step_desc=step.desc,
            user_question=context.get("user_question", ""),
            schema_text=context.get("schema_text", ""),
            ref_text=context.get("ref_text", ""),
            failed_sql=failed_sql,
            error_msg=error_msg or "Execution error",
        )
        # 记录完整的 SQL 自修正 prompt
        dialogues.append({
            "role": "user",
            "content": json.dumps(
                {
                    "prompt_type": "REVISION_PROMPT",
                    "error_msg": error_msg,
                    "raw_prompt": prompt,
                },
                ensure_ascii=False,
            ),
        })
        response = self._chat_completion(prompt, self.args.revise_temperature, 8192)
        revised = (safe_json_loads(response) or {}).get("revised_sql", failed_sql)
        dialogues.append({"role": "assistant", "content": json.dumps({"revised_sql": revised}, ensure_ascii=False)})
        self._log_correction(step, failed_sql, revised, error_msg)
        return revised

    def _log_correction(self, step: StepPlan, prev_sql: str, revised_sql: str, error: Optional[str]):
        path = getattr(self.args, "corrections_file", "")
        if not path:
            return
        payload = {
            "step_desc": step.desc,
            "prev_sql": prev_sql,
            "revised_sql": revised_sql,
            "error": error,
            "timestamp": int(time.time()),
            "model": self.args.model_used,
            "endpoint_type": self.args.endpoint_type_resolved,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fw:
            fw.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Streaming writer
# ---------------------------------------------------------------------------

class StreamJsonArrayWriter:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._fh = None
        self._first = True

    def __enter__(self):
        Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists and has content
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            # Remove the last ']' if present to append
            with open(self.file_path, "rb+") as f:
                f.seek(0, os.SEEK_END)
                # Scan backwards for ']'
                pos = f.tell()
                found = False
                # Scan last 128 bytes
                scan_len = min(pos, 128)
                f.seek(-scan_len, os.SEEK_END)
                tail = f.read(scan_len)
                
                # Find last ']'
                last_bracket = tail.rfind(b']')
                if last_bracket != -1:
                    # Calculate absolute position
                    truncate_pos = pos - scan_len + last_bracket
                    f.seek(truncate_pos)
                    f.truncate()
                    found = True
                    
                    # Check if the file is now effectively empty (just '[')
                    # This is a simple heuristic
                    if truncate_pos <= 5: # e.g. "[\n"
                        f.seek(0)
                        content = f.read().strip()
                        if content == b'[':
                            self._first = True
                        else:
                            self._first = False
                    else:
                        self._first = False
                else:
                    # No closing bracket found, assume appending to list
                    self._first = False
            
            self._fh = open(self.file_path, "a", encoding="utf-8")
        else:
            self._fh = open(self.file_path, "w", encoding="utf-8")
            self._fh.write("[\n")
            self._first = True
        
        return self

    def write(self, obj: Dict[str, Any]):
        if not self._fh:
            raise RuntimeError("writer not opened")
        if not self._first:
            self._fh.write(",\n")
        else:
            self._first = False
        self._fh.write(json.dumps(obj, ensure_ascii=False, indent=2))
        self._fh.flush()

    def __exit__(self, exc_type, exc, tb):
        if self._fh:
            self._fh.write("\n]\n")
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
# Augmentation agent orchestrator
# ---------------------------------------------------------------------------

class AugmentationAgent:
    def __init__(self, args):
        self.args = args
        llm_client = None if args.offline else LLMClient(args.llm_model, args.decompose_temperature)
        self.planner = StepPlanner(llm_client)
        self.executor = StepExecutor(llm_client, SQLExecutionTool(), args)

        if args.offline:
            self.reflection_client = None
        else:
            ref_model = args.reflection_model if args.reflection_model else args.llm_model
            if ref_model == args.llm_model:
                self.reflection_client = llm_client
            else:
                self.reflection_client = LLMClient(ref_model, default_temperature=0.1)

    def process_dataset(self, records: List[Dict[str, Any]]):
        total = 0
        with StreamJsonArrayWriter(self.args.output_file) as writer:
            for idx, record in enumerate(records):
                variants = self._process_record(record)
                for variant in variants:
                    writer.write(variant)
                    total += 1
                print(f"[augment] processed sample {idx + 1}/{len(records)} with {len(variants)} variants")
        print(f"[augment] total variants: {total}")

    def _process_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        instruction = record.get("instruction") or record.get("question") or ""
        context_raw = parse_cn_prompt_blocks(record.get("input", ""))
        if not context_raw.get("user_question"):
            context_raw["user_question"] = instruction
        final_sql = record.get("sql") or record.get("output") or ""
        if not final_sql or not context_raw.get("user_question"):
            return []

        variants: List[Dict[str, Any]] = []
        for vidx in range(self.args.variants_per_question):
            n_steps = random.randint(self.args.min_steps, self.args.max_steps)
            steps = self.planner.plan(context_raw, n_steps, final_sql, self.args.offline)
            dialogues: List[Dict[str, Any]] = [
                {
                    "role": "note",
                    "content": f"endpoint={self.args.endpoint_type_resolved}, model={self.args.model_used}, base_url={self.args.openai_base_url}",
                }
            ]

            self.executor.reset_history()

            validated_sqls: List[str] = []
            step_results: List[StepSQLResult] = []
            print("step", steps)
            for idx_step, step in enumerate(steps):
                res = self.executor.run_step(
                    step,
                    context_raw,
                    final_sql,
                    is_final_step=self._is_last_non_meta(steps, idx_step),
                    validated_sqls=validated_sqls,
                    dialogues=dialogues,
                    variant_id=f"{record.get('question_id', 'unknown')}-v{vidx}",
                )
                step_results.append(res)
                validated_sqls.append(res.sql)

            # Reflection phase: use ground truth to refine steps
            # 每一轮基于上一轮的 step_results，形成串联的反思改写链
            for r_idx in range(self.args.reflection_rounds):
                step_results = self._reflect_and_refine(context_raw, steps, step_results, final_sql, dialogues, r_idx)
            validated_sqls = [r.sql for r in step_results]

            dialogues.append({"role": "user", "content": "请基于已验证SQL生成最终答案"})
            final_sql_generated = self._synthesize_final(context_raw, steps, validated_sqls, final_sql)
            dialogues.append({"role": "assistant", "content": json.dumps({"final_sql": final_sql_generated}, ensure_ascii=False)})

            variant = {
                "original_id": record.get("question_id"),
                "variant_id": f"{record.get('question_id', 'unknown')}-v{vidx}",
                "question": context_raw.get("user_question", ""),
                "schema": context_raw.get("schema_text", ""),
                "reference": context_raw.get("ref_text", ""),
                "final_sql": final_sql,
                "step_plan": [step.__dict__ for step in steps],
                "step_sqls": [res.sql for res in step_results],
                "step_exec_status": [res.status for res in step_results],
                "step_exec_errors": [res.error for res in step_results],
                "step_rows_sample": [res.rows_sample for res in step_results],
                "dialogues": dialogues,
                "verification": {
                    "all_steps_success": all(r.status in {"success", "meta"} for r in step_results),
                    "failed_steps": [idx for idx, r in enumerate(step_results) if r.status == "error"],
                    # 标记哪些步骤之间的SQL在归一化后是重复的，方便后处理过滤或重跑
                    "duplicate_step_sql_pairs": [
                        [i, j]
                        for i in range(len(step_results))
                        for j in range(i + 1, len(step_results))
                        if self._normalize_sql(step_results[i].sql) == self._normalize_sql(step_results[j].sql)
                    ],
                },
                "n_steps": n_steps,
                "final_sql_generated": final_sql_generated,
                "final_sql_match": self._normalize_sql(final_sql_generated) == self._normalize_sql(final_sql),
                "llm_model": self.args.model_used,
                "endpoint_type": self.args.endpoint_type_resolved,
                "openai_base_url": self.args.openai_base_url,
            }
            variants.append(variant)
        return variants

    def _reflect_and_refine(
        self,
        context: Dict[str, str],
        steps: List[StepPlan],
        step_results: List[StepSQLResult],
        final_sql: str,
        dialogues: List[Dict[str, Any]],
        round_idx: int = 0
    ) -> List[StepSQLResult]:
        if self.args.offline or self.reflection_client is None:
            return step_results

        print(f"[augment] starting reflection phase round {round_idx + 1} (step-by-step)...")
        new_results = []
        # 预先构造所有步骤及SQL，作为反思提示的一部分，帮助模型减少不同step之间的完全重复
        all_steps_bundle = json.dumps(
            {
                "steps": [s.desc for s in steps],
                "step_sqls": [r.sql for r in step_results],
            },
            ensure_ascii=False,
        )
        
        for idx, (step, res) in enumerate(zip(steps, step_results)):
            if step.meta:
                new_results.append(res)
                continue
            
            reflection_history: List[Dict[str, str]] = [
                {"role": "system", "content": REFLECTION_SYSTEM_PROMPT}
            ]

            # Prompt for this step（带上标准答案 SQL + 所有步骤SQL，仅用于评估/修正当前子 SQL，不直接替换）
            prompt = REFLECTION_STEP_PROMPT.format(
                user_question=context.get("user_question", ""),
                schema_text=context.get("schema_text", ""),
                final_sql=final_sql,
                all_steps_and_sqls=all_steps_bundle,
                step_idx=idx,
                step_desc=step.desc,
                current_sql=res.sql,
            )

            dialogues.append({
                "role": "user",
                "content": json.dumps(
                    {
                        "prompt_type": "REFLECTION_STEP_PROMPT",
                        "round": round_idx + 1,
                        "step_index": idx,
                        "raw_prompt": prompt,
                    },
                    ensure_ascii=False,
                ),
            })

            try:
                reflection_history.append({"role": "user", "content": prompt})
                response = self.reflection_client.invoke(
                    temperature=0.1,
                    max_tokens=2048,
                    messages=[msg.copy() for msg in reflection_history],
                    prompt=prompt,
                )
                reflection_history.append({"role": "assistant", "content": response})
                dialogues.append({"role": "assistant", "content": response})
                
                data = safe_json_loads(response)
                if data and not data.get("is_correct") and "revised_sql" in data:
                    # 在单轮反思中，允许多次修正+执行尝试，受 max_step_retries 控制
                    max_try = max(1, getattr(self.args, "max_step_retries", 2))
                    base_attempts = res.attempts.copy()
                    current_sql = res.sql
                    last_error = res.error
                    exec_result = {"status": res.status, "error": res.error, "rows_sample": res.rows_sample}

                    for local_try in range(1, max_try + 1):
                        revised_sql = data.get("revised_sql", "").strip() or current_sql
                        # 若修正后与当前 SQL 完全一致，则不再进入无意义循环
                        if not revised_sql or self._normalize_sql(revised_sql) == self._normalize_sql(current_sql):
                            break

                        print(f"[augment] Step {idx} revised by reflection (round {round_idx+1}, try {local_try}).")
                        current_sql = revised_sql
                        exec_result = self.executor.sql_tool.run(current_sql)

                        base_attempts.append({
                            "attempt": f"reflection-r{round_idx+1}-t{local_try}",
                            "sql": current_sql,
                            "status": exec_result["status"],
                            "error": exec_result.get("error"),
                            "rows_sample": exec_result.get("rows_sample"),
                        })

                        if exec_result["status"] == "success":
                            last_error = None
                            break

                        last_error = exec_result.get("error")
                        if local_try >= max_try:
                            break

                        # 基于最新错误再请求一次反思模型，保持在同一轮对话语境下连续修正
                        follow_prompt = REFLECTION_STEP_PROMPT.format(
                            user_question=context.get("user_question", ""),
                            schema_text=context.get("schema_text", ""),
                            final_sql=final_sql,
                            all_steps_and_sqls=all_steps_bundle,
                            step_idx=idx,
                            step_desc=step.desc,
                            current_sql=current_sql,
                        )
                        dialogues.append({
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "prompt_type": "REFLECTION_STEP_PROMPT_CONTINUE",
                                    "round": round_idx + 1,
                                    "step_index": idx,
                                    "try": local_try + 1,
                                    "raw_prompt": follow_prompt,
                                    "last_error": last_error,
                                },
                                ensure_ascii=False,
                            ),
                        })
                        reflection_history.append({"role": "user", "content": follow_prompt})
                        follow_resp = self.reflection_client.invoke(
                            temperature=0.1,
                            max_tokens=2048,
                            messages=[msg.copy() for msg in reflection_history],
                            prompt=follow_prompt,
                        )
                        reflection_history.append({"role": "assistant", "content": follow_resp})
                        dialogues.append({"role": "assistant", "content": follow_resp})
                        data = safe_json_loads(follow_resp) or {}

                    new_res = StepSQLResult(
                        plan=res.plan,
                        sql=current_sql,
                        status=exec_result["status"],
                        error=last_error,
                        rows_sample=exec_result.get("rows_sample", []),
                        attempts=base_attempts,
                    )
                    new_results.append(new_res)
                else:
                    new_results.append(res)

            except Exception as e:
                print(f"[augment] Reflection error at step {idx}: {e}")
                new_results.append(res)
                
        return new_results

    @staticmethod
    def _is_last_non_meta(steps: List[StepPlan], idx: int) -> bool:
        non_meta_indices = [i for i, step in enumerate(steps) if not step.meta]
        return bool(non_meta_indices) and idx == non_meta_indices[-1]

    def _synthesize_final(self, context: Dict[str, str], steps: List[StepPlan], validated_sqls: List[str], fallback_final: str) -> str:
        if self.args.offline or self.executor.llm_client is None:
            return fallback_final
        bundle = json.dumps(
            {
                "steps": [step.__dict__ for step in steps],
                "step_sqls": validated_sqls,
            },
            ensure_ascii=False,
        )
        prompt = FINAL_SYNTHESIS_PROMPT.format(
            user_question=context.get("user_question", ""),
            schema_text=context.get("schema_text", ""),
            ref_text=context.get("ref_text", ""),
            steps_and_sqls=bundle,
        )
        response = self.executor.llm_client.invoke(prompt, temperature=self.args.sqlgen_temperature, max_tokens=1024)
        return (safe_json_loads(response) or {}).get("final_sql", fallback_final)

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        return re.sub(r"\s+", " ", (sql or "").strip()).lower()


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def load_input_dataset(path: str, limit: int) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        records = [{"question_id": int(k), "question": "", "sql": v} for k, v in data.items()]
    else:
        records = data
    if limit != -1:
        records = records[:limit]
    return records


def resolve_endpoint(args) -> None:
    base_url = args.openai_base_url or os.getenv("OPENAI_BASE_URL", "")
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    parsed = urlparse(base_url) if base_url else None
    inferred = "unknown"
    if parsed and parsed.hostname:
        inferred = "vllm" if parsed.hostname in {"127.0.0.1", "localhost"} else "online"
    if args.offline:
        inferred = "offline"

    args.endpoint_type_resolved = args.endpoint_type if args.endpoint_type != "auto" else inferred
    args.model_used = "offline-rules" if args.offline else args.llm_model
    args.openai_base_url = base_url
    args.openai_api_key = api_key

    print(f"[augment] endpoint_type={args.endpoint_type_resolved}, model={args.model_used}, base_url={base_url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--min_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--variants_per_question", type=int, default=1)
    parser.add_argument("--max_step_retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--llm_model", default="Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--decompose_temperature", type=float, default=0.2)
    parser.add_argument("--sqlgen_temperature", type=float, default=0.3)
    parser.add_argument("--revise_temperature", type=float, default=0.1)
    parser.add_argument("--reflection_rounds", type=int, default=1)
    parser.add_argument("--reflection_model", default="")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--endpoint_type", choices=["auto", "online", "vllm", "offline"], default="auto")
    parser.add_argument("--corrections_file", default="")
    parser.add_argument("--openai_base_url", default="")
    parser.add_argument("--openai_api_key", default="")
    args = parser.parse_args()

    resolve_endpoint(args)
    records = load_input_dataset(args.input_file, args.limit)
    agent = AugmentationAgent(args)
    agent.process_dataset(records)
    DEFAULT_COST_RECORDER.print_profile()


if __name__ == "__main__":
    main()
