import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class MetricsTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._runs: List[Dict[str, Any]] = []
        self._runs_by_id: Dict[str, Dict[str, Any]] = {}
        self._stop_words = {
            "the", "is", "are", "a", "an", "and", "or", "to", "of", "for", "in", "on",
            "with", "i", "you", "my", "me", "your", "what", "how", "can", "please",
        }

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _timestamp_name(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    def _empty_run(self, run_id: str, session_id: str, user_message: str) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "session_id": session_id,
            "user_message": user_message,
            "started_at": self._now_iso(),
            "finished_at": None,
            "latency_ms": None,
            "status": "running",
            "error": None,
            "intent": "unknown",
            "agents_planned": [],
            "agents_executed": [],
            "local_db_calls": 0,
            "local_db_successes": 0,
            "vector_db_calls": 0,
            "vector_db_successes": 0,
            "internet_search_calls": 0,
            "internet_search_successes": 0,
            "tool_errors": 0,
            "memory_recall_calls": 0,
            "memory_items_used": 0,
            "memory_save_calls": 0,
            "final_output": "",
            "final_output_chars": 0,
            "relevance_score": None,
            "_start_perf": time.perf_counter(),
        }

    def start_run(self, run_id: str, session_id: str, user_message: str) -> None:
        with self._lock:
            run = self._empty_run(run_id=run_id, session_id=session_id, user_message=user_message)
            self._runs.append(run)
            self._runs_by_id[run_id] = run

    def record_route(self, run_id: str, intent: str, agents: List[str]) -> None:
        with self._lock:
            run = self._runs_by_id.get(run_id)
            if not run:
                return
            run["intent"] = intent or "unknown"
            run["agents_planned"] = list(agents or [])

    def record_agent_execution(self, run_id: str, agent_name: str) -> None:
        with self._lock:
            run = self._runs_by_id.get(run_id)
            if not run:
                return
            run["agents_executed"].append(agent_name)

    def record_memory_recall(self, run_id: str, items_used: int) -> None:
        with self._lock:
            run = self._runs_by_id.get(run_id)
            if not run:
                return
            run["memory_recall_calls"] += 1
            run["memory_items_used"] += max(0, int(items_used))

    def record_memory_save(self, run_id: str) -> None:
        with self._lock:
            run = self._runs_by_id.get(run_id)
            if not run:
                return
            run["memory_save_calls"] += 1

    def record_tool_event(
        self, run_id: Optional[str], tool_name: str, source: str, success: bool, error: bool = False
    ) -> None:
        if not run_id:
            return
        with self._lock:
            run = self._runs_by_id.get(run_id)
            if not run:
                return

            if source == "local_db":
                run["local_db_calls"] += 1
                if success:
                    run["local_db_successes"] += 1
            elif source == "vector_db":
                run["vector_db_calls"] += 1
                if success:
                    run["vector_db_successes"] += 1
            elif source == "internet":
                run["internet_search_calls"] += 1
                if success:
                    run["internet_search_successes"] += 1

            if error or not success:
                run["tool_errors"] += 1

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        return [t for t in tokens if len(t) > 2 and t not in self._stop_words]

    def _compute_relevance(self, user_message: str, final_output: str) -> float:
        if not user_message or not final_output:
            return 0.0
        query_tokens = set(self._tokenize(user_message))
        answer_tokens = set(self._tokenize(final_output))
        if not query_tokens or not answer_tokens:
            return 0.0

        overlap = len(query_tokens.intersection(answer_tokens))
        coverage = overlap / len(query_tokens)
        return round(max(0.0, min(1.0, coverage)), 4)

    def end_run(self, run_id: str, final_output: str = "", error: Optional[str] = None) -> None:
        with self._lock:
            run = self._runs_by_id.get(run_id)
            if not run:
                return

            run["finished_at"] = self._now_iso()
            run["latency_ms"] = round((time.perf_counter() - run["_start_perf"]) * 1000.0, 2)
            run["status"] = "error" if error else "ok"
            run["error"] = error
            run["final_output"] = final_output or ""
            run["final_output_chars"] = len(run["final_output"])

            try:
                run["relevance_score"] = self._compute_relevance(
                    user_message=run["user_message"], final_output=run["final_output"]
                )
            except Exception:
                run["relevance_score"] = None

            run.pop("_start_perf", None)

    def get_recent_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            limit = max(1, min(500, int(limit)))
            runs = list(reversed(self._runs[-limit:]))
            return [dict(run) for run in runs]

    def snapshot(self, limit: int = 500) -> Dict[str, Any]:
        return {
            "generated_at": self._now_iso(),
            "summary": self.summary(),
            "runs": self.get_recent_runs(limit=limit),
        }

    def save_to_local(self, filepath: Optional[str] = None, limit: int = 500) -> str:
        if filepath:
            target_path = filepath
        else:
            os.makedirs("metrics_store", exist_ok=True)
            target_path = os.path.join("metrics_store", f"metrics-{self._timestamp_name()}.json")

        payload = self.snapshot(limit=limit)
        with open(target_path, "w", encoding="utf-8") as f:
            import json

            json.dump(payload, f, ensure_ascii=True, indent=2)
        return target_path

    def reset(self) -> None:
        with self._lock:
            self._runs = []
            self._runs_by_id = {}

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            completed = [r for r in self._runs if r.get("status") in {"ok", "error"}]
            total = len(completed)
            if total == 0:
                return {
                    "total_queries": 0,
                    "local_data_usage_rate": 0.0,
                    "internet_usage_rate": 0.0,
                    "web_fallback_rate": 0.0,
                    "local_hit_success_rate": 0.0,
                    "avg_relevance_score": 0.0,
                    "low_relevance_rate": 0.0,
                    "avg_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "memory_context_usage_rate": 0.0,
                    "avg_memory_items_used": 0.0,
                    "medical_query_rate": 0.0,
                    "conversation_query_rate": 0.0,
                    "tool_error_rate": 0.0,
                    "route_distribution": {},
                }

            latencies = sorted([r["latency_ms"] for r in completed if r.get("latency_ms") is not None])
            p95_idx = min(len(latencies) - 1, max(0, int(math.ceil(0.95 * len(latencies))) - 1))
            p95_latency = latencies[p95_idx] if latencies else 0.0

            used_local = [r for r in completed if (r["local_db_calls"] + r["vector_db_calls"]) > 0]
            used_internet = [r for r in completed if r["internet_search_calls"] > 0]
            fallback_runs = [
                r
                for r in completed
                if (r["local_db_calls"] > 0 and r["local_db_successes"] == 0 and r["internet_search_calls"] > 0)
            ]

            local_calls = sum(r["local_db_calls"] for r in completed)
            local_successes = sum(r["local_db_successes"] for r in completed)
            tool_calls = (
                sum(r["local_db_calls"] + r["vector_db_calls"] + r["internet_search_calls"] for r in completed) or 1
            )
            tool_errors = sum(r["tool_errors"] for r in completed)

            memory_context_used = [r for r in completed if r["memory_items_used"] > 0]
            relevance_values = [r["relevance_score"] for r in completed if isinstance(r.get("relevance_score"), float)]
            low_relevance = [v for v in relevance_values if v < 0.45]

            route_distribution: Dict[str, int] = {}
            for r in completed:
                key = r.get("intent", "unknown") or "unknown"
                route_distribution[key] = route_distribution.get(key, 0) + 1

            medical_runs = [
                r
                for r in completed
                if any(a in {"symptom_matcher_agent", "disease_info_agent"} for a in r["agents_planned"])
            ]
            conversation_runs = [
                r for r in completed if r.get("agents_planned") == ["conversation_agent"]
            ]

            return {
                "total_queries": total,
                "local_data_usage_rate": round(len(used_local) / total, 4),
                "internet_usage_rate": round(len(used_internet) / total, 4),
                "web_fallback_rate": round(len(fallback_runs) / total, 4),
                "local_hit_success_rate": round((local_successes / local_calls), 4) if local_calls else 0.0,
                "avg_relevance_score": round(sum(relevance_values) / len(relevance_values), 4) if relevance_values else 0.0,
                "low_relevance_rate": round(len(low_relevance) / len(relevance_values), 4) if relevance_values else 0.0,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                "p95_latency_ms": round(p95_latency, 2),
                "memory_context_usage_rate": round(len(memory_context_used) / total, 4),
                "avg_memory_items_used": round(
                    sum(r["memory_items_used"] for r in completed) / total, 2
                ),
                "medical_query_rate": round(len(medical_runs) / total, 4),
                "conversation_query_rate": round(len(conversation_runs) / total, 4),
                "tool_error_rate": round(tool_errors / tool_calls, 4),
                "route_distribution": route_distribution,
            }


metrics_tracker = MetricsTracker()
