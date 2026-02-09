from app.core.metrics_tracker import MetricsTracker


def test_metrics_tracker_summary_and_rates():
    tracker = MetricsTracker()
    run_id = "run-1"
    tracker.start_run(run_id=run_id, session_id="s1", user_message="I have fever and cough")
    tracker.record_route(
        run_id=run_id,
        intent="symptom_analysis",
        agents=["symptom_matcher_agent", "disease_info_agent"],
    )
    tracker.record_agent_execution(run_id=run_id, agent_name="symptom_matcher_agent")
    tracker.record_tool_event(run_id=run_id, tool_name="symptom", source="vector_db", success=True)
    tracker.record_tool_event(run_id=run_id, tool_name="disease", source="local_db", success=False)
    tracker.record_tool_event(run_id=run_id, tool_name="google", source="internet", success=True)
    tracker.record_memory_recall(run_id=run_id, items_used=2)
    tracker.record_memory_save(run_id=run_id)
    tracker.end_run(run_id=run_id, final_output="Your fever and cough may be viral.")

    summary = tracker.summary()
    assert summary["total_queries"] == 1
    assert summary["local_data_usage_rate"] == 1.0
    assert summary["internet_usage_rate"] == 1.0
    assert summary["web_fallback_rate"] == 1.0
    assert summary["medical_query_rate"] == 1.0
    assert summary["memory_context_usage_rate"] == 1.0
    assert "symptom_analysis" in summary["route_distribution"]


def test_metrics_tracker_recent_runs_and_reset():
    tracker = MetricsTracker()
    tracker.start_run(run_id="run-2", session_id="s2", user_message="hello")
    tracker.record_route(run_id="run-2", intent="conversation", agents=["conversation_agent"])
    tracker.end_run(run_id="run-2", final_output="Hello! How can I help?")

    runs = tracker.get_recent_runs(limit=5)
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run-2"
    assert runs[0]["status"] == "ok"
    assert isinstance(runs[0]["relevance_score"], float)

    tracker.reset()
    assert tracker.summary()["total_queries"] == 0


def test_metrics_tracker_save_to_local(tmp_path):
    tracker = MetricsTracker()
    tracker.start_run(run_id="run-save", session_id="s-save", user_message="hello")
    tracker.record_route(run_id="run-save", intent="conversation", agents=["conversation_agent"])
    tracker.end_run(run_id="run-save", final_output="Hi there")

    out_file = tmp_path / "metrics-export.json"
    saved_path = tracker.save_to_local(filepath=str(out_file), limit=50)

    assert saved_path == str(out_file)
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "\"summary\"" in content
    assert "\"runs\"" in content
