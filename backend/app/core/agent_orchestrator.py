from app.agents.factories.agent_factory import AgentFactory
from app.core.agent_context import AgentContext
from app.core.metrics_tracker import metrics_tracker
import logging
import uuid

logger = logging.getLogger(__name__)


class AgentOrchetrator:
    def __init__(self, context: AgentContext):
        self.context = context
        self.factory = AgentFactory(context)

    async def run(self, user_message: str, capture_trace: bool = False):
        run_id = str(uuid.uuid4())
        session_id = getattr(self.context, "session_id", "unknown-session")
        metrics_tracker.start_run(run_id=run_id, session_id=session_id, user_message=user_message)
        try:
            logger.info(f"Starting agent orchestrator run for user message: {user_message}", extra={"run_id": run_id, "user_message": user_message})

            logger.debug(f"Invoking DeciderAgent with message: {user_message}", extra={"run_id": run_id})
            decider = self.factory.create("decider_agent")
            plan = await decider.run(user_message, run_id=run_id)
            logger.debug(f"DeciderAgent returned plan: {plan}", extra={"run_id": run_id, "plan": plan})
            metrics_tracker.record_route(
                run_id=run_id,
                intent=plan.get("intent", "unknown"),
                agents=plan.get("agents") or ["conversation_agent"],
            )

            state = {"user_message": user_message, "run_id": run_id, "intent": plan.get("intent", "unknown")}
            agent_sequence = plan.get("agents") or ["conversation_agent"]
            trace = {
                "intent": plan.get("intent", "unknown"),
                "agents_planned": list(agent_sequence),
                "agent_outputs": {},
            }

            for agent_name in agent_sequence:
                logger.debug(f"Invoking agent: {agent_name} with state: {state}", extra={"run_id": run_id, "agent_name": agent_name, "state_before": state})
                try:
                    agent = self.factory.create(agent_name)
                except ValueError:
                    logger.warning(
                        "Unknown agent from decider; skipping",
                        extra={"run_id": run_id, "agent_name": agent_name},
                    )
                    continue

                metrics_tracker.record_agent_execution(run_id=run_id, agent_name=agent_name)
                result = await agent.run(**state)
                state.update(result)
                trace["agent_outputs"][agent_name] = result
                logger.debug(f"Agent {agent_name} returned result: {result}", extra={"run_id": run_id, "agent_name": agent_name, "result": result, "state_after": state})

            is_conversation_only = (
                agent_sequence == ["conversation_agent"]
                and state.get("intent") in {"conversation", "small_talk", "chitchat", "fallback", "unknown"}
            )

            if is_conversation_only and state.get("llm_output"):
                final_output = state["llm_output"]
            else:
                logger.debug(f"Invoking ReasoningAgent with final state: {state}", extra={"run_id": run_id, "state_before_reasoning": state})
                reasoner = self.factory.create("reasoning_agent")
                final = await reasoner.run(**state)
                logger.debug(f"ReasoningAgent returned final output: {final}", extra={"run_id": run_id, "final_output_reasoning": final})
                final_output = final["final_output"]
                trace["agent_outputs"]["reasoning_agent"] = final

            try:
                if not hasattr(self.context, "long_memory") or not hasattr(self.context, "short_memory"):
                    raise AttributeError("Context memory objects are unavailable")

                existing_history = await self.context.long_memory.get(session_id)
                last_entry = existing_history[-1] if existing_history else None
                already_saved = (
                    isinstance(last_entry, dict)
                    and last_entry.get("user_message") == user_message
                    and last_entry.get("agent_output") == final_output
                )

                if not already_saved:
                    await self.context.short_memory.save(
                        session_id=session_id,
                        user_message=user_message,
                        agent_output=final_output,
                    )
                    await self.context.long_memory.save(
                        session_id=session_id,
                        user_message=user_message,
                        agent_output=final_output,
                    )
            except Exception:
                logger.debug("Failed to save final response in memory", extra={"run_id": run_id}, exc_info=True)

            metrics_tracker.end_run(run_id=run_id, final_output=final_output)
            response = {"final_output": final_output, "run_id": run_id}
            if capture_trace:
                response["trace"] = trace
            return response
        except Exception as exc:
            metrics_tracker.end_run(run_id=run_id, error=str(exc))
            raise


AgentOrchestrator = AgentOrchetrator
