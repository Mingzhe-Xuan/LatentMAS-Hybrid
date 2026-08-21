"""Run the standard experiment with a two-agent Planner -> Judger topology.

All argument parsing, dataset loading, evaluation, metrics, and result formats are
provided by :mod:`run`. This entry point changes only the agents used by the MAS
methods: Planner produces the intermediate reasoning and Judger produces the
visible final answer. Baseline remains a single-agent baseline by definition.
"""

from typing import Dict, List

from methods import Agent


def simple_agents() -> List[Agent]:
    """Return the fixed two-agent topology used by the simple experiment."""
    return [
        Agent(name="Planner", role="planner"),
        Agent(name="Judger", role="judger"),
    ]


def _install_simple_topology() -> None:
    """Override the factories imported by each MAS implementation."""
    import methods.latent_mas as latent_mas
    import methods.latent_mas_hybrid as latent_mas_hybrid
    import methods.text_mas as text_mas

    latent_mas.default_agents = simple_agents
    latent_mas_hybrid.default_agents = simple_agents
    text_mas.default_agents = simple_agents

    # The standard sequential TextMAS judger prompt describes the four-agent
    # chain. Keep the task-specific answer instructions unchanged while making
    # its topology and context labels truthful for this two-agent entry point.
    original_builder = text_mas.build_agent_messages_sequential_text_mas

    def build_simple_sequential_messages(
        role: str,
        question: str,
        context: str = "",
        method=None,
        args=None,
    ) -> List[Dict[str, str]]:
        messages = original_builder(role, question, context, method, args)
        if role == "judger":
            replacements = {
                "planner -> critic -> refiner -> solver": "planner -> solver",
                "Refiner Agent's plan": "Planner Agent's plan",
                "Refined Plan from Previous Agents:": "Plan from Planner Agent:",
            }
            for message in messages:
                content = message.get("content", "")
                for old, new in replacements.items():
                    content = content.replace(old, new)
                message["content"] = content
        return messages

    text_mas.build_agent_messages_sequential_text_mas = build_simple_sequential_messages


def main() -> None:
    _install_simple_topology()

    # Importing run here keeps its normal CLI untouched and ensures that every
    # evaluation/metric path stays exactly aligned with the standard entry point.
    import run

    run.main()


if __name__ == "__main__":
    main()
