"""The agents that make up the research team."""

from research_assistant.agents.critic import make_critic
from research_assistant.agents.planner import make_planner
from research_assistant.agents.researcher import make_researcher
from research_assistant.agents.supervisor import make_supervisor, route_after_supervisor
from research_assistant.agents.writer import make_writer

__all__ = [
    "make_critic",
    "make_planner",
    "make_researcher",
    "make_supervisor",
    "make_writer",
    "route_after_supervisor",
]
