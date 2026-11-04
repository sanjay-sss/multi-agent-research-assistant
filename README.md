# Multi-Agent Research Assistant

A multi-agent system built with [LangGraph](https://github.com/langchain-ai/langgraph) and Google Gemini that turns a research topic into a cited report.

A **supervisor** agent coordinates a team of specialists:

| Agent | Role |
|-------|------|
| Supervisor | Decides which agent works next |
| Planner | Breaks the topic into focused research questions |
| Researcher | Searches the web and gathers sources |
| Writer | Drafts a structured Markdown report with citations |
| Critic | Reviews the draft and requests revisions until it meets the bar |

> Work in progress: setup and usage instructions are coming soon.
