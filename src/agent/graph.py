"""LangGraph single-node interview graph.

- Uses `interrupt(...)` to truly pause for candidate input.
- No custom checkpointer (compatible with `langgraph dev`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from langgraph.graph import StateGraph
from langgraph.types import interrupt

# --------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------
console = Console()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEFAULT_MODEL = os.environ.get("INTERVIEW_MODEL", "gpt-4.1-mini")


# --------------------------------------------------------------------
# LLM wrapper
# --------------------------------------------------------------------
class LLM:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.client = OpenAI()
        self.model = model

    def complete(
        self,
        system: str,
        user: str,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str | Dict[str, Any]:
        if json_schema:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_schema", "json_schema": json_schema},
                temperature=0.3,
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        else:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()


# --------------------------------------------------------------------
# Data Models
# --------------------------------------------------------------------
class Topic(BaseModel):
    title: str
    question: str
    expected_answer_bullets: List[str] = Field(
        description="3-6 bullets describing the gold-standard answer"
    )


class InterviewPlan(BaseModel):
    role_summary: str
    topics: List[Topic]


class Turn(BaseModel):
    role: str  # "agent" | "candidate"
    content: str
    topic_index: Optional[int] = None
    rating: Optional[float] = None
    completeness: Optional[str] = None  # "complete" | "partial" | "missing"


class Transcript(BaseModel):
    interviewee_name: Optional[str] = None
    role_title: str
    jd_excerpt: str
    created_at: str
    plan: InterviewPlan
    # use default_factory to avoid mutable default list bug
    turns: List[Turn] = Field(default_factory=list)


class Assessment(BaseModel):
    overall_score: float
    strengths: List[str]
    improvements: List[str]
    topic_scores: List[float]


# --------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------
class PlannerAgent:
    SYSTEM = (
        "You are an expert interview planner for tech roles. Produce exactly 5 topics. "
        "Base topics on the JD and Resume. Each topic includes: a human-friendly title, "
        "one sharp question, and 3-6 bullet points that constitute an excellent answer."
    )

    @staticmethod
    def plan(llm: LLM, jd: str, resume: str, role_title: str) -> InterviewPlan:
        user = f"""
ROLE TITLE: {role_title}
JOB DESCRIPTION:
{jd}

RESUME:
{resume}
        """.strip()
        schema = {
            "name": "interview_plan_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "role_summary": {"type": "string"},
                    "topics": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "question": {"type": "string"},
                                "expected_answer_bullets": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 3,
                                    "maxItems": 6,
                                },
                            },
                            "required": ["title", "question", "expected_answer_bullets"],
                        },
                        "minItems": 5,
                        "maxItems": 5,
                    },
                },
                "required": ["role_summary", "topics"],
                "additionalProperties": False,
            },
        }
        data = llm.complete(PlannerAgent.SYSTEM, user, json_schema=schema)
        return InterviewPlan(**data)


class GreeterAgent:
    SYSTEM = (
        "You are a concise, warm interview greeter. Your goals: "
        "(1) greet the interviewee by first name if known, (2) briefly state the role, "
        "(3) confirm the candidate's full name. Keep it under 3 sentences."
    )

    @staticmethod
    def greet(llm: LLM, name_guess: Optional[str], role_title: str) -> str:
        guess = name_guess or "there"
        user = (
            f"Candidate hint name: {guess}\nRole title: {role_title}. "
            "Write your greeting and end with a direct question asking to confirm their full name."
        )
        return llm.complete(GreeterAgent.SYSTEM, user)


class InterviewAgent:
    SYSTEM_CHECK = (
        "You are an interview co-pilot. Given a topic, the question, the candidate's answer, "
        "and the expected bullet points, assess if the answer is complete, partial, or missing "
        "with a one-sentence rationale."
    )

    SYSTEM_RATE = (
        "You are a rigorous interviewer. Rate the candidate's answer on a 1.0-5.0 scale "
        "(one decimal) based on correctness, depth, clarity, and relevance. Respond with only the number."
    )

    @staticmethod
    def encourage_prompt() -> str:
        return "Thanks. Could you add any specific details, examples, metrics, or trade-offs you considered?"

    @staticmethod
    def check_completeness(llm: LLM, topic: Topic, answer: str) -> Dict[str, str]:
        user = json.dumps(
            {
                "topic_title": topic.title,
                "question": topic.question,
                "expected_answer_bullets": topic.expected_answer_bullets,
                "candidate_answer": answer,
            }
        )
        schema = {
            "name": "completeness_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["complete", "partial", "missing"]},
                    "rationale": {"type": "string"},
                },
                "required": ["status", "rationale"],
            },
        }
        return llm.complete(InterviewAgent.SYSTEM_CHECK, user, json_schema=schema)

    @staticmethod
    def rate_answer(llm: LLM, topic: Topic, answer: str) -> float:
        user = (
            f"QUESTION: {topic.question}\n"
            f"EXPECTED BULLETS: {topic.expected_answer_bullets}\n"
            f"CANDIDATE ANSWER: {answer}\n"
            "Respond only with a number between 1.0 and 5.0 with one decimal."
        )
        val = llm.complete(InterviewAgent.SYSTEM_RATE, user)
        try:
            return float(str(val).strip())
        except Exception:
            return 3.0


class AssessorAgent:
    SYSTEM = (
        "You are an interview assessor. Given the plan topics (with expected bullets) and the transcript, "
        "evaluate per-topic alignment, give a per-topic score (1-5), summarize strengths, and list practical improvements. "
        "Return JSON with: overall_score (1-5), strengths[], improvements[], topic_scores[] (length 5)."
    )

    @staticmethod
    def assess(llm: LLM, plan: InterviewPlan, transcript: Transcript) -> Assessment:
        user = json.dumps({"plan": plan.model_dump(), "transcript": transcript.model_dump()})
        schema = {
            "name": "assessment_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "overall_score": {"type": "number"},
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "improvements": {"type": "array", "items": {"type": "string"}},
                    "topic_scores": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 5,
                        "maxItems": 5,
                    },
                },
                "required": ["overall_score", "strengths", "improvements", "topic_scores"],
            },
        }
        data = llm.complete(AssessorAgent.SYSTEM, user, json_schema=schema)
        return Assessment(**data)


# --------------------------------------------------------------------
# LangGraph Orchestration
# --------------------------------------------------------------------
@dataclass
class GraphState:
    jd: str
    resume: str
    role_title: str
    candidate_name: Optional[str] = None
    plan: Optional[InterviewPlan] = None
    transcript: Optional[Transcript] = None
    assessment: Optional[Assessment] = None


def node_plan(state: GraphState) -> GraphState:
    console.rule("[bold green]PlannerAgent")
    llm = LLM()
    plan = PlannerAgent.plan(llm, state.jd, state.resume, state.role_title)
    state.plan = plan
    state.transcript = Transcript(
        interviewee_name=None,
        role_title=state.role_title,
        jd_excerpt=(state.jd or "")[:800],
        created_at=datetime.utcnow().isoformat(),
        plan=plan,
    )
    console.print(Panel.fit("Interview plan generated (5 topics).", title="PlannerAgent", border_style="green"))
    return state


def node_greet(state: GraphState) -> GraphState:
    console.rule("[bold cyan]GreeterAgent")
    assert state.plan is not None and state.transcript is not None
    llm = LLM()
    greeting = GreeterAgent.greet(llm, state.candidate_name, state.role_title)
    console.print("\n[bold]Agent:[/bold] ", greeting)

    state.transcript.turns.append(Turn(role="agent", content=greeting))

    # Pause for candidate's name (dev server / Studio UI will show Resume input)
    name = interrupt({"expect": "candidate_name", "prompt": "Please confirm your full name."}) or ""
    name = name.strip()
    state.candidate_name = name
    state.transcript.interviewee_name = name
    state.transcript.turns.append(Turn(role="candidate", content=name))
    return state


def node_interview(state: GraphState) -> GraphState:
    console.rule("[bold magenta]InterviewAgent")
    assert state.plan is not None and state.transcript is not None
    llm = LLM()

    for idx, topic in enumerate(state.plan.topics):
        console.print(Panel.fit(f"Topic {idx + 1}: {topic.title}", border_style="magenta"))
        q = topic.question
        console.print(f"[bold]Agent:[/bold] {q}")
        state.transcript.turns.append(Turn(role="agent", content=q, topic_index=idx))

        # Pause for initial answer
        cand = interrupt({"expect": "answer", "topic_index": idx, "question": q}) or ""
        state.transcript.turns.append(Turn(role="candidate", content=cand, topic_index=idx))

        # Completeness check
        comp = InterviewAgent.check_completeness(llm, topic, cand)
        completeness = comp.get("status", "partial")
        rationale = comp.get("rationale", "")
        state.transcript.turns.append(
            Turn(
                role="agent",
                content=f"Completeness check: {completeness} — {rationale}",
                topic_index=idx,
                completeness=completeness,
            )
        )
        console.print(f"[dim]Completeness: {completeness} — {rationale}[/dim]")

        # Encourage if not complete, then pause again
        if completeness != "complete":
            nudge = InterviewAgent.encourage_prompt()
            console.print(f"[bold]Agent:[/bold] {nudge}")
            state.transcript.turns.append(Turn(role="agent", content=nudge, topic_index=idx))

            extra = interrupt({"expect": "answer_extra", "topic_index": idx}) or ""
            if extra.strip():
                state.transcript.turns.append(Turn(role="candidate", content=extra, topic_index=idx))
                cand = cand + "\n" + extra

        # Rate
        rating = InterviewAgent.rate_answer(llm, topic, cand)
        state.transcript.turns.append(
            Turn(role="agent", content=f"Rating: {rating:.1f}", topic_index=idx, rating=rating)
        )
        console.print(f"[bold]Agent:[/bold] Thanks. Score for this question: [bold]{rating:.1f}/5.0[/bold]\n")

    return state


def node_assess(state: GraphState) -> GraphState:
    console.rule("[bold yellow]AssessorAgent")
    assert state.plan is not None and state.transcript is not None
    llm = LLM()
    assessment = AssessorAgent.assess(llm, state.plan, state.transcript)
    state.assessment = assessment

    table = Table(title="Interview Feedback", show_lines=True)
    table.add_column("Aspect")
    table.add_column("Details")
    table.add_row("Overall Score", f"{assessment.overall_score:.1f} / 5.0")
    table.add_row("Strengths", "\n".join(assessment.strengths))
    table.add_row("Improvements", "\n".join(assessment.improvements))
    console.print(table)
    return state


def node_finish(state: GraphState) -> GraphState:
    tsdir = Path("transcripts")
    tsdir.mkdir(exist_ok=True)
    tsname = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_transcript.json"
    out = tsdir / tsname
    with out.open("w", encoding="utf-8") as f:
        json.dump(state.transcript.model_dump(), f, ensure_ascii=False, indent=2)
    console.print(Panel.fit(f"Transcript saved to {out}", border_style="yellow"))

    if state.assessment:
        aout = tsdir / tsname.replace("_transcript.json", "_assessment.json")
        with aout.open("w", encoding="utf-8") as f:
            json.dump(state.assessment.model_dump(), f, ensure_ascii=False, indent=2)
        console.print(Panel.fit(f"Assessment saved to {aout}", border_style="yellow"))

    console.rule("[bold]Done")
    return state


# --------------------------------------------------------------------
# Graph definition (no custom checkpointer)
# --------------------------------------------------------------------
graph = (
    StateGraph(GraphState)
    .add_node("plan", node_plan)
    .add_node("greet", node_greet)
    .add_node("interview", node_interview)
    .add_node("assess", node_assess)
    .add_node("finish", node_finish)
    .set_entry_point("plan")
    .add_edge("plan", "greet")
    .add_edge("greet", "interview")
    .add_edge("interview", "assess")
    .add_edge("assess", "finish")
    .compile(name="New Graph")
)
