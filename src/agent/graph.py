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

    def rate(
        self,
        system: str,
        user: str,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str | InterviewEvaluation:
        if json_schema:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_schema", "json_schema": json_schema},
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
            )
            return resp.choices[0].message.content.strip()

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
    rating: Optional[InterviewEvaluation] = None
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

class CriterionEvaluation(BaseModel):
    score: int = Field(..., ge=1, le=5)
    justification: str

class InterviewEvaluation(BaseModel):
    question: str
    answer: str
    scores: dict[str, CriterionEvaluation]

# --------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------
class PlannerAgent:
    SYSTEM = (
        "You are an expert interview planner. "
        "Produce exactly 5 topics. "
        "Base topics on the JD and Resume. Each topic includes: a human-friendly title, "
        "one sharp question, and 3-6 bullet points that constitute an excellent answer."
    )

    NEW_SYSTEM = (
        "You are an expert recruitment assistant specializing in generating interview questions tailored to a specific job description and candidate resume."
        "Your goal is to produce high-quality interview questions that test both technical/functional skills and behavioral/soft skills, while also reflecting the hiring company's values and success factors for the role."
        "Instructions:"
        "   1. Read the job description and resume carefully."
        "   2. Identify key skills and experiences relevant to the role."
        "   3. Identify potential gaps in the candidate's experience."
        "   4. Identify business values from job description."
        "   5. Generate questions that assess these skills and experiences."
        "Rules for output:"
        "- Produce exactly 5 topics:"
        "    1. Technical / domain expertise"
        "    2. Problem-solving & execution"
        "    3. Leadership & collaboration"
        "    4. Values / culture fit"
        "    5. Growth and adaptability"
        "- Each topic includes: a human-friendly title, one sharp question, and 3-6 bullet points that constitute an excellent answer."
        "- Each question must be tailored to both the job description and the candidate’s resume."
        "- At least **one question must explicitly ask about a past experience** (e.g., 'Tell me about a time when…' or 'Give me an example of…')."
        "- Wording should be professional and neutral, without bias or assumptions."
        "- Output must be JSON for programmatic use."
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
        data = llm.complete(PlannerAgent.NEW_SYSTEM, user, json_schema=schema)
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
        "Examples:"
        "    - topic: Leadership and Team Development in AI Engineering"
        "    - question: How do you foster a culture of continuous improvement and mentorship within a diverse AI engineering team?"
        "    - expected answer bullets:"
        "        --Encourage open communication and regular micro feedback to promote mutual accountability and shared ownership."
        "        --Implement coaching and mentorship programs tailored to individual engineers' growth areas and career goals."
        "        --Promote knowledge sharing practices such as pair programming and code reviews to enhance team collaboration and skill development."
        "        --Create an environment where diverse perspectives are actively sought and valued to drive innovation and optimal outcomes."
        "        --Set clear goals and celebrate achievements to motivate the team and reinforce a culture of continuous learning and improvement."
        " Complete answer example:"
        "   For me, continuous improvement starts with creating trust and open communication. I make sure we do regular retros and encourage micro-feedback so that everyone feels safe pointing out what can be better. I also run a mentorship program that’s flexible—some folks need technical coaching on ML ops, others want career guidance—so it’s tailored. We also use pair programming and code reviews as a norm, not just for catching bugs but for spreading knowledge. Because our team is diverse, I actively invite different perspectives into design discussions; I’ve seen the best ideas come from the least expected places. Finally, I make sure we celebrate small wins—whether it’s shipping a new model to prod or a junior engineer leading a demo—because those moments reinforce that we’re always learning and moving forward together."
        "   Response:"
        "       Completeness: complete"
        "       Rationale: Covers all expected points—feedback, mentorship, knowledge sharing, valuing diversity, and celebrating achievements."
        " Partial answer example:"
        "   I believe the best way to keep an AI team improving is by encouraging open communication and feedback. We use retrospectives and 1:1s to make sure issues don’t pile up. I’m also a big fan of code reviews because they naturally create mentoring moments, especially for juniors. And when someone hits a milestone, I like to acknowledge it publicly—it keeps motivation high. Diversity is important to me too, though I’d admit I could be more deliberate in building structures to make sure every perspective gets heard."
        "   Response:"
        "       Completeness: partial"
        "       Rationale: Touches feedback, knowledge sharing, and recognition, but mentorship programs and fostering diversity aren’t fully addressed."
        "       Follow-up: What specific mentorship programs have you implemented, and how do you ensure diverse perspectives are included in discussions?"
        " Missing answer example:"
        "   I usually focus on making sure projects are delivered on time and that the team knows what’s expected of them. I keep track of progress, assign tasks clearly, and make sure blockers are resolved quickly. I believe when people have clarity on their work, they naturally perform better."
        "   Response:"
        "       Completeness: missing"
        "       Rationale: Stays at task management level, missing key aspects like mentorship, feedback culture, knowledge sharing, and diversity inclusion."
        "       Follow-up: What strategies do you have in place to promote a culture of continuous improvement within your team?"

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
                    "follow_up": {"type": "string"},
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
        
    @staticmethod
    def rate_answer_V2(llm: LLM, topic: Topic, answer: str, jd: str, resume: str, role_title: str) -> InterviewEvaluation:
            system = (
                "You are an experienced hiring manager and interview evaluator. "
                "Your job is to assess candidate answers fairly, consistently, and based on evidence, not personal bias. "
                "You evaluate only what the candidate says, not what you assume. "
                "Use the following evaluation criteria: "
                "   1. Content & Relevance – Did the candidate directly answer the question and stay on topic? Staying on topic and providing relevant information increases the score. Digressions or irrelevant information will lower the score."
                "   2. Clarity & Structure – Was the answer logical, structured, and easy to follow? Logical, structured and easy to follow answers will receive higher scores. Chaotic or poorly structured answers will lower the score."
                "   3. Depth & Insight – Did the candidate demonstrate deep knowledge, critical thinking, or reflection? Higher score will be given to candidates who provide more in-depth answers and demonstrate critical thinking. Lower scores will be given to those who provide superficial or generic responses."
                "   4. Impact & Results – Did the candidate provide evidence of outcomes, metrics, or business value? Focus on business value and measurable outcomes will increase the score. Score will be decreased for vague or generic responses or responses lacking evidence and neglecting business context."
                "   5. Behavioral Signals – Did the candidate demonstrate ownership, collaboration, adaptability, or leadership? Are the character desired for the role?"
                "   6. Communication Style – Was the communication clear, confident, and professional? When candidate was asked about his/her experience, was he/she answering using plural (we did) or singular (I did)?"
                "   7. Personality Coherence - If the candidate was referring to past experience was the experience in line with resume? Did the candidate choose the right examples?"
                "   8. Cultural Fit - Was candidate professional and respectful? Was language style appropriate? Did candidate information that typically should remain confidential? (e.g., company secrets, sensitive data, customer name of current and past employers)"
                "Scoring scale (for each criterion): "
                "   - 5 = Excellent (clear, specific, strong evidence, highly relevant) "
                "   - 4 = Good (mostly strong, some minor gaps) "
                "   - 3 = Adequate (meets baseline, but lacks depth or structure) "
                "   - 2 = Weak (vague, generic, incomplete) "
                "   - 1 = Poor (off-topic, incoherent, or irrelevant) "
                "Always: "
                "   - Give a score (1–5) for each criterion. 1 is low, 3 is neutral, 5 is high. If you do not have enough data to score apply neutral score (3) "
                "   - Justify each score with short evidence-based notes (e.g., quotes or summary from the candidate’s answer). "
                "Return a JSON object with: "
                "   - question (string): the interview question "
                "   - answer (string): the candidate's answer "
                "   - scores (object): for each criterion, an object with 'score' (int, 1-5) and 'justification' (string) "
                "Example: {\"question\": ..., \"answer\": ..., \"scores\": {\"content_relevance\": {\"score\": 4, \"justification\": \"...\"}, ...}} "
            )
            user = (
                f"Job Description: {jd}\n"
                f"Resume: {resume}\n"
                f"Role Title: {role_title}\n"
                f"QUESTION: {topic.question}\n"
                f"EXPECTED BULLETS: {topic.expected_answer_bullets}\n"
                f"CANDIDATE ANSWER: {answer}\n"
            )
            schema = {
                "name": "interview_evaluation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "scores": {
                            "type": "object",
                            "properties": {
                                "content_relevance": {
                                    "type": "object",
                                    "properties": {
                                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                                        "justification": {"type": "string"}
                                    },
                                    "required": ["score", "justification"]
                                },
                                "clarity_structure": {
                                    "type": "object",
                                    "properties": {
                                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                                        "justification": {"type": "string"}
                                    },
                                    "required": ["score", "justification"]
                                },
                                "depth_insight": {
                                    "type": "object",
                                    "properties": {
                                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                                        "justification": {"type": "string"}
                                    },
                                    "required": ["score", "justification"]
                                },
                                "impact_results": {
                                    "type": "object",
                                    "properties": {
                                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                                        "justification": {"type": "string"}
                                    },
                                    "required": ["score", "justification"]
                                },
                                "behavioral_signals": {
                                    "type": "object",
                                    "properties": {
                                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                                        "justification": {"type": "string"}
                                    },
                                    "required": ["score", "justification"]
                                },
                                "communication_style": {
                                    "type": "object",
                                    "properties": {
                                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                                        "justification": {"type": "string"}
                                    },
                                    "required": ["score", "justification"]
                                },
                                "personality_coherence": {
                                    "type": "object",
                                    "properties": {
                                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                                        "justification": {"type": "string"}
                                    },
                                    "required": ["score", "justification"]
                                },
                                "cultural_fit": {
                                    "type": "object",
                                    "properties": {
                                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                                        "justification": {"type": "string"}
                                    },
                                    "required": ["score", "justification"]
                                }
                            },
                            "required": [
                                "content_relevance",
                                "clarity_structure",
                                "depth_insight",
                                "impact_results",
                                "behavioral_signals",
                                "communication_style",
                                "personality_coherence",
                                "cultural_fit"
                            ]
                        }
                    },
                    "required": ["question", "answer", "scores"]
                }
            }
            val = llm.rate(system, user, json_schema=schema)
            try:
                return InterviewEvaluation(**val)
            except Exception:
                return {}


class AssessorAgent:
    SYSTEM = (
        "You are an interview assessor. Given the plan topics (with expected bullets) and the transcript, "
        "evaluate per-topic alignment, give a per-topic score (1-5), summarize strengths, and list practical improvements. "
        "Calculate the topic score based on the following algorythm:"
        "   1. Average the scores for each sub-criterion within a topic to get the topic score."
        "   2. Multiply the average topic score by waight of completness where: complete=1, partial=0.5, missing=0"
        "Return JSON with: overall_score (1-5), strengths[], improvements[], topic_scores[] (length 5)."
    )

    @staticmethod
    def assess(llm: LLM, plan: InterviewPlan, transcript: Transcript) -> Assessment:
        user = json.dumps({"plan": plan.model_dump(), "transcript": transcript.turns.model_dump()})
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
    transcript_path: Optional[str] = None
    assessment_path: Optional[str] = None


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
        follow_up = comp.get("follow_up", "")
        state.transcript.turns.append(
            Turn(
                role="agent",
                content=f"Completeness check: {completeness} — {rationale} - {follow_up}",
                topic_index=idx,
                completeness=completeness,
                follow_up=follow_up,
            )
        )
        console.print(f"[dim]Completeness: {completeness} — {rationale}[/dim]")

        # Encourage if not complete, then pause again
        if completeness != "complete":
            nudge = follow_up
            console.print(f"[bold]Agent:[/bold] {nudge}")
            state.transcript.turns.append(Turn(role="agent", content=nudge, topic_index=idx))

            extra = interrupt({"expect": "answer_extra", "topic_index": idx, "question": nudge}) or ""
            if extra.strip():
                state.transcript.turns.append(Turn(role="candidate", content=extra, topic_index=idx))
                cand = cand + "\n" + extra

        comp = InterviewAgent.check_completeness(llm, topic, cand)
        completeness = comp.get("status", "partial")
        rationale = comp.get("rationale", "")

        # Rate
        rating = InterviewAgent.rate_answer_V2(llm, topic, cand, state.jd, state.resume, state.role_title)
        # Ensure rating is a valid InterviewEvaluation instance

        state.transcript.turns.append(
            Turn(
                role="agent", 
                content=f"Rating: {rating}, Completeness: {completeness}", 
                topic_index=idx, 
                rating=rating, 
                completeness=completeness
            )
        )
        console.print(f"[bold]Agent:[/bold] Thanks. Score for this question: [bold]{rating}[/bold]\n")

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
    state.transcript_path = str(out.resolve())
    console.print(Panel.fit(f"Transcript saved to {out}", border_style="yellow"))

    if state.assessment:
        aout = tsdir / tsname.replace("_transcript.json", "_assessment.json")
        with aout.open("w", encoding="utf-8") as f:
            json.dump(state.assessment.model_dump(), f, ensure_ascii=False, indent=2)
        state.assessment_path = str(aout.resolve())
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
