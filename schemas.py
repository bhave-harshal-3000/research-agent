from typing import Literal
from pydantic import BaseModel, Field, field_validator


class ResearchTask(BaseModel):
    subquestion: str = Field(
        description="Specific research subquestion to investigate"
    )
    tool: Literal["web_search", "search_arxiv", "both"] = Field(
        default="both",
        description="Assigned tool for this task: 'web_search', 'search_arxiv', or 'both'"
    )
    query: str = Field(
        description="Targeted keyword search query optimized for search engines and academic databases"
    )


class ResearchPlan(BaseModel):
    objective: str = Field(
        description="The main objective of the research"
    )
    tasks: list[ResearchTask] = Field(
        default_factory=list,
        description="List of specific research tasks with assigned tools and search queries"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the research approach and tool allocation strategy"
    )


class SourceItem(BaseModel):
    id: int | None = Field(
        default=None,
        description="Evidence ID this source was cited as in the findings"
    )
    title: str = Field(
        description="Title of the source or paper"
    )
    url: str = Field(
        description="URL or entry ID link to the source"
    )


class Finding(BaseModel):
    claim: str = Field(
        description="A single factual claim supported by the cited evidence"
    )
    source_ids: list[int] = Field(
        default_factory=list,
        description="IDs of the evidence items that directly support the claim"
    )


class ResearchReport(BaseModel):
    user_question: str = Field(
        description="The user's original research question"
    )
    short_answer: str = Field(
        description="A concise, direct summary answering the question"
    )
    key_findings: list[Finding] = Field(
        default_factory=list,
        description="Key findings, each with the evidence IDs that support it"
    )
    sources: list[SourceItem] = Field(
        default_factory=list,
        description="Filled in by code from the evidence cited by verified findings, never by the LLM"
    )
    confidence: Literal["High", "Medium", "Low"] = Field(
        default="Medium",
        description="Assessment of confidence: High, Medium, or Low"
    )
    limitations_and_assumptions: list[str] = Field(
        default_factory=list,
        description="Known limitations of the evidence and assumptions made"
    )
    suggested_next_steps: list[str] = Field(
        default_factory=list,
        description="Actionable follow-up steps or areas for further investigation"
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalise_confidence(cls, value):
        if isinstance(value, str):
            return value.strip().capitalize()
        return value
