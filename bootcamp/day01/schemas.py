"""Structured output for the support-ticket classifier."""

from typing import Literal

from pydantic import BaseModel, Field


class Entity(BaseModel):
    type: Literal["order_id", "product", "person", "email", "amount", "date", "other"]
    value: str


class Ticket(BaseModel):
    category: Literal["billing", "technical", "shipping", "account", "feedback", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    sentiment: Literal["negative", "neutral", "positive"]
    language: str = Field(description="ISO 639-1 code of the customer's message, e.g. 'en'")
    summary: str = Field(description="One sentence, in English, max 25 words")
    entities: list[Entity]
    needs_human: bool = Field(description="True if legal threats, safety issues, or the request is unclear")
