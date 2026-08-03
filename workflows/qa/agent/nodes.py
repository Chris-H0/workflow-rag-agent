from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from workflows.qa.agent.prompts import (
    FOLLOWUP_QUERY_PROMPT,
    GENERATE_PROMPT,
    INITIAL_RETRIEVAL_PROMPT,
    RETRIEVAL_DECISION_PROMPT,
    REWRITE_PROMPT,
)


MAX_RETRIEVAL_ROUNDS = 3


class RetrievalDecision(BaseModel):
    """Decide whether to answer, retrieve more, or rewrite the question."""

    decision: Literal["answer", "retrieve_more", "rewrite"] = Field(
        description="Next action after retrieval"
    )


def get_retrieved_context(state: MessagesState):
    return "\n\n".join(
        message.content
        for message in state["messages"]
        if isinstance(message, ToolMessage)
    )


def count_retrieval_rounds(state: MessagesState):
    return sum(1 for message in state["messages"] if getattr(message, "tool_calls", None))


def build_generate_query_or_respond(response_model, retriever_tool):
    def generate_query_or_respond(state: MessagesState):
        """Either answer directly or call the retriever tool."""
        messages = [SystemMessage(content=INITIAL_RETRIEVAL_PROMPT), *state["messages"]]
        response = response_model.bind_tools([retriever_tool]).invoke(messages)
        return {"messages": [response]}

    return generate_query_or_respond


def build_decide_after_retrieval(decision_model):
    def decide_after_retrieval(
        state: MessagesState,
    ) -> Literal["generate_answer", "generate_followup_query", "rewrite_question"]:
        """Route after retrieval based on relevance, completeness, and search budget."""
        if count_retrieval_rounds(state) >= MAX_RETRIEVAL_ROUNDS:
            return "generate_answer"

        question = state["messages"][0].content
        context = get_retrieved_context(state)
        prompt = RETRIEVAL_DECISION_PROMPT.format(question=question, context=context)
        response = decision_model.with_structured_output(RetrievalDecision).invoke(
            [{"role": "user", "content": prompt}]
        )

        if response.decision == "answer":
            return "generate_answer"

        if response.decision == "retrieve_more":
            return "generate_followup_query"

        return "rewrite_question"

    return decide_after_retrieval


def build_rewrite_question(response_model):
    def rewrite_question(state: MessagesState):
        """Rewrite the original user question."""
        question = state["messages"][0].content
        prompt = REWRITE_PROMPT.format(question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=response.content)]}

    return rewrite_question


def build_generate_followup_query():
    def generate_followup_query(state: MessagesState):
        """Ask the model to make one more focused retrieval query."""
        question = state["messages"][0].content
        context = get_retrieved_context(state)
        prompt = FOLLOWUP_QUERY_PROMPT.format(question=question, context=context)
        return {"messages": [HumanMessage(content=prompt)]}

    return generate_followup_query


def build_generate_answer(response_model):
    def generate_answer(state: MessagesState):
        """Generate an answer from the original question and retrieved context."""
        question = state["messages"][0].content
        context = get_retrieved_context(state)
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}

    return generate_answer
