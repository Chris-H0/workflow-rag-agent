from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from agent.prompts import GENERATE_PROMPT, GRADE_PROMPT, REWRITE_PROMPT


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def build_generate_query_or_respond(response_model, retriever_tool):
    def generate_query_or_respond(state: MessagesState):
        """Either answer directly or call the retriever tool."""
        response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}

    return generate_query_or_respond


def build_grade_documents(grader_model):
    def grade_documents(
        state: MessagesState,
    ) -> Literal["generate_answer", "rewrite_question"]:
        """Route based on whether retrieved documents are relevant."""
        question = state["messages"][0].content
        context = state["messages"][-1].content

        prompt = GRADE_PROMPT.format(question=question, context=context)
        response = grader_model.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )

        if response.binary_score == "yes":
            return "generate_answer"

        return "rewrite_question"

    return grade_documents


def build_rewrite_question(response_model):
    def rewrite_question(state: MessagesState):
        """Rewrite the original user question."""
        question = state["messages"][0].content
        prompt = REWRITE_PROMPT.format(question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [HumanMessage(content=response.content)]}

    return rewrite_question


def build_generate_answer(response_model):
    def generate_answer(state: MessagesState):
        """Generate an answer from the original question and retrieved context."""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}

    return generate_answer
