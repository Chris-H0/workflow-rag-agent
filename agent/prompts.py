RETRIEVAL_DECISION_PROMPT = (
    "You are deciding the next step for a retrieval-augmented QA system.\n"
    "Original question: {question}\n\n"
    "Retrieved context so far:\n{context}\n\n"
    "Choose exactly one decision:\n"
    "- answer: the context is relevant and sufficient to answer the original question.\n"
    "- retrieve_more: the context is relevant but missing information needed to answer.\n"
    "- rewrite: the context is not relevant to the original question.\n"
)

FOLLOWUP_QUERY_PROMPT = (
    "The retrieved context so far is relevant but incomplete.\n"
    "Original question: {question}\n\n"
    "Retrieved context so far:\n{context}\n\n"
    "Ask a focused additional retrieval query that would help answer the original question."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say 'unknown'. "
    "Return only the shortest answer phrase, name, entity, date, location, or yes/no answer. "
    "Do not explain your reasoning. "
    "Do not include a full sentence unless the answer cannot be expressed as a short phrase.\n"
    "Question: {question} \n"
    "Context: {context}"
)
