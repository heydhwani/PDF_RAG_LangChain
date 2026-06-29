def get_prompt(context, question):
    """
    Creates the prompt for the Gemini model.
    """

    prompt = f"""
You are an intelligent AI assistant.

Answer the user's question ONLY using the provided context.

If the answer is not available in the context, reply exactly:

"I could not find this information in the document."

-------------------------
Context:
{context}

-------------------------
Question:
{question}

Answer:
"""

    return prompt