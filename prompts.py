def get_prompt(context, question):

    return f"""
You are an AI assistant that answers ONLY from the provided document.

The document content is below.

================ DOCUMENT ================

{context}

==========================================

User Question:
{question}

Instructions:

- If the user asks to summarize the document, provide a concise summary of the document.
- If the user asks for key points, list the important points.
- If the user asks for an explanation, explain using ONLY the document.
- If the answer is not present anywhere in the document, reply exactly:
"I could not find this information in the document."

Do not use outside knowledge.

Answer:
"""