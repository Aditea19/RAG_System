import os
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def make_conversational_chain(retriever, model_name: str = "gemini-2.5-flash"):
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )

    # HISTORY AWARE RETRIEVER
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        prompt=ChatPromptTemplate.from_template("""
You are a retrieval-optimized assistant.
Rewrite the user's question ONLY if needed for better search.
Chat history: {chat_history}
Original question: {input}
""")
    )

    # ANSWER FORMATTING 
    qa_prompt = ChatPromptTemplate.from_template("""
You are an AI assistant answering strictly from the provided context.  
Follow these rules:

1. *Use ONLY the information from the context.*  
2. If the answer is not present, reply exactly: *"Not available in the document."*
3. Be concise, clear, factual.
4. No assumptions. No extra knowledge.
5. Provide a small, clean paragraph — not too long, not too short.
6. If the user asks a definition, give a simple explanation.
7. If multiple documents mention the answer, merge the information.
8. Write the answer in your own words.
9. If the question requires details, provide them but avoid copying entire chunks.

--------------------
CONTEXT:
{context}
--------------------

Question: {input}

Give the best possible answer using ONLY what is above.
""")

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, qa_chain)


# Summarizer

def make_summarizer_chain(model_name: str = "gemini-2.5-flash"):
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)

    from langchain_classic.chains import LLMChain
    from langchain_core.prompts import PromptTemplate

    template = """
Summarize the document into {bullet_count} short, clear bullet points.

Your rules:
- Use ONLY the information from the document.
- Do NOT make the bullets too long.
- Do NOT oversimplify; include all important ideas.
- Do NOT hallucinate.
- Merge repetitive content smartly.
- Keep wording concise and readable.

Document:
{doc}
"""

    prompt = PromptTemplate(
        input_variables=["doc", "bullet_count"],
        template=template
    )

    return LLMChain(llm=llm, prompt=prompt)



# Compare chain

def make_compare_chain(model_name: str = "gemini-2.5-flash"):
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)

    from langchain_classic.chains import LLMChain
    from langchain_core.prompts import PromptTemplate

    template = """
You are comparing *multiple documents* strictly based on the content provided.

The documents you must compare:
{filenames}

Your rules:
- Use ONLY the information found in the provided text.
- Do NOT invent or assume anything.
- If a topic appears ONLY in a specific document, mention exactly which one.
- Do NOT merge concepts unless they appear in multiple documents.
- Do NOT hallucinate similarities or differences.
- Maintain a clear comparison structure.

Below is the combined text of all documents, grouped by filename:

{all_docs}

Now produce a structured comparison with these sections:

1. *Overview of Each Document*
   - Summarize each document separately by filename.
   - Mention the unique focus or purpose of each.

2. *Similarities Across Documents*
   - ONLY list similarities that truly appear in more than one document.
   - Clearly mention which documents share the similarity.

3. *Differences Between Documents*
   - Be VERY precise.
   - Mention which document discusses what.
   - If a topic or concept appears in only one document, state that explicitly.

4. *Conclusion*
   - A clear and concise comparison statement summarizing the relationship among all documents.

Remember:
- NO hallucinations.
- NO assumptions.
- ONLY text-based evidence.
"""

    prompt = PromptTemplate(
        input_variables=["all_docs", "filenames"],
        template=template
    )

    return LLMChain(llm=llm, prompt=prompt)