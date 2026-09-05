# This is the "generation" half of RAG: takes the chunks Qdrant already found relevant
# and asks Gemini to compose a natural-language answer grounded in them. No search happens
# here -- retrieval is qdrant_manager.search_text_chunks(), called by app.py/query.py
# before this class is ever invoked.
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

class GeminiRAG:
    # "pro" models require billing enabled (this free-tier key gets 0 quota for them);
    # "flash" models have real free-tier quota and are plenty capable for grounded Q&A
    # over retrieved text chunks (no vision needed since the pipeline is text-only).
    def __init__(self, model_name="gemini-3.6-flash", temperature=0.2):
        """
        Initializes the Gemini chat model used to generate answers.
        Low temperature (0.2) favors grounded, consistent answers over creative ones --
        appropriate for a factual Q&A tool over technical/financial documents.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key)
        self.system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the provided text context (excerpts extracted from PDF documents) to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Provide a detailed, accurate response based ONLY on the provided context."
        )

    def answer_question(self, question: str, context_chunks: list[dict]) -> str:
        """
        Answers a question using the provided list of retrieved text chunks as context.
        Each chunk is a dict with "text" and optional "page_number" keys (as returned by
        QdrantManager.search_text_chunks).
        """
        print(f"Sending request to Gemini LLM ({self.llm.model}) ...")

        # Concatenate all retrieved chunks into one context block, tagging each with its
        # source page so the model (and, indirectly, the user) can see where an answer
        # came from.
        context_text = "\n\n".join(
            f"[Page {chunk.get('page_number', '?')}]\n{chunk['text']}"
            for chunk in context_chunks
        )
        user_content = f"Context:\n{context_text}\n\nQuestion: {question}"

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content)
        ]

        response = self.llm.invoke(messages)
        # Newer Gemini models return content as a list of structured blocks (text +
        # metadata) rather than a plain string; .text normalizes either shape into a
        # single string, which is what QueryResponse.answer (a plain str field) expects.
        return response.text
