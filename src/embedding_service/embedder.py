# Thin wrapper around Google's Gemini embedding API. Both text chunks (at ingestion time)
# and user questions (at query time) are embedded with the SAME model, so their vectors
# live in the same space and can be compared with cosine similarity in Qdrant.
import os
from google import genai
from PIL import Image

class GeminiEmbedder:
    def __init__(self, model_name="gemini-embedding-2"):
        """
        Initializes the Google GenAI client and sets the model.
        The user specified "gemini-embedding-2".
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def embed_image(self, image: Image.Image) -> list[float]:
        """
        Embeds a single PIL Image.
        Note: kept for the legacy image-based pipeline (see src/embedding_service/main.py);
        the current text-chunk RAG pipeline only uses embed_text() below.
        """
        print(f"Embedding image using model: {self.model_name}...")
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=image
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"Error embedding image: {e}")
            raise e

    def embed_text(self, text: str) -> list[float]:
        """
        Embeds a text string (either a document chunk during ingestion, or a user
        question at query time) into a single vector.
        """
        print(f"Embedding text using model: {self.model_name}...")
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            # embed_content returns a list of embeddings (one per input); we only ever
            # pass a single string, so we always take the first result.
            return result.embeddings[0].values
        except Exception as e:
            print(f"Error embedding text: {e}")
            raise e

