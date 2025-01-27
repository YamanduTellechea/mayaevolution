import os
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import spacy

class FineTunedRAGChatbot:
    def __init__(self, dataset_path, embedding_file="movie_embeddings.npz",
                 embedding_model_name="all-MPNet-base-v2", fine_tuned_model_path="./fine_tuned_model"):
        """
        Inicialización del chatbot RAG con un modelo fine-tuneado.
        """
        print("Inicializando FineTunedRAGChatbot...")

        # Cargar el modelo de embeddings
        print(f"Cargando modelo de embeddings {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name, device="cuda")

        # Cargar el modelo de NER
        print("Cargando modelo de NER con spaCy...")
        self.nlp = spacy.load("en_core_web_sm")

        # Cargar el modelo fine-tuneado
        print(f"Cargando modelo fine-tuneado desde {fine_tuned_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_model_path).to("cuda")

        # Cargar datos y embeddings
        self.movies = []
        self.embeddings = None
        self.dataset_path = dataset_path
        self.embedding_file = embedding_file

        # Inicializar datos
        self._initialize_data()

    def _initialize_data(self):
        """
        Carga los datos y embeddings si no están disponibles.
        """
        print(f"Cargando dataset desde {self.dataset_path}...")
        self._load_movies(self.dataset_path)

        if os.path.exists(self.embedding_file):
            print(f"Cargando embeddings desde {self.embedding_file}...")
            self.load_embeddings(self.embedding_file)
        else:
            print("Generando embeddings para todas las películas del dataset...")
            self.embeddings = self._generate_embeddings()
            print(f"Guardando embeddings generados en {self.embedding_file}...")
            self.save_embeddings(self.embedding_file)

    def extract_entities(self, query):
        """
        Extrae actores, géneros y nombres de películas de la consulta del usuario usando NER.
        """
        doc = self.nlp(query)
        actors = []
        genres = []
        movies = []

        # Detectar entidades en la consulta
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                actors.append(ent.text)
            elif ent.label_ == "WORK_OF_ART":
                movies.append(ent.text)
            elif ent.text.lower() in {"action", "comedy", "drama", "thriller", "romance", "horror", "family"}:
                genres.append(ent.text)

        return {
            "actors": list(set(actors)),
            "genres": list(set(genres)),
            "movies": list(set(movies))
        }

    def generate_answer(self, query: str):
        """
        Genera una respuesta utilizando el modelo fine-tuneado, considerando entidades detectadas.
        """
        entities = self.extract_entities(query)
        print(f"Entidades detectadas: {entities}")

        enriched_query = query
        if entities["actors"]:
            enriched_query += f" Actors: {', '.join(entities['actors'])}."
        if entities["genres"]:
            enriched_query += f" Genres: {', '.join(entities['genres'])}."
        if entities["movies"]:
            enriched_query += f" Movies: {', '.join(entities['movies'])}."

        print(f"Consulta enriquecida: {enriched_query}")
        query_emb = self.embed(enriched_query)

        similarities = torch.nn.functional.cosine_similarity(query_emb, self.embeddings, dim=-1)
        top_indices = torch.topk(similarities, k=5).indices.cpu().numpy()

        contexts = self.get_contexts(top_indices)
        prompt = self.build_prompt(query, contexts)

        full_response = self._model_generate(prompt)
        print(f"Respuesta completa: {full_response}")

        return full_response

    def _generate_embeddings(self):
        """
        Genera embeddings para todas las películas utilizando información enriquecida.
        """
        print("Generando embeddings...")
        texts = [
            f"Movie Title: {movie['title']}. Description: {movie['overview']}. "
            f"Genres: {', '.join(movie['genres'])}. Actors: {', '.join(movie['actors'])}. "
            f"Keywords: {', '.join(movie['keywords'])}. Rating: {movie['rating']}."
            for movie in self.movies
        ]

        embeddings = self.embedding_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        print("Embeddings generados correctamente.")
        return embeddings

    def load_embeddings(self, file_path):
        print(f"Cargando embeddings desde {file_path}...")
        data = np.load(file_path, allow_pickle=True)
        self.embeddings = torch.tensor(data["embeddings"]).to("cuda")
        print("Embeddings cargados correctamente.")

    def save_embeddings(self, file_path):
        print(f"Guardando embeddings en {file_path}...")
        np.savez(file_path, embeddings=self.embeddings.cpu().numpy())
        print("Embeddings guardados correctamente.")

    def embed(self, text):
        emb = self.embedding_model.encode(text, convert_to_tensor=True)
        return emb.unsqueeze(0)

    def get_contexts(self, indices):
        contexts = []
        for i in indices:
            if 0 <= i < len(self.movies):
                movie = self.movies[i]
                contexts.append(
                    f"Title: {movie['title']}\nOverview: {movie['overview']}\nGenres: {', '.join(movie['genres'])}\n"
                    f"Rating: {movie['rating']}\nActors: {', '.join(movie['actors'])}\nKeywords: {', '.join(movie['keywords'])}"
                )
        return contexts

    def build_prompt(self, query, contexts):
        context_str = "\n---\n".join(contexts)
        prompt = (
            f"Based on the user's request:\n'{query}'\n\n"
            f"Here are some movies:\n{context_str}\n\n"
            f"Provide a concise response based on the user's preferences."
        )
        return prompt

    def _model_generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        output = self.model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
