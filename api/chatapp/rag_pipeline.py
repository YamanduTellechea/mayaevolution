
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import spacy


class RAGChatbot:
    def __init__(self, dataset_path, embedding_file="movie_embeddings.faiss",
                 embedding_model_name="sentence-transformers/all-distilroberta-v1",
                 similarity_threshold=0.5, openai_api_key=None):
        """
        Inicialización del chatbot RAG con FAISS para búsquedas eficientes.
        """
        print("Inicializando RAGChatbot...")

        self.similarity_threshold = similarity_threshold
        self.openai_api_key = openai_api_key

        # Configurar OpenAI
        if openai_api_key:
            import openai
            openai.api_key = openai_api_key

        # Carga de modelo de embeddings
        print(f"Cargando modelo de embeddings {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name, device="cuda")

        # Carga del modelo de NER
        print("Cargando modelo de NER con spaCy...")
        self.nlp = spacy.load("en_core_web_sm")

        # FAISS Index
        self.index = None
        self.id_to_movie = {}
        self.dataset_path = dataset_path
        self.embedding_file = embedding_file

        # Inicializar datos y embeddings
        self._initialize_data()

    def _initialize_data(self):
        """
        Carga los datos y crea el índice FAISS si no existe.
        """
        print(f"Cargando dataset desde {self.dataset_path}...")
        self._load_movies(self.dataset_path)

        if os.path.exists(self.embedding_file):
            print(f"Cargando índice FAISS desde {self.embedding_file}...")
            self.load_faiss_index(self.embedding_file)
        else:
            print("Generando embeddings para las películas y construyendo índice FAISS...")
            embeddings = self._generate_embeddings()
            self.create_faiss_index(embeddings)
            print(f"Guardando índice FAISS en {self.embedding_file}...")
            self.save_faiss_index(self.embedding_file)

    def _load_movies(self, dataset_path):
        """
        Carga y procesa los datos del dataset.
        """
        movies_file = os.path.join(dataset_path, "movies_metadata.csv")
        credits_file = os.path.join(dataset_path, "credits.csv")
        keywords_file = os.path.join(dataset_path, "keywords.csv")

        if not all(os.path.exists(f) for f in [movies_file, credits_file, keywords_file]):
            raise FileNotFoundError("Uno o más archivos de dataset no encontrados.")

        df_movies = pd.read_csv(movies_file, low_memory=False)
        df_credits = pd.read_csv(credits_file, low_memory=False)
        df_keywords = pd.read_csv(keywords_file, low_memory=False)

        df_movies = df_movies.dropna(subset=["title", "overview", "id", "vote_average", "genres"])
        df_movies["id"] = df_movies["id"].astype(str)
        df_credits["id"] = df_credits["id"].astype(str)
        df_keywords["id"] = df_keywords["id"].astype(str)

        def extract_top_actors(cast_string):
            try:
                cast = eval(cast_string)
                return [member["name"] for member in cast[:5]]
            except:
                return []

        def parse_keywords(keywords_string):
            try:
                keywords = eval(keywords_string)
                return [keyword["name"] for keyword in keywords]
            except:
                return []

        def parse_genres(genre_string):
            try:
                genres = json.loads(genre_string.replace("'", '"'))
                return [genre["name"] for genre in genres if "name" in genre]
            except:
                return []

        df_credits["top_actors"] = df_credits["cast"].apply(extract_top_actors)
        df_keywords["keywords_list"] = df_keywords["keywords"].apply(parse_keywords)
        df_movies["genres_list"] = df_movies["genres"].apply(parse_genres)

        df_movies = df_movies.merge(df_credits[["id", "top_actors"]], on="id", how="left")
        df_movies = df_movies.merge(df_keywords[["id", "keywords_list"]], on="id", how="left")

        self.movies = []
        for _, row in df_movies.iterrows():
            movie_data = {
                "id": row["id"],
                "title": row["title"],
                "overview": row["overview"],
                "genres": row["genres_list"],
                "actors": row["top_actors"] if isinstance(row["top_actors"], list) else [],
                "keywords": row["keywords_list"] if isinstance(row["keywords_list"], list) else [],
                "rating": round(float(row["vote_average"]), 2) if row["vote_average"] > 0 else "N/A"
            }
            self.movies.append(movie_data)
            self.id_to_movie[len(self.movies) - 1] = movie_data

        print(f"Total de películas cargadas: {len(self.movies)}")

    def _generate_embeddings(self):
        """
        Genera embeddings para las películas.
        """
        texts = [
            f"Movie Title: {movie['title']}. Description: {movie['overview']}. "
            f"Genres: {', '.join(movie['genres'])}. Actors: {', '.join(movie['actors'])}. "
            f"Keywords: {', '.join(movie['keywords'])}. Rating: {movie['rating']}."
            for movie in self.movies
        ]
        return self.embedding_model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)

    def create_faiss_index(self, embeddings):
        """
        Crea un índice FAISS y lo almacena en memoria.
        """
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(embeddings)

    def save_faiss_index(self, file_path):
        """
        Guarda el índice FAISS en disco.
        """
        faiss.write_index(self.index, file_path)

    def load_faiss_index(self, file_path):
        """
        Carga el índice FAISS desde disco.
        """
        self.index = faiss.read_index(file_path)

    def extract_entities(self, query):
        """
        Extrae actores, géneros y nombres de películas de la consulta del usuario.
        """
        doc = self.nlp(query)
        actors = []
        genres = []
        movies = []

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
        Genera una respuesta utilizando el índice FAISS para la búsqueda por similaridad.
        """
        query_emb = self.embedding_model.encode([query], convert_to_tensor=False, normalize_embeddings=True)
        distances, indices = self.index.search(query_emb, k=3)

        contexts = [
            self.id_to_movie[i]
            for i, dist in zip(indices[0], distances[0])
            if dist > self.similarity_threshold
        ]

        if not contexts:
            return {"query": query, "answer": "No se encontraron recomendaciones relevantes."}
        
        response = [
                 {
                     "title": movie["title"],
                     "overview": movie["overview"],
                     "genres": ", ".join(movie["genres"]),
                     "actors": ", ".join(movie["actors"]),
                     "rating": movie["rating"]
                 }
                 for movie in contexts
             ]

        return response