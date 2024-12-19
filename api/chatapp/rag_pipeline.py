# import os
# import json
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import pandas as pd
# import spacy

# class RAGChatbot:
#     def __init__(self, dataset_path, embedding_file="movie_embeddings.npz",
#                  embedding_model_name="all-MPNet-base-v2", base_model_name="mistralai/Mistral-7B-Instruct-v0.3"):
#         """
#         Inicialización del chatbot RAG especializado en recomendaciones de películas.
#         """
#         print("Inicializando RAGChatbot...")
#         print(f"Cargando modelo de embeddings {embedding_model_name}...")
#         self.embedding_model = SentenceTransformer(embedding_model_name, device="cuda")

#         print("Cargando modelo de NER con spaCy...")
#         self.nlp = spacy.load("en_core_web_sm")  # Carga el modelo de spaCy

#         print(f"Cargando dataset desde {dataset_path}...")
#         self.movies = []
#         self.ratings = None
#         self._load_movies(dataset_path)

#         # Cargar/generar embeddings
#         if os.path.exists(embedding_file):
#             self.load_embeddings(embedding_file)
#         else:
#             print("Generando embeddings para todas las películas del dataset...")
#             self.embeddings = self._generate_embeddings()
#             self.save_embeddings(embedding_file)

#         print(f"Cargando modelo base: {base_model_name}...")
#         self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             base_model_name,
#             torch_dtype=torch.float16
#         ).to("cuda")

   
#     def extract_entities(self, query):
#         """
#         Extrae actores, géneros y nombres de películas de la consulta del usuario usando NER.
#         """
#         doc = self.nlp(query)
#         actors = []
#         genres = []
#         movies = []

#         # Detectar entidades en la consulta
#         for ent in doc.ents:
#             if ent.label_ == "PERSON":
#                 actors.append(ent.text)
#             elif ent.label_ == "WORK_OF_ART":
#                 movies.append(ent.text)
#             elif ent.text.lower() in {"action", "comedy", "drama", "thriller", "romance", "horror", "family"}:
#                 genres.append(ent.text)

#         return {
#             "actors": list(set(actors)),
#             "genres": list(set(genres)),
#             "movies": list(set(movies))
#         }    

#     def _load_movies(self, dataset_path):
#         """
#         Carga películas de movies_metadata.csv, credits.csv y keywords.csv con géneros, actores y palabras clave procesados.
#         """
#         movies_file = os.path.join(dataset_path, "movies_metadata.csv")
#         credits_file = os.path.join(dataset_path, "credits.csv")
#         keywords_file = os.path.join(dataset_path, "keywords.csv")

#         if not os.path.exists(movies_file):
#             raise FileNotFoundError(f"No se encontró el dataset {movies_file}")
#         if not os.path.exists(credits_file):
#             raise FileNotFoundError(f"No se encontró el dataset {credits_file}")
#         if not os.path.exists(keywords_file):
#             raise FileNotFoundError(f"No se encontró el dataset {keywords_file}")

#         print(f"Cargando datos desde {movies_file}, {credits_file} y {keywords_file}...")

#         # Cargar los datos
#         df_movies = pd.read_csv(movies_file, low_memory=False)
#         df_credits = pd.read_csv(credits_file, low_memory=False)
#         df_keywords = pd.read_csv(keywords_file, low_memory=False)

#         # Filtrar filas con datos válidos
#         df_movies = df_movies.dropna(subset=["title", "overview", "id", "vote_average", "genres"])
#         df_movies = df_movies[df_movies["overview"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]

#         # Asegurarse de que las columnas `id` tengan el mismo tipo
#         df_movies["id"] = df_movies["id"].astype(str)
#         df_credits["id"] = df_credits["id"].astype(str)
#         df_keywords["id"] = df_keywords["id"].astype(str)

#         # Procesar actores principales de los créditos
#         def extract_top_actors(cast_string):
#             try:
#                 cast = eval(cast_string)  # Convertir la cadena a lista de diccionarios
#                 return [member["name"] for member in cast[:5]]  # Tomar los primeros 5 actores
#             except (SyntaxError, TypeError):
#                 return []

#         df_credits["top_actors"] = df_credits["cast"].apply(extract_top_actors)

#         # Procesar palabras clave
#         def parse_keywords(keywords_string):
#             try:
#                 keywords = eval(keywords_string)  # Convertir la cadena a lista de diccionarios
#                 return [keyword["name"] for keyword in keywords]
#             except (SyntaxError, TypeError):
#                 return []

#         df_keywords["keywords_list"] = df_keywords["keywords"].apply(parse_keywords)

#         # Unir los DataFrames
#         df_movies = df_movies.merge(df_credits[["id", "top_actors"]], on="id", how="left")
#         df_movies = df_movies.merge(df_keywords[["id", "keywords_list"]], on="id", how="left")

#         # Parsear géneros de JSON a lista de nombres
#         def parse_genres(genre_string):
#             try:
#                 genres = json.loads(genre_string.replace("'", '"'))  # Asegura que sea un JSON válido
#                 return [genre["name"] for genre in genres if "name" in genre]
#             except (json.JSONDecodeError, TypeError):
#                 return []

#         df_movies["genres_list"] = df_movies["genres"].apply(parse_genres)

#         # Procesar cada película
#         for _, row in df_movies.iterrows():
#             movie_id = int(row["id"]) if str(row["id"]).isdigit() else None
#             if movie_id is None:
#                 continue

#             # Calificación promedio redondeada a 2 decimales
#             vote_average = round(float(row["vote_average"]), 2)
#             movie = {
#                 "id": movie_id,
#                 "title": str(row["title"]),
#                 "overview": str(row["overview"]),
#                 "genres": row["genres_list"],  # Lista de géneros
#                 "actors": row["top_actors"] if isinstance(row["top_actors"], list) else [],  # Lista de actores principales
#                 "keywords": row["keywords_list"] if isinstance(row["keywords_list"], list) else [],  # Lista de palabras clave
#                 "rating": vote_average if vote_average > 0 else "N/A",  # Si la calificación es 0, marcar como N/A
#             }
#             self.movies.append(movie)

#         print(f"Total de películas cargadas: {len(self.movies)}")

      
    
#     def generate_answer(self, query: str):
#         """
#         Genera una respuesta en base a la consulta del usuario usando RAG, considerando entidades detectadas.
#         """
#         # Extraer entidades de la consulta
#         entities = self.extract_entities(query)
#         print(f"Entidades detectadas en la consulta: {entities}")

#         # Construir texto enriquecido para el embedding de la consulta
#         enriched_query = query
#         if entities["actors"]:
#             enriched_query += f" Actors: {', '.join(entities['actors'])}."
#         if entities["genres"]:
#             enriched_query += f" Genres: {', '.join(entities['genres'])}."
#         if entities["movies"]:
#             enriched_query += f" Related movies: {', '.join(entities['movies'])}."

#         print(f"Consulta enriquecida para embeddings: {enriched_query}")

#         # Generar embedding para la consulta enriquecida
#         query_emb = self.embed(enriched_query)

#         # Calcular similitudes entre el embedding de la consulta y los embeddings de las películas
#         similarities = torch.nn.functional.cosine_similarity(query_emb, self.embeddings, dim=-1)
#         top_indices = torch.topk(similarities, k=5).indices.cpu().numpy()

#         # Recuperar contextos y construir el prompt
#         retrieved_contexts = self.get_contexts(top_indices)
#         print(retrieved_contexts)

#         prompt = self.build_prompt(query, retrieved_contexts)

#         # Generar respuesta del modelo base
#         full_response = self._model_generate(prompt)
#         # Mostrar la respuesta completa generada por el modelo antes de filtrar
#         print(f"Respuesta completa del modelo:\n{full_response}")
#         filtered_response = self._format_response(full_response)
#         # Mostrar la respuesta filtrada para depuración
#         print(f"Respuesta filtrada:\n{filtered_response}")

#         return filtered_response

    
#     def get_contexts(self, indices):
#         contexts = []
#         for i in indices:
#             if 0 <= i < len(self.movies):
#                 movie = self.movies[i]
#                 title = movie["title"]
#                 overview = movie["overview"]
#                 genres = ", ".join(movie["genres"])
#                 rating = movie["rating"]
#                 actors = ", ".join(movie.get("actors", []))  # Obtiene los actores principales si están disponibles
#                 keywords = ", ".join(movie.get("keywords", []))  # Obtiene las palabras clave si están disponibles

#                 context_text = (
#                     f"Title: {title}\n"
#                     f"Overview: {overview}\n"
#                     f"Genres: {genres}\n"
#                     f"Rating: {rating}\n"
#                     f"Actors: {actors}\n"
#                     f"Keywords: {keywords}"
#                 )

#                 contexts.append(context_text)
#             else:
#                 contexts.append("Información no encontrada.")
#         return contexts



#     def build_prompt(self, query, contexts):
#         """
#         Constructs a clean and concise prompt for the model in English.
#         """
#         context_str = "\n---\n".join(contexts)
#         prompt = (
#             f"Based on the user's request:\n'{query}'\n\n"
#             f"Here are some movies:\n"
#             f"{context_str}\n\n"
#             f"Please respond with the following format for each movie:\n"
#             f"Title: <movie title>\n"
#             f"Description: <brief description>\n"
#             f"Rating: <rating or N/A>\n\n"
#             f"Response:"
#         )
#         return prompt

#     def _format_response(self, response_text):
#         """
#         Formatea la respuesta del modelo en un JSON estructurado.
#         """
#         sections = response_text.split("---")
#         movies = []

#         for section in sections:
#             lines = section.strip().split("\n")
#             movie = {}
#             for line in lines:
#                 if line.startswith("Title:"):
#                     movie["title"] = line.split("Title:", 1)[1].strip()
#                 elif line.startswith("Overview:"):
#                     movie["overview"] = line.split("Overview:", 1)[1].strip()
#                 elif line.startswith("Genres:"):
#                     movie["genres"] = line.split("Genres:", 1)[1].strip()
#                 elif line.startswith("Rating:"):
#                     movie["rating"] = line.split("Rating:", 1)[1].strip()
#             if movie:
#                 movies.append(movie)
        
#         return movies

    
#     def _generate_embeddings(self):
#         """
#         Genera embeddings para todas las películas utilizando información enriquecida
#         (título, sinopsis, actores, géneros, palabras clave y calificación).
#         """
#         print("Generando embeddings para todas las películas...")

#         # Crear textos enriquecidos para generar embeddings
#         texts = []
#         for movie in self.movies:
#             title = movie.get("title", "").strip()
#             overview = movie.get("overview", "").strip()
#             genres = ", ".join(movie.get("genres", [])).strip()
#             actors = ", ".join(movie.get("actors", [])).strip()
#             keywords = ", ".join(movie.get("keywords", [])).strip()
#             rating = movie.get("rating", "N/A")

#             # Preprocesamiento para enriquecer el contexto y evitar ruido
#             enriched_text = (
#                 f"Movie Title: {title}. "
#                 f"Description: {overview}. "
#                 f"Genres: {genres}. "
#                 f"Main Actors: {actors}. "
#                 f"Keywords: {keywords}. "
#                 f"Rating: {rating}."
#             )

#             # Limitar textos muy largos (si es necesario)
#             if len(enriched_text) > 512:
#                 enriched_text = enriched_text[:512] + "..."

#             texts.append(enriched_text)

#         # Generar embeddings utilizando el modelo
#         try:
#             embeddings = self.embedding_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
#         except Exception as e:
#             print(f"Error al generar embeddings: {e}")
#             embeddings = None

#         print("Embeddings generados correctamente.")
#         return embeddings


#     def load_embeddings(self, file_path):
#         print(f"Cargando embeddings desde {file_path}...")
#         data = np.load(file_path, allow_pickle=True)
#         self.embeddings = torch.tensor(data["embeddings"]).to("cuda")
#         print("Embeddings cargados correctamente.")

#     def save_embeddings(self, file_path):
#         print(f"Guardando embeddings en {file_path}...")
#         np.savez(file_path, embeddings=self.embeddings.cpu().numpy())
#         print("Embeddings guardados correctamente.")

#     def embed(self, text):
#         emb = self.embedding_model.encode(text, convert_to_numpy=True).reshape(1, -1)
#         return torch.tensor(emb, device="cuda").float()

#     def _model_generate(self, prompt):
#         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
#         output = self.model.generate(
#             **inputs,
#             max_new_tokens=150,
#             do_sample=True,
#             top_p=0.9,
#             temperature=0.7,
#             pad_token_id=self.tokenizer.eos_token_id,
#             repetition_penalty=1.2
#         )
#         return self.tokenizer.decode(output[0], skip_special_tokens=True)

import os
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import spacy
import bitsandbytes as bnb

class RAGChatbot:
    def __init__(self, dataset_path, embedding_file="movie_embeddings.npz",
             embedding_model_name="all-MPNet-base-v2", base_model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        """
        Inicialización del chatbot RAG especializado en recomendaciones de películas.
        """
        print("Inicializando RAGChatbot...")

        # Carga de modelo de embeddings
        print(f"Cargando modelo de embeddings {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name, device="cuda")

        # Carga del modelo de NER
        print("Cargando modelo de NER con spaCy...")
        self.nlp = spacy.load("en_core_web_sm")

        # Carga del modelo base cuantizado
        print(f"Cargando modelo base cuantizado: {base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",  # Asigna automáticamente entre CPU/GPU
            torch_dtype=torch.float16,  # Precisión reducida
            load_in_4bit=True  # Activar cuantización de 4 bits
        )


        # Carga de datos y embeddings
        self.movies = []
        self.embeddings = None
        self.dataset_path = dataset_path
        self.embedding_file = embedding_file

        # Inicializar datos
        self._initialize_data()

    def _initialize_data(self):
        """
        Carga los datos y embeddings solo si no están disponibles.
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

    def _load_movies(self, dataset_path):
        """
        Carga películas de movies_metadata.csv, credits.csv y keywords.csv con géneros, actores y palabras clave procesados.
        """
        movies_file = os.path.join(dataset_path, "movies_metadata.csv")
        credits_file = os.path.join(dataset_path, "credits.csv")
        keywords_file = os.path.join(dataset_path, "keywords.csv")

        if not all(os.path.exists(f) for f in [movies_file, credits_file, keywords_file]):
            raise FileNotFoundError("Uno o más archivos de dataset no encontrados.")

        print(f"Cargando datos desde {movies_file}, {credits_file} y {keywords_file}...")
        df_movies = pd.read_csv(movies_file, low_memory=False)
        df_credits = pd.read_csv(credits_file, low_memory=False)
        df_keywords = pd.read_csv(keywords_file, low_memory=False)

        # Procesar y limpiar datos
        df_movies = df_movies.dropna(subset=["title", "overview", "id", "vote_average", "genres"])
        df_movies["id"] = df_movies["id"].astype(str)
        df_credits["id"] = df_credits["id"].astype(str)
        df_keywords["id"] = df_keywords["id"].astype(str)

        def extract_top_actors(cast_string):
            try:
                cast = eval(cast_string)
                return [member["name"] for member in cast[:5]]
            except (SyntaxError, TypeError):
                return []

        def parse_keywords(keywords_string):
            try:
                keywords = eval(keywords_string)
                return [keyword["name"] for keyword in keywords]
            except (SyntaxError, TypeError):
                return []

        def parse_genres(genre_string):
            try:
                genres = json.loads(genre_string.replace("'", '"'))
                return [genre["name"] for genre in genres if "name" in genre]
            except (json.JSONDecodeError, TypeError):
                return []

        df_credits["top_actors"] = df_credits["cast"].apply(extract_top_actors)
        df_keywords["keywords_list"] = df_keywords["keywords"].apply(parse_keywords)
        df_movies["genres_list"] = df_movies["genres"].apply(parse_genres)

        df_movies = df_movies.merge(df_credits[["id", "top_actors"]], on="id", how="left")
        df_movies = df_movies.merge(df_keywords[["id", "keywords_list"]], on="id", how="left")

        for _, row in df_movies.iterrows():
            movie_id = int(row["id"]) if row["id"].isdigit() else None
            if movie_id is None:
                continue

            self.movies.append({
                "id": movie_id,
                "title": row["title"],
                "overview": row["overview"],
                "genres": row["genres_list"],
                "actors": row["top_actors"] if isinstance(row["top_actors"], list) else [],
                "keywords": row["keywords_list"] if isinstance(row["keywords_list"], list) else [],
                "rating": round(float(row["vote_average"]), 2) if row["vote_average"] > 0 else "N/A"
            })

        print(f"Total de películas cargadas: {len(self.movies)}")

    def generate_answer(self, query: str):
        """
        Genera una respuesta en base a la consulta del usuario usando RAG, considerando entidades detectadas.
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

        filtered_response = self._format_response(full_response)
        print(f"Respuesta filtrada: {filtered_response}")
        return filtered_response

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
            f"Please respond with the following format for each movie:\n"
            f"Title: <movie title>\nDescription: <brief description>\nRating: <rating or N/A>\n\nResponse:"
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

    def _format_response(self, response_text):
        sections = response_text.split("---")
        movies = []

        for section in sections:
            lines = section.strip().split("\n")
            movie = {}
            for line in lines:
                if line.startswith("Title:"):
                    movie["title"] = line.split("Title:", 1)[1].strip()
                elif line.startswith("Overview:"):
                    movie["overview"] = line.split("Overview:", 1)[1].strip()
                elif line.startswith("Genres:"):
                    movie["genres"] = line.split("Genres:", 1)[1].strip()
                elif line.startswith("Rating:"):
                    movie["rating"] = line.split("Rating:", 1)[1].strip()
            if movie:
                movies.append(movie)

        return movies
