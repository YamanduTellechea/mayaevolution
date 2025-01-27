

import os
import json
import pandas as pd
import random

class MovieDataProcessor:
    def __init__(self, dataset_path, max_movies=50, instructions_per_movie=5):
        """
        Procesador para cargar datos de películas desde CSV y generar un archivo JSONL.
        """
        self.dataset_path = dataset_path
        self.movies = []
        self.max_movies = max_movies
        self.instructions_per_movie = instructions_per_movie

    def _load_movies(self):
        """
        Carga películas de movies_metadata.csv, credits.csv y keywords.csv con géneros, actores y palabras clave procesados.
        """
        movies_file = os.path.join(self.dataset_path, "movies_metadata.csv")
        credits_file = os.path.join(self.dataset_path, "credits.csv")
        keywords_file = os.path.join(self.dataset_path, "keywords.csv")

        # Verifica que los archivos existen
        if not all(os.path.exists(f) for f in [movies_file, credits_file, keywords_file]):
            raise FileNotFoundError("Uno o más archivos de dataset no encontrados.")

        print(f"Cargando datos desde {movies_file}, {credits_file} y {keywords_file}...")
        df_movies = pd.read_csv(movies_file, low_memory=False)
        df_credits = pd.read_csv(credits_file, low_memory=False)
        df_keywords = pd.read_csv(keywords_file, low_memory=False)

        # Filtrar y limpiar datos esenciales
        df_movies = df_movies.dropna(subset=["title", "overview", "id", "vote_average", "genres"])
        df_movies["id"] = df_movies["id"].astype(str)
        df_credits["id"] = df_credits["id"].astype(str)
        df_keywords["id"] = df_keywords["id"].astype(str)

        # Funciones para procesar datos específicos
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

        # Aplicar las funciones de limpieza
        df_credits["top_actors"] = df_credits["cast"].apply(extract_top_actors)
        df_keywords["keywords_list"] = df_keywords["keywords"].apply(parse_keywords)
        df_movies["genres_list"] = df_movies["genres"].apply(parse_genres)

        # Unir los datos
        df_movies = df_movies.merge(df_credits[["id", "top_actors"]], on="id", how="left")
        df_movies = df_movies.merge(df_keywords[["id", "keywords_list"]], on="id", how="left")

        # Guardar en self.movies y limitar a max_movies
        for _, row in df_movies.iterrows():
            if len(self.movies) >= self.max_movies:
                break
            movie = {
                "title": row["title"],
                "overview": row["overview"],
                "genres": row["genres_list"],
                "actors": row["top_actors"] if isinstance(row["top_actors"], list) else [],
                "keywords": row["keywords_list"] if isinstance(row["keywords_list"], list) else [],
                "rating": round(float(row["vote_average"]), 2) if row["vote_average"] > 0 else "N/A"
            }
            self.movies.append(movie)

        print(f"Total de películas cargadas: {len(self.movies)}")

    def generate_jsonl(self, output_file="movies_data.jsonl"):
        """
        Genera un archivo JSONL desde las películas cargadas.
        """
        print(f"Generando archivo JSONL en {output_file}...")
        instructions = [
            "Recommend me a movie about {keywords}.",
            "What is a good {genres} movie?",
            "Can you suggest a movie starring {actors}?",
            "Which movie would you recommend for fans of {keywords}?",
            "Tell me about a {genres} film with {actors}."
        ]

        with open(output_file, "w", encoding="utf-8") as f:
            for movie in self.movies:
                for _ in range(self.instructions_per_movie):
                    # Seleccionar una instrucción aleatoria
                    instruction_template = random.choice(instructions)
                    keywords = ", ".join(movie["keywords"][:3]) if movie["keywords"] else "something interesting"
                    genres = ", ".join(movie["genres"][:2]) if movie["genres"] else "a specific genre"
                    actors = ", ".join(movie["actors"][:2]) if movie["actors"] else "some famous actors"

                    instruction = instruction_template.format(
                        keywords=keywords,
                        genres=genres,
                        actors=actors
                    )

                    # Construir el texto de "response"
                    response = (
                        f"This movie, '{movie['title']}', is a {', '.join(movie['genres'])} film "
                        f"starring {', '.join(movie['actors'][:3])}. It is rated {movie['rating']} out of 10. "
                        f"Based on the overview, it seems like a great pick for fans of {', '.join(movie['keywords'][:5])}."
                    )

                    entry = {
                        "instruction": instruction,
                        "response": response
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Archivo JSONL generado exitosamente en {output_file}.")

if __name__ == "__main__":
    dataset_path = ""  # Cambia esto a la ruta de tus CSVs
    output_file = "movies_data_50_multiple_entries.jsonl"

    processor = MovieDataProcessor(dataset_path, max_movies=50, instructions_per_movie=5)
    processor._load_movies()
    processor.generate_jsonl(output_file)
