import os
import json
import pandas as pd

class MovieDataProcessor:
    def __init__(self, dataset_path):
        """
        Procesador para cargar datos de películas desde CSV y generar un archivo JSONL.
        """
        self.dataset_path = dataset_path
        self.movies = []

    def _load_movies(self):
        """
        Carga películas de movies_metadata.csv, credits.csv y keywords.csv con géneros, actores y palabras clave procesados.
        """
        movies_file = os.path.join(self.dataset_path, "movies_metadata.csv")
        credits_file = os.path.join(self.dataset_path, "credits.csv")
        keywords_file = os.path.join(self.dataset_path, "keywords.csv")

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

    def generate_jsonl(self, output_file="movies_data.jsonl"):
        """
        Genera un archivo JSONL desde las películas cargadas.
        """
        print(f"Generando archivo JSONL en {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            for movie in self.movies:
                instruction = (
                    f"Title: {movie['title']}\n"
                    f"Overview: {movie['overview']}\n"
                    f"Genres: {', '.join(movie['genres'])}\n"
                    f"Actors: {', '.join(movie['actors'])}\n"
                    f"Keywords: {', '.join(movie['keywords'])}\n"
                    f"Rating: {movie['rating']}"
                )
                response = "Provide a summary or recommendation based on this information."

                entry = {
                    "instruction": instruction,
                    "response": response
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Archivo JSONL generado exitosamente en {output_file}.")

if __name__ == "__main__":
    dataset_path = ""  # Cambia esto por la ruta donde están los CSV
    output_file = "movies_data.jsonl"   # Nombre del archivo JSONL de salida

    processor = MovieDataProcessor(dataset_path)
    processor._load_movies()
    processor.generate_jsonl(output_file)
