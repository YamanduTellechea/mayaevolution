import faiss
import numpy as np
import csv
from sentence_transformers import SentenceTransformer

# Ruta del archivo .tsv (ajusta esta ruta según tu dataset)
DATASET_PATH = "dataset/movie_lines.tsv"  # Cambia esta ruta al archivo .tsv
INDEX_PATH = "api/chatapp/embeddings/faiss_index.bin"
MODEL_NAME = "all-MiniLM-L6-v2"  # Modelo ligero para embeddings

def load_corpus(filepath):
    """
    Carga las líneas de diálogo desde un archivo .tsv.
    """
    corpus = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            # Suponemos que las líneas de diálogo están en la segunda columna
            if len(row) > 1:  # Asegúrate de que haya una columna con datos
                corpus.append(row[1].strip())  # Ajusta el índice de columna según sea necesario
    return corpus

def generate_embeddings(corpus, model_name):
    """
    Genera embeddings para cada línea del corpus.
    """
    print(f"Cargando modelo {model_name} para generar embeddings...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def build_faiss_index(embeddings, d):
    """
    Construye el índice FAISS.
    """
    print("Creando índice FAISS...")
    index = faiss.IndexFlatL2(d)  # L2 es la distancia euclidiana
    index.add(embeddings)
    return index

def save_faiss_index(index, path):
    """
    Guarda el índice FAISS en un archivo.
    """
    print(f"Guardando índice FAISS en {path}...")
    faiss.write_index(index, path)

def main():
    print("Cargando el corpus desde el archivo .tsv...")
    corpus = load_corpus(DATASET_PATH)
    print(f"Corpus cargado con {len(corpus)} líneas de diálogo.")

    print("Generando embeddings...")
    embeddings = generate_embeddings(corpus, MODEL_NAME)

    print("Construyendo el índice FAISS...")
    d = embeddings.shape[1]  # Dimensión de los embeddings
    index = build_faiss_index(embeddings, d)

    save_faiss_index(index, INDEX_PATH)
    print("¡Índice FAISS generado con éxito!")

if __name__ == "__main__":
    main()
