import os
import csv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np


class RAGChatbot:

    def __init__(self, dataset_path, embedding_file="embeddings.npz", embedding_model_name="all-MiniLM-L6-v2"):
        """
        Inicialización del chatbot RAG con soporte para embeddings precalculados.
        """
        print(f"Cargando modelo de embeddings {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name, device="cuda")

        # Cargar dataset
        print(f"Cargando dataset desde {dataset_path}...")
        # Inicialización de estructuras de datos
        self.lines = {}  # Mapeo lineID -> texto
        self.line_to_movie = {}  # Mapeo lineID -> movieID
        self.movies = {}  # Mapeo movieID -> movie_title
        self._load_dataset(dataset_path)

        # Cargar o generar embeddings
        if os.path.exists(embedding_file):            
            self.load_embeddings(embedding_file)
        else:
            print("Generando embeddings para todas las líneas del dataset...")
            self.embeddings = self._generate_embeddings()
            self.save_embeddings(embedding_file)

        # Cargar modelo de generación
        print("Cargando modelo de generación (mistralai/Mistral-7B-Instruct-v0.3)...")
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            torch_dtype=torch.float16
        ).to("cuda")

    def _load_dataset(self, dataset_path):
        """Carga el dataset desde los archivos TSV."""
        lines_path = f"{dataset_path}/movie_lines.tsv"
        movies_path = f"{dataset_path}/movie_titles_metadata.tsv"

        self._load_lines(lines_path)
        self._load_movies(movies_path)

        print(f"Total de líneas cargadas: {len(self.lines)}")
        print(f"Total de líneas con películas asociadas: {len(self.line_to_movie)}")
        print(f"Total de películas cargadas: {len(self.movies)}")

    def _load_lines(self, lines_path):
        """Carga las líneas de diálogo desde movie_lines.tsv."""
        print(f"Cargando líneas desde {lines_path}...")
        with open(lines_path, "r", encoding="iso-8859-1") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 5:  # Asegúrate de que haya suficientes columnas
                    line_id = row[0].strip()  # ID de la línea
                    movie_id = row[2].strip()  # ID de la película
                    text = row[4].strip()  # Texto de la línea

                    # Verificar si los IDs están presentes y son válidos
                    if not line_id or not movie_id:
                        print(f"Saltando línea con datos incompletos: line_id={line_id}, movie_id={movie_id}")
                        continue

                    # Guardar el texto de la línea
                    self.lines[line_id] = text

                    # Mapear la línea al ID de la película
                    if movie_id:
                        self.line_to_movie[line_id] = movie_id
                      

        print(f"Líneas cargadas: {len(self.lines)}")
        print(f"Líneas con películas: {len(self.line_to_movie)}")


    def _load_movies(self, movies_path):
        """Carga los títulos de las películas desde movie_titles_metadata.txt."""
        print(f"Cargando películas desde {movies_path}...")
        with open(movies_path, "r", encoding="iso-8859-1") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    movie_id = row[0].strip()
                    movie_title = row[1].strip()
                    self.movies[movie_id] = movie_title

    def generate_answer(self, query: str):
        """
        Genera una respuesta basada en el contexto más similar al query.
        """
        print(f"Generando embeddings para la consulta: {query}")

        # Generar embeddings para la consulta
        query_emb = self.embed(query)

        # Calcular similitud con todas las líneas
        similarities = torch.nn.functional.cosine_similarity(query_emb, self.embeddings, dim=-1)
        top_indices = torch.topk(similarities, k=5).indices.cpu().numpy()

        # Recuperar contextos relevantes con detalles de película
        retrieved_contexts = self.get_contexts(top_indices)

        # Construir el prompt
        prompt = self.build_prompt(query, retrieved_contexts)
        print(f"Prompt construido (solo para modelo):\n{prompt}")

        # Generar la respuesta usando el modelo de lenguaje
        print("Generando respuesta con el modelo de lenguaje...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        # output = self.model.generate(
        #     **inputs,
        #     max_length=200,          # Aumenta la longitud máxima para evitar cortes
        #     do_sample=True,          # Activa muestreo
        #     temperature=0.7,         # Ajusta la creatividad del modelo (valor entre 0.1 y 1.0)
        #     top_p=0.9,               # Nucleus sampling (ajusta para evitar palabras irrelevantes)
        #     repetition_penalty=1.2,  # Penaliza repeticiones
        #     pad_token_id=self.tokenizer.eos_token_id  # Evita warnings de padding
        # )

        output = self.model.generate(
            **inputs,
            max_length=200,          # Ajusta si necesitas más o menos tokens
            do_sample=False,         # Generación determinista
            repetition_penalty=1.2,  # Penaliza repeticiones
            pad_token_id=self.tokenizer.eos_token_id
        )

        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return self._filter_generated_answer(answer, query)

    def get_contexts(self, indices):
        """
        Recupera los textos asociados a los índices devueltos por similitud, junto con sus películas correspondientes.
        """
        print(f"Recuperando contextos para índices: {indices}")
        contexts = []

        for i in indices:
            if 0 <= i < len(self.line_id_list):  # Validación de índice
                line_id = self.line_id_list[i]
                text = self.lines[line_id]
                movie_id = self.line_to_movie.get(line_id, None)
                movie_title = self.movies.get(movie_id, "Película no encontrada")
                print(f"Línea: {line_id}, Película: {movie_id}, Título: {movie_title}, Texto: {text}")
                contexts.append(f"{text} - {movie_title}")
            else:
                contexts.append(f"Texto no encontrado para índice {i}")

        return contexts


    def embed(self, text):
        """Convierte el texto en un embedding utilizando el modelo de embeddings."""
        print("Generando embeddings para el texto...")
        emb = self.embedding_model.encode(text, convert_to_numpy=True).reshape(1, -1)
        return torch.tensor(emb, device="cuda").float()

    def _filter_generated_answer(self, full_response, query):
        """
        Filtra la respuesta generada por el modelo para eliminar contextos y mantener solo la parte relevante.
        """
        if "Assistant:" in full_response:
            return full_response.split("Assistant:")[-1].strip()
        return full_response.strip()
    
    def build_prompt(self, query, contexts):
        """
        Construye el prompt para el modelo de generación usando el query y los contextos recuperados.
        """
        print("Construyendo prompt para el modelo...")
        context_str = "\n".join(contexts)
        return (
            f"The user is asking about the phrase '{query}'. Below are movie lines containing the phrase or similar phrases:\n"
            f"{context_str}\n\n"
            f"Please respond with the relevant movie lines and their corresponding movie titles in the following format:\n"
            f"Phrase - Movie Title\n\n"
            f"For example:\n"
            f"Sayonara baby - Terminator 2\n"
            f"GET BOND OUT OF THERE - Tomorrow Never Dies\n\n"
            f"User Query: {query}\n"
            f"Assistant:"
        )

        
    def _generate_embeddings(self):
        """Genera embeddings para todas las líneas del dataset."""
        print("Generando embeddings para todas las líneas...")
        self.line_id_list = list(self.lines.keys())  # Lista ordenada de IDs de línea
        texts = [self.lines[line_id] for line_id in self.line_id_list]
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
        print("Embeddings generados correctamente.")
        return embeddings

    def load_embeddings(self, file_path):
        """Carga los embeddings y la lista de IDs desde un archivo .npz."""
        print(f"Cargando embeddings desde {file_path}...")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el archivo de embeddings: {file_path}")

        data = np.load(file_path, allow_pickle=True)
        self.embeddings = torch.tensor(data["embeddings"]).to("cuda")
        self.line_id_list = data["line_ids"].tolist()  # Cargar la lista de IDs de línea
        print("Embeddings y line_ids cargados correctamente.")

    def save_embeddings(self, file_path):
        """Guarda los embeddings y la lista de IDs de línea en un archivo .npz."""
        print(f"Guardando embeddings en {file_path}...")
        np.savez(file_path, embeddings=self.embeddings.cpu().numpy(), line_ids=self.line_id_list)
        print("Embeddings guardados correctamente.")




# import os
# import csv
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import numpy as np


# class RAGChatbot:

#     def __init__(self, dataset_path, embedding_file="embeddings.npz", embedding_model_name="all-MiniLM-L6-v2"):
#         """
#         Inicialización del chatbot RAG con soporte para embeddings precalculados.
#         """
#         print(f"Cargando modelo de embeddings {embedding_model_name}...")
#         self.embedding_model = SentenceTransformer(embedding_model_name, device="cuda")

#         # Cargar dataset
#         print(f"Cargando dataset desde {dataset_path}...")
#         # Inicialización de estructuras de datos
#         self.lines = {}  # Mapeo lineID -> texto
#         self.line_to_movie = {}  # Mapeo lineID -> movieID
#         self.movies = {}  # Mapeo movieID -> movie_title
#         self.conversations = []  # Lista de conversaciones
#         self._load_dataset(dataset_path)

#         # Cargar o generar embeddings
#         if os.path.exists(embedding_file):
#             self.load_embeddings(embedding_file)
#         else:
#             print("Generando embeddings para todas las líneas del dataset...")
#             self.embeddings = self._generate_embeddings()
#             self.save_embeddings(embedding_file)

#         # Cargar modelo de generación
#         print("Cargando modelo de generación (bigscience/bloom-560m)...")
#         self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
#         self.model = AutoModelForCausalLM.from_pretrained(
#             "bigscience/bloom-560m",
#             torch_dtype=torch.float16
#         ).to("cuda")


#     def _load_dataset(self, dataset_path):
#         """Carga el dataset desde los archivos TSV."""
#         lines_path = f"{dataset_path}/movie_lines.tsv"
#         conversations_path = f"{dataset_path}/movie_conversations.tsv"
#         movies_path = f"{dataset_path}/movie_titles_metadata.tsv"

#         self._load_lines(lines_path)
#         self._load_conversations(conversations_path)
#         self._load_movies(movies_path)


#     def _load_lines(self, lines_path):
#         """Carga las líneas de diálogo desde movie_lines.tsv."""
#         print(f"Cargando líneas desde {lines_path}...")
#         with open(lines_path, "r", encoding="iso-8859-1") as f:
#             reader = csv.reader(f, delimiter="\t")
#             for row in reader:
#                 if len(row) >= 5:  # Asegúrate de que haya suficientes columnas
#                     line_id = row[0]  # ID de la línea
#                     movie_id = row[2]  # ID de la película
#                     text = row[4]  # Texto de la línea

#                     # Guardar el texto de la línea
#                     self.lines[line_id] = text

#                     # Mapear la línea al ID de la película
#                     if movie_id:
#                         self.line_to_movie[line_id] = movie_id

    
#     def _load_conversations(self, conversations_path):
#         """Carga las conversaciones desde movie_conversations.tsv."""
#         with open(conversations_path, "r", encoding="iso-8859-1") as f:
#             reader = csv.reader(f, delimiter="\t")
#             for row in reader:
#                 if len(row) == 4:
#                     utterance_ids = eval(row[3])  # Convertir la lista de strings
#                     conversation = {
#                         "lines": [self.lines[line_id] for line_id in utterance_ids if line_id in self.lines],
#                         "movies": {self.line_to_movie[line_id] for line_id in utterance_ids if line_id in self.line_to_movie}
#                     }
#                     self.conversations.append(conversation)
    
#     def _load_movies(self, movies_path):
#         """Carga los títulos de las películas desde movie_titles_metadata.txt."""
#         with open(movies_path, "r", encoding="iso-8859-1") as f:
#             reader = csv.reader(f, delimiter="\t")
#             for row in reader:
#                 if len(row) >= 2:
#                     movie_id, movie_title = row[:2]
#                     self.movies[movie_id] = movie_title

#     def _generate_embeddings(self):
#         """Genera embeddings para todas las líneas del dataset."""
#         texts = list(self.lines.values())
#         return self.embedding_model.encode(texts, convert_to_tensor=True)


#     def generate_answer(self, query: str):
#         """
#         Genera una respuesta basada en el contexto más similar al query.
#         """
#         print(f"Generando embeddings para la consulta: {query}")
        
#         # Generar embeddings para la consulta
#         query_emb = self.embed(query)

#         # Calcular similitud con todas las líneas
#         similarities = torch.nn.functional.cosine_similarity(query_emb, self.embeddings, dim=-1)
#         top_indices = torch.topk(similarities, k=5).indices.cpu().numpy()

#         # Recuperar contextos relevantes con detalles de película
#         retrieved_contexts = self.get_contexts(top_indices)

#         # Construir el prompt (solo para uso interno del modelo)
#         prompt = self.build_prompt(query, retrieved_contexts)
#         print(f"Prompt construido (solo para modelo):\n{prompt}")

#         # Generar la respuesta usando el modelo de lenguaje
#         print("Generando respuesta con el modelo de lenguaje...")
#         inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
#         output = self.model.generate(
#             **inputs,
#             max_length=150,        # Limita la longitud de la respuesta
#             do_sample=True,        # Usa muestreo
#             temperature=0.3,       # Reducción de creatividad
#             top_p=0.85,            # Nucleus sampling
#             repetition_penalty=1.3 # Penaliza repeticiones
#         )
#         answer = self.tokenizer.decode(output[0], skip_special_tokens=True)

#         # Filtrar la respuesta para eliminar el contexto y mantener solo lo relevante
#         response = self._filter_generated_answer(answer, query)
#         return response



#     def _filter_generated_answer(self, full_response, query):
#         """
#         Filtra la respuesta generada por el modelo para eliminar contextos y mantener solo la parte relevante.
#         """
#         # Opcional: Busca el texto después de "Assistant:" si aparece
#         if "Assistant:" in full_response:
#             filtered_response = full_response.split("Assistant:")[-1].strip()
#         else:
#             filtered_response = full_response

#         # Elimina repeticiones o ruidos generados
#         lines = filtered_response.split("\n")
#         return "\n".join(dict.fromkeys(lines)).strip()  # Eliminar duplicados



    # def build_prompt(self, query, contexts):
    #     """
    #     Construye el prompt para el modelo de generación usando el query y los contextos recuperados.
    #     """
    #     print("Construyendo prompt para el modelo...")
    #     context_str = "\n".join(contexts)
    #     return (
    #         f"The user is asking about the phrase '{query}'. Below are some relevant lines from movies related to the query:\n"
    #         f"{context_str}\n\n"
    #         f"Please respond in the following format:\n"
    #         f"Example:\n"
    #         f"Sayonara baby - Arnold Schwarzenegger - Terminator 2\n\n"
    #         f"User Query: {query}\n"
    #         f"Assistant:"
    #     )   
    

    
    # def embed(self, text):
    #     """
    #     Convierte el texto en un embedding utilizando el modelo de embeddings.
    #     """
    #     print("Generando embeddings para el texto...")
    #     emb = self.embedding_model.encode(text, convert_to_numpy=True).reshape(1, -1)
    #     return torch.tensor(emb, device="cuda").float()  # Mover a GPU como tensores float
    
    # def get_contexts(self, indices):
    #     """
    #     Recupera los textos asociados a los índices devueltos por similitud, junto con sus películas correspondientes.
    #     """
    #     print(f"Recuperando contextos para índices: {indices}")
    #     contexts = []
    #     line_keys = list(self.lines.keys())  # Lista ordenada de IDs de línea

    #     for i in indices:
    #         line_id = line_keys[i] if 0 <= i < len(line_keys) else None
    #         if line_id:
    #             print(f"Procesando línea {line_id}...")
    #             movie_id = self.line_to_movie.get(line_id, None)
    #             print(f"ID de película encontrado: {movie_id}")
    #             if movie_id in self.movies:
    #                 print(f"Película asociada: {self.movies[movie_id]}")
    #             else:
    #                 print("Película no encontrada en self.movies")
    #         else:
    #             print(f"Índice inválido: {i}")

    #     return contexts
    
    # def save_embeddings(self, file_path):
    #     """
    #     Guarda los embeddings y textos en un archivo .npz.
    #     """
    #     print(f"Guardando embeddings en {file_path}...")
    #     np.savez(file_path, embeddings=self.embeddings.cpu().numpy(), texts=list(self.lines.values()))
    #     print("Embeddings guardados correctamente.")

    # def load_embeddings(self, file_path):
    #     """
    #     Carga los embeddings y textos desde un archivo .npz.
    #     """
    #     print(f"Cargando embeddings desde {file_path}...")
    #     if not os.path.exists(file_path):
    #         raise FileNotFoundError(f"No se encontró el archivo de embeddings: {file_path}")
        
    #     data = np.load(file_path)
    #     self.embeddings = torch.tensor(data["embeddings"]).to("cuda")  # Mover a GPU
    #     self.lines = {i: text for i, text in enumerate(data["texts"])}  # Reconstruir mapeo lineID -> texto
    #     print("Embeddings cargados correctamente.")




