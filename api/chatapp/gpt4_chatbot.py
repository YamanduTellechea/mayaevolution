import openai
import json
# Configurar clave de API directamente en el código

class GPT4Chatbot:
    def __init__(self):
        """
        Inicialización del chatbot basado en GPT-4.
        """
        if not openai.api_key:
            raise ValueError("La clave de API de OpenAI no está configurada correctamente.")
        print("Chatbot GPT-4 inicializado correctamente.")

    def generate_answer(self, query: str):
        """
        Genera una respuesta en formato estructurado usando GPT-4.
        Args:
            query (str): Pregunta realizada por el usuario.
        Returns:
            str: Respuesta generada en formato estructurado.
        """
        # Construir el prompt para GPT-4
        prompt = self._build_prompt(query)

        # Llamar a la API de GPT-4
        response = self._call_gpt4(prompt)

        # Formatear la respuesta
        return self._format_response(response)

    def _build_prompt(self, query: str):
        """
        Construye el prompt para GPT-4.
        Args:
            query (str): Pregunta realizada por el usuario.
        Returns:
            str: Prompt para enviar a GPT-4.
        """
        return (
        f"The user asked the following question about 3 movies recommendation: '{query}'.\n\n"        
        f"Please respond in JSON format as follows:\n\n"
        f"{{\n"
        f"    \"query\": \"<original user query>\",\n"
        f"    \"answer\": [\n"
        f"        {{\n"
        f"            \"title\": \"<movie title>\",\n"
        f"            \"overview\": \"<brief description of the movie>\",\n"
        f"            \"genres\": \"<comma-separated genres>\",\n"
        f"            \"actors\": \"<comma-separated main actors>\",\n"
        f"            \"rating\": \"<rating or N/A>\"\n"
        f"        }}\n"
        f"        // Add additional movies as needed\n"
        f"    ]\n"
        f"}}\n\n"
        f"Ensure the response is a valid JSON object. Do not include additional explanations, comments, or text outside the JSON structure."
    )
       
    def _call_gpt4(self, prompt: str):
        """
        Llama a la API de GPT-4 para obtener una respuesta.
        Args:
            prompt (str): Prompt enviado a GPT-4.
        Returns:
            str: Respuesta generada por GPT-4.
        """
        print("Llamando a GPT-4...")
        try:
            response = openai.chat.completions.create(            
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in movie recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            raw_answer = response.choices[0].message.content.strip()
            
            # Intentar parsear el JSON
            try:
                parsed_answer = json.loads(raw_answer)               

                return parsed_answer['answer']
            except json.JSONDecodeError:
                print("La respuesta no es un JSON válido.")
                return {"error": "La respuesta no es un JSON válido.", "raw_answer": raw_answer}
                     
        except Exception as e:
            print(f"Error al llamar a GPT-4: {e}")
            return "Lo siento, hubo un error al generar la respuesta."

    def _format_response(self, response: str):
        """
        Formatea la respuesta generada por GPT-4 en caso de ser necesario.
        Args:
            response (str): Respuesta original generada por GPT-4.
        Returns:
            str: Respuesta formateada.
        """
        # En este caso, simplemente devolvemos la respuesta generada tal cual,
        # ya que el prompt especifica que la respuesta debe estar en el formato estructurado.
        return response



