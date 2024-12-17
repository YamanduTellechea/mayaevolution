from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_response(prompt, model_name="bigscience/bloom-560m", max_length=100, temperature=0.7):
    """
    Genera una respuesta a partir de un prompt usando el modelo especificado.
    """
    # Verificar si CUDA está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando el dispositivo: {device}")

    # Cargar el tokenizer y el modelo
    print(f"Cargando el modelo {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Tokenizar el prompt
    print("Tokenizando el prompt...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generar texto
    print("Generando texto...")
    output = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature
    )

    # Decodificar y retornar la respuesta generada
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Definir el prompt inicial
    prompt = "Tell me about the moon landing."
    
    # Generar respuesta
    response = generate_response(prompt)
    print("\nRespuesta generada por el modelo:\n")
    print(response)
    print("hola")
