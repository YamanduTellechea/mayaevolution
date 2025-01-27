from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Ruta del modelo fine-tuneado
ft_model_path = "chatapp/dataset/fine_tuned_model_phi4"

# Verificar si se puede cargar el modelo fine-tuneado
try:
    print("Cargando el modelo fine-tuneado...")
    tokenizer = AutoTokenizer.from_pretrained(ft_model_path)

    # No usar `.to("cuda")` con modelos cuantizados
    model = AutoModelForCausalLM.from_pretrained(
        ft_model_path,
        device_map="auto",  # Asigna automáticamente dispositivos
        torch_dtype=torch.float16,  # Precisión reducida para ahorrar memoria
    )
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo fine-tuneado: {e}")
    exit()

# Bucle para probar el modelo con consultas del usuario
print("\nModelo listo para probar. Escribe 'exit' para salir.")
while True:
    user_query = input("Usuario: ").strip()
    if user_query.lower() == "exit":
        print("Saliendo del programa.")
        break

    # Construir el prompt para el modelo
    prompt = f"User query: {user_query}\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    # Asegúrate de mover los tensores al dispositivo correcto
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Generar respuesta
    try:
        output = model.generate(
            **inputs,
            max_length=200,  # Longitud máxima de la respuesta
            do_sample=True,  # Sampling para respuestas variadas
            top_p=0.9,       # Nucleus sampling
            temperature=0.7, # Controla la creatividad
            repetition_penalty=1.2,  # Penalización para evitar repeticiones
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        print(f"Chatbot: {response}")
    except Exception as e:
        print(f"Error al generar respuesta: {e}")
