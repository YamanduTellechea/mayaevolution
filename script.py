import torch

# Verificar si CUDA está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando el dispositivo: {device}")

# Crear un tensor en la CPU
a = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f"Tensor original en la CPU: {a}")

# Mover el tensor a la GPU
a = a.to(device)
print(f"Tensor en la GPU: {a}")

# Realizar una operación en la GPU
b = a * 2
print(f"Resultado después de la operación en la GPU: {b}")

# Volver a la CPU para verificar el resultado
b = b.to("cpu")
print(f"Resultado movido de vuelta a la CPU: {b}")
