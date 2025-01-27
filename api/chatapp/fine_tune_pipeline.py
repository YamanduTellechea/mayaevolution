from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import os


def fine_tune_model(
    base_model_name="microsoft/phi-4",
    train_file="dataset/movies_data_tokenized_phi-4.jsonl",
    output_dir="fine_tuned_model_phi4",
    num_train_epochs=3,
    batch_size=1,  # Reducido para manejar limitaciones de memoria
    learning_rate=5e-5,
    gradient_accumulation_steps=8,  # Mantener tamaño de lote efectivo
    max_grad_norm=1.0,
):
    # Configurar el dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando el dispositivo: {device}")

    # Configuración de cuantización con offload de 32 bits en la CPU
    print("Cargando modelo y tokenizer...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,  # Offload de FP32 en la CPU
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Asignar automáticamente entre CPU y GPU
        torch_dtype=torch.float16,  # Menor precisión para ahorrar memoria
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Carga del dataset tokenizado
    print(f"Cargando dataset desde {train_file}...")
    dataset = load_dataset("json", data_files={"train": train_file})

    def collate_fn(batch):
        # Collate para datos ya tokenizados
        input_ids = pad_sequence(
            [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [torch.tensor(item["labels"], dtype=torch.long) for item in batch],
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device),
        }

    train_dataloader = DataLoader(
        dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Proceso de fine-tuning
    print("Iniciando el proceso de fine-tuning...")
    model.train()
    for epoch in range(num_train_epochs):
        print(f"Inicio de la época {epoch + 1}/{num_train_epochs}")
        total_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            if (step + 1) % 100 == 0:
                print(f"Step {step + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Época {epoch + 1} finalizada. Pérdida promedio: {avg_loss:.4f}")

    # Guardar el modelo fine-tuneado
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuning completado. Modelo guardado en: {output_dir}")


# Llamar a la función con el archivo JSONL tokenizado
fine_tune_model(train_file="dataset/movies_data_tokenized_phi-4.jsonl", output_dir="./fine_tuned_model_phi4")
