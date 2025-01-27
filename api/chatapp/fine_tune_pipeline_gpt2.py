from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import os


def fine_tune_model(
    base_model_name="gpt2",
    train_file="dataset/movies_data_tokenized_gpt2.jsonl",  # Archivo JSONL ya tokenizado
    output_dir="fine_tuned_model_gpt2",
    num_train_epochs=3,
    batch_size=2,
    learning_rate=5e-5,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando el dispositivo: {device}")

    print("Cargando modelo y tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)

    # Añadir token especial si no existe en el tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Cargando dataset desde {train_file}...")
    dataset = load_dataset("json", data_files={"train": train_file})

    def collate_fn(batch):
        input_ids = pad_sequence(
            [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        labels = input_ids.clone()  # En modelos causales, las etiquetas son iguales a input_ids
        attention_mask = pad_sequence(
            [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch],
            batch_first=True,
            padding_value=0,
        )
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device),
        }

    # Usa directamente los datos del archivo tokenizado
    train_dataloader = DataLoader(
        dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)

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

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuning completado. Modelo guardado en: {output_dir}")


# Llamar a la función con el archivo JSONL tokenizado
fine_tune_model()
