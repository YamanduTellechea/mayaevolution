from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
import os

def fine_tune_model(
    base_model_name="mistralai/Mistral-7B-Instruct-v0.3",
    train_file="movies_data.jsonl",
    output_dir="./fine_tuned_model",
    num_train_epochs=3,
    batch_size=2,
    learning_rate=5e-5,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0
):
    """
    Fine-tuning personalizado del modelo.
    """
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"El archivo {train_file} no existe.")

    print(f"Cargando dataset desde {train_file}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16
    ).to("cuda")

    dataset = load_dataset("json", data_files={"train": train_file})
    def tokenize_function(example):
        input_text = example["instruction"] + "\n" + example["response"]
        tokenized = tokenizer(
            input_text,
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=["instruction", "response"])

    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: {
            "input_ids": torch.tensor([item["input_ids"] for item in x], dtype=torch.long).to("cuda"),
            "attention_mask": torch.tensor([item["attention_mask"] for item in x], dtype=torch.long).to("cuda"),
            "labels": torch.tensor([item["labels"] for item in x], dtype=torch.long).to("cuda"),
        }
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

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
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


fine_tune_model(train_file="movies_data.jsonl", output_dir="./fine_tuned_model")
