# Basismodell für das Fine-Tuning
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Pfad zum vorbereiteten Trainingsdatensatz
TRAIN_FILE = "/Trainingsset.csv"

# Verzeichnis zum Speichern des trainierten LoRA-Modells
OUTPUT_DIR = "lora_model"

#Jira-Prioritätsstufen
PRIORITIES = ["Blocker","Critical","Major","Minor","Trivial"]

# Random Seed für reproduzierbare Trainingsläufe
RANDOM_SEED = 42

# maximale Tokenlänge für Eingaben
MAX_LENGTH = 512

# benötigte Bibliotheken importieren
import torch
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

from trl import SFTTrainer

# Seeds setzen, damit Ergebnisse reproduzierbar bleiben
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Trainingsdatensatz laden
df = pd.read_csv(
    TRAIN_FILE,
    sep=";",
    engine="python",
    on_bad_lines="skip"
)

# nur gültige Prioritäten behalten
df = df[df["Priority"].isin(PRIORITIES)]

# fehlende Texte durch leere Strings ersetzen
df["Summary"] = df["Summary"].fillna("").astype(str)
df["Description"] = df["Description"].fillna("").astype(str)

# DataFrame in HuggingFace Dataset umwandeln
dataset = Dataset.from_pandas(df)

# Datensatz in Trainings- und Validierungsdaten aufteilen
dataset = dataset.train_test_split(test_size=0.1, seed=RANDOM_SEED)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# kurze Übersicht über den Datensatz ausgeben
print("Train size:",len(train_dataset))
print("Eval size:",len(eval_dataset))

print("\nPriority distribution:")
print(df["Priority"].value_counts())

# Tokenizer des Basismodells laden
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# EOS Token als Padding Token verwenden
tokenizer.pad_token = tokenizer.eos_token

# maximale Tokenlänge festlegen
tokenizer.model_max_length = MAX_LENGTH

# Promptstruktur für das Training definieren
# das Modell soll lernen:
# Issue Beschreibung → passende Priorität
def build_prompt(example):

    return {
        "text":f"""
Issue Summary:
{example["Summary"]}

Issue Description:
{example["Description"]}

Priority:
{example["Priority"]}
"""
    }

# Promptstruktur auf Trainingsdaten anwenden
train_dataset = train_dataset.map(build_prompt, remove_columns=train_dataset.column_names)

# Promptstruktur auf Evaluationsdaten anwenden
eval_dataset = eval_dataset.map(build_prompt, remove_columns=eval_dataset.column_names)

# Konfiguration der 4-Bit Quantisierung (QLoRA)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Basismodell laden
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    torch_dtype=torch.float16
)

# Modell für Low-Bit Training vorbereiten
model = prepare_model_for_kbit_training(model)

# Cache deaktivieren (notwendig für Training)
model.config.use_cache = False

# LoRA Adapter konfigurieren
lora_config = LoraConfig(

    # Rank der LoRA Matrizen
    r=16,

    # Skalierungsfaktor
    lora_alpha=32,

    # Attention Layer, die angepasst werden
    target_modules=["q_proj","k_proj","v_proj","o_proj"],

    # Dropout zur Stabilisierung
    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)

# LoRA Adapter in das Modell integrieren
model = get_peft_model(model,lora_config)

# anzeigen wie viele Parameter trainiert werden
model.print_trainable_parameters()

# Trainingsparameter definieren
training_args = TrainingArguments(

    # Speicherort des Modells
    output_dir=OUTPUT_DIR,

    # Anzahl der Trainingsepochen
    num_train_epochs=6,

    # Batchgrößen
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,

    # simuliert größere Batchgrößen
    gradient_accumulation_steps=4,

    # Lernrate
    learning_rate=2e-5,

    # Warmup Phase
    warmup_steps=100,

    # Loggingintervall
    logging_steps=10,

    # Evaluation nach jeder Epoche
    eval_strategy="epoch",

    # Modell nach jeder Epoche speichern
    save_strategy="epoch",

    # bestes Modell am Ende laden
    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    # Mixed Precision Training
    bf16=True,

    # reduziert Speicherbedarf
    gradient_checkpointing=True,

    seed=RANDOM_SEED,

    report_to="none"
)

# Trainer initialisieren
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    formatting_func=lambda x: x["text"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Training starten
trainer.train()

# trainiertes LoRA Modell speichern
trainer.model.save_pretrained(OUTPUT_DIR)

# Tokenizer ebenfalls speichern
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training finished.")
print("Model saved to:",OUTPUT_DIR)

# Trainingslogs ausgeben
for log in trainer.state.log_history:

    if "loss" in log:
        print(f"Step {log.get('step')} - Loss: {log['loss']}")

    if "eval_loss" in log:
        print(f"Step {log.get('step')} - Eval Loss: {log['eval_loss']}")

# Losskurve erstellen
logs = trainer.state.log_history

train_steps=[]
train_losses=[]

eval_steps=[]
eval_losses=[]

for log in logs:

    if "loss" in log and "step" in log:
        train_steps.append(log["step"])
        train_losses.append(log["loss"])

    if "eval_loss" in log and "step" in log:
        eval_steps.append(log["step"])
        eval_losses.append(log["eval_loss"])

plt.figure(figsize=(8,5))

if train_steps:
    plt.plot(train_steps,train_losses,label="train_loss")

if eval_steps:
    plt.plot(eval_steps,eval_losses,label="eval_loss")

plt.xlabel("Training Steps")
plt.ylabel("Loss")

plt.title("LoRA Training Loss Curve")

plt.legend()

plt.tight_layout()

plt.savefig("training_loss_curve.png")

plt.show()