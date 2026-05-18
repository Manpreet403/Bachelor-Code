# grundlegende Konfiguration für das Experiment
# hier wird festgelegt, dass es sich um einen Zero-Shot Lauf handelt,
# welcher Datensatz verwendet wird und wie die Ergebnisdateien heißen

MODEL_TYPE = "zero"
TEST_FILE = "/Testset.csv"
OUTPUT_FILE = "results_run_zs"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# benötigte Bibliotheken laden
# torch für das Modell, pandas für den Datensatz,
# json für die Verarbeitung der Modellantworten

import torch
import json
import re
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# fixer Seed sorgt dafür, dass Ergebnisse reproduzierbar bleiben
torch.manual_seed(42)

# Jira-Prioritäten
PRIORITIES = ["Blocker", "Critical", "Major", "Minor", "Trivial"]

# Bewertungen der Kriterien
RATINGS = ["Low", "Medium", "High"]

# Funktion zum Erstellen des Prompts für das Sprachmodell
# Summary und Description eines Issues werden hier eingefügt
def build_prompt(Summary, Description):
    return f"""
You are an experienced IT Project Manager.

Your task is to assign exactly ONE Jira Priority to the following issue.

Before deciding the Priority you MUST evaluate the following five criteria.

Each criterion MUST be rated using ONLY one of these values:
Low, Medium, High.

You MUST fill ALL fields.

1. Business Value:
Does the issue provide clear business or user value?

2. Urgency:
Does the issue indicate time pressure (e.g., blocking work, deadlines, urgent problems)?

3. Impact:
How many users, systems, or core workflows are affected?

4. Risk if not implemented:
Are there negative consequences such as failures, security risks, or process breakdowns?

5. Estimated Effort:
Does the issue appear simple or complex to implement based on the Description?

After evaluating the criteria you MUST determine the final Jira Priority.

Allowed priorities:
Blocker, Critical, Major, Minor, Trivial.

Return ONLY raw JSON.
Do NOT include markdown, explanations, or code blocks.

Use EXACTLY this structure and fill EVERY field:

{{
  "Priority": "Blocker | Critical | Major | Minor | Trivial",
  "criteria_assessment": {{
      "business_value": "Low | Medium | High",
      "urgency": "Low | Medium | High",
      "impact": "Low | Medium | High",
      "risk": "Low | Medium | High",
      "estimated_effort": "Low | Medium | High"
  }},
  "reasoning": "Explain briefly why this Priority was chosen."
}}

Summary:
{Summary}

Description:
{Description}
"""

# Modellantwort kann manchmal zusätzlichen Text enthalten
# diese Funktion versucht nur den JSON-Teil herauszufiltern
def extract_json(text):

    # Markdown-Codeblöcke entfernen
    text = text.replace("```json", "").replace("```", "")

    # Bereich zwischen erster und letzter geschweifter Klammer extrahieren
    start = text.find("{")
    end = text.rfind("}") + 1

    try:
        parsed = json.loads(text[start:end])
        return parsed

    # falls das Parsen fehlschlägt wird ein Fehlerobjekt zurückgegeben
    except Exception:
        return {
            "Priority": "ERROR",
            "criteria_assessment": {},
            "reasoning": "PARSE_ERROR"
        }

# stellt sicher, dass nur erlaubte Prioritäten gespeichert werden
def normalize_Priority(value):
    if value in PRIORITIES:
        return value
    return "ERROR"

# stellt sicher, dass nur erlaubte Bewertungen gespeichert werden
def normalize_rating(value):
    if value in RATINGS:
        return value
    return "ERROR"

# Konfiguration der 4-Bit Quantisierung
# damit kann das 7B Modell mit deutlich weniger GPU-Speicher geladen werden
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Sprachmodell laden
# device_map="auto" verteilt das Modell automatisch auf verfügbare Hardware
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb,
    device_map="auto",
    torch_dtype=torch.float16
)

# Modell wird nur für Inferenz genutzt
model.eval()

# Testdatensatz laden
df = pd.read_csv(TEST_FILE, sep=";")

# das Experiment wird fünfmal ausgeführt
# dadurch kann geprüft werden, ob das Modell stabile Ergebnisse liefert
for run in range(1, 6):

    print("\n=============================")
    print(f"START RUN {run} (ZERO-SHOT)")
    print("=============================\n")

    results = []

    # alle Issues aus dem Testdatensatz werden nacheinander verarbeitet
    for idx, row in df.iterrows():

        # Prompt aus Summary und Description erzeugen
        prompt = build_prompt(row["Summary"], row["Description"])

        # Prompt in das Chatformat des Modells umwandeln
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenisierung der Eingabe
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        # Startzeit für Inferenzmessung
        start = time.time()

        # Modell generiert eine Antwort
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        # benötigte Zeit berechnen
        duration = round(time.time() - start, 3)

        # nur die neu generierten Tokens extrahieren
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

        # Tokens wieder in Text umwandeln
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # JSON aus der Modellantwort extrahieren
        parsed = extract_json(decoded)
        criteria = parsed.get("criteria_assessment", {}) or {}

        # Ergebnisse sammeln
        results.append({
            "Key": row["Key"],
            "Summary": row["Summary"],
            "Description": row["Description"],
            "original_Priority": row["Priority"],
            "predicted_Priority": normalize_Priority(parsed.get("Priority")),
            "business_value": normalize_rating(criteria.get("business_value")),
            "urgency": normalize_rating(criteria.get("urgency")),
            "impact": normalize_rating(criteria.get("impact")),
            "risk": normalize_rating(criteria.get("risk")),
            "estimated_effort": normalize_rating(criteria.get("estimated_effort")),
            "reasoning": parsed.get("reasoning", ""),
            "inference_time": duration
        })

        print(f"Run {run}: {idx + 1}/{len(df)} processed")

    # Ergebnisse in ein DataFrame überführen
    results_df = pd.DataFrame(results)

    # Datei pro Durchlauf speichern
    file_name = f"{OUTPUT_FILE}_{run}.csv"
    results_df.to_csv(file_name, index=False, encoding="utf-8")

    print("\nResults saved to:", file_name)