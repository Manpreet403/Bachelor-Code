# Modelltyp (LoRA-basierte Modellvariante)
MODEL_TYPE = "lora"

# Pfad zum Testdatensatz
TEST_FILE = "/Testset.csv"

# Präfix für die Ergebnisdateien
OUTPUT_FILE = "results_run_lora"

# verwendetes Basismodell
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Pfad zu den trainierten LoRA-Adaptern
LORA_PATH = "./lora_model/ora_model"

# benötigte Bibliotheken importieren
import torch
import json
import re
import time
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Random Seed setzen für reproduzierbare Ergebnisse
torch.manual_seed(42)

# Prioritätsklassen
PRIORITIES = ["Blocker", "Critical", "Major", "Minor", "Trivial"]

# Bewertungsstufen der Kriterien
RATINGS = ["Low", "Medium", "High"]

# Prompt zur Vorhersage der Jira-Priorität
def build_Priority_prompt(Summary, Description):

    return f"""
Determine the correct Jira issue Priority.

Return ONLY ONE word from this list:
Blocker, Critical, Major, Minor, Trivial.

Issue Summary:
{Summary}

Issue Description:
{Description}

Priority:
"""

# Prompt zur nachträglichen Erklärung der Prioritätsentscheidung
def build_prompt(Summary, Description, Priority):

    return f"""
You are an experienced IT Project Manager.

A Jira Priority has already been assigned to the following issue.

Your task is to evaluate the issue using the five criteria below
and explain why the assigned Priority is appropriate.

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

The assigned Jira Priority is:

{Priority}

Return ONLY raw JSON.
Do NOT include markdown, explanations, or code blocks.

Use EXACTLY this structure and fill EVERY field:

{{
  "Priority": "{Priority}",
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

# JSON aus der Modellantwort extrahieren
def extract_json(text):

    text = text.replace("```json", "").replace("```", "")

    start = text.find("{")
    end = text.rfind("}") + 1

    try:
        parsed = json.loads(text[start:end])
        return parsed
    except Exception:
        return {
            "Priority": "ERROR",
            "criteria_assessment": {},
            "reasoning": "PARSE_ERROR"
        }

# Priorität aus der Modellantwort extrahieren
def extract_Priority(text):

    match = re.search(r"\b(Blocker|Critical|Major|Minor|Trivial)\b", text)

    if match:
        return match.group(1)

    return "ERROR"

# überprüfen ob die Priorität gültig ist
def normalize_Priority(value):

    if value in PRIORITIES:
        return value

    return "ERROR"

# überprüfen ob die Kriterienbewertungen gültig sind
def normalize_rating(value):

    if value in RATINGS:
        return value

    return "ERROR"

# Konfiguration der 4-Bit Quantisierung
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Basismodell laden
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb,
    device_map="auto",
    torch_dtype=torch.float16
)

# trainierte LoRA Adapter laden
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# Modell in Evaluationsmodus setzen
model.eval()

# Testdatensatz laden
df = pd.read_csv(TEST_FILE, sep=";")

# Experiment wird mehrfach durchgeführt
for run in range(1, 6):

    print(f"\nSTART RUN {run} (LORA)\n")

    results = []

    # jedes Issue im Testdatensatz verarbeiten
    for idx, row in df.iterrows():

        # Priorität vorhersagen

        Priority_prompt = build_Priority_prompt(
            row["Summary"],
            row["Description"]
        )

        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": Priority_prompt}],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0,
            do_sample=False
        )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

        decoded_Priority = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )

        predicted_Priority = extract_Priority(decoded_Priority)

        # Kriterien und Begründung generieren

        prompt = build_prompt(
            row["Summary"],
            row["Description"],
            predicted_Priority
        )

        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

        start = time.time()

        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0,
            do_sample=False
        )

        duration = round(time.time() - start, 3)

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

        parsed = extract_json(decoded)
        criteria = parsed.get("criteria_assessment", {}) or {}

        # Ergebnisse speichern
        results.append({

            "Key": row["Key"],
            "Summary": row["Summary"],
            "Description": row["Description"],
            "original_Priority": row["Priority"],
            "predicted_Priority": predicted_Priority,
            "business_value": normalize_rating(criteria.get("business_value")),
            "urgency": normalize_rating(criteria.get("urgency")),
            "impact": normalize_rating(criteria.get("impact")),
            "risk": normalize_rating(criteria.get("risk")),
            "estimated_effort": normalize_rating(criteria.get("estimated_effort")),
            "reasoning": parsed.get("reasoning", ""),
            "inference_time": duration
        })

        print(f"Run {run}: {idx + 1}/{len(df)} processed")

    # Ergebnisse als CSV-Datei speichern
    results_df = pd.DataFrame(results)

    file_name = f"{OUTPUT_FILE}_{run}.csv"

    results_df.to_csv(file_name, index=False, encoding="utf-8")

    print("\nResults saved to:", file_name)
