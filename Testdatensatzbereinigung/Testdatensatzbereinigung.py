import re
import pandas as pd

# Testdatenbereinigung:
# Random Seed für reproduzierbare Stichproben
RANDOM_SEED = 42

# Pfad zur zuvor erzeugten JSONL-Datei mit den extrahierten Issues
SOURCE_FILE = r"C:\Users\Mani\Documents\BWI Jahr 3\Bachelorarbeit\Data\SpringMariaJira\all_compatible_with_source.jsonl"

# Zielpfad für den finalen Trainingsdatensatz
TARGET_TESTCASE_FILE = r"C:\Users\Mani\Documents\BWI Jahr 3\Bachelorarbeit\Data\SpringMariaJira\Trainingsset.csv"

# definierte Reihenfolge der Jira-Prioritäten
PRIORITY_ORDER = ["Blocker", "Critical", "Major", "Minor", "Trivial"]

# minimale und maximale Wortanzahl für das Summary-Feld
MIN_WORDS_SUMMARY = 3
MAX_WORDS_SUMMARY = 30

# minimale und maximale Wortanzahl für die Beschreibung
MIN_WORDS_DESC = 30
MAX_WORDS_DESC = 300

# diese Issue-Keys gehören zum späteren Testdatensatz
# sie werden daher bewusst aus dem Trainingsdatensatz ausgeschlossen
EXCLUDED_KEYS = {
    "XD-1005","MDEV-15187","MOD-261","UPM-2562","UPM-2745",
    "AMKT-1079","AMKT-929","AG-624","UPM-2663","SOCIALTW-19",
    "DATACMNS-1161","RCP-566","IDE-978","MDEV-8651","MXS-1112",
    "CONPY-114","MDEV-5156","AC-1262","PL-411","DATACASS-119",
    "STS-3628","UPM-370","DATAREST-407"
}

# HTML Tags entfernen
HTML_TAGS = re.compile(r"<[^>]+>", re.IGNORECASE)

# HTML Entities wie &nbsp; oder &#123;
HTML_ENTITIES = re.compile(r"&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;", re.IGNORECASE)

# URLs entfernen
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

# typische Jira-Markup-Elemente entfernen
JIRA_MARKUP = re.compile(
    r"(\{\{.*?\}\}|\{quote\}|\{code(:[^\}]*)?\}|\{panel(:[^\}]*)?\}|\{noformat\}|\{color(:[^\}]*)?\})",
    re.IGNORECASE | re.DOTALL
)

# Referenzen auf Bilddateien entfernen
FILE_ARTIFACTS = re.compile(r"(\.png|\.jpg|\.jpeg|\.gif|\.bmp|\.svg)", re.IGNORECASE)

# Funktion zur Bereinigung der Textfelder
# entfernt technische Artefakte und vereinheitlicht Whitespace
def clean_text(t):

    # falls der Text leer oder NaN ist
    if pd.isna(t) or t is None:
        return ""

    t = str(t)

    # verschiedene Artefakte entfernen
    t = re.sub(JIRA_MARKUP, " ", t)
    t = re.sub(URL_PATTERN, " ", t)
    t = re.sub(HTML_TAGS, " ", t)
    t = re.sub(HTML_ENTITIES, " ", t)
    t = re.sub(FILE_ARTIFACTS, " ", t)

    # Markdown-Zeichen entfernen
    t = re.sub(r"[*_`]+", "", t)

    # mehrfachen Whitespace auf ein Leerzeichen reduzieren
    t = re.sub(r"\s+", " ", t).strip()

    return t

# JSONL-Datei laden
# je nach Struktur kann pandas entweder direkt lesen oder benötigt das lines=True Flag
try:
    df = pd.read_json(SOURCE_FILE)
except ValueError:
    df = pd.read_json(SOURCE_FILE, lines=True)

# definierte Test-Issues aus dem Datensatz entfernen
if "Key" in df.columns:
    df = df[~df["Key"].isin(EXCLUDED_KEYS)]

# Kopie des DataFrames erstellen, um die Originaldaten unverändert zu lassen
work = df.copy()

# Textfelder bereinigen
work["Summary"] = work["Summary"].apply(clean_text)
work["Description"] = work["Description"].apply(clean_text)

# Funktion zur Berechnung der Wortanzahl
def word_count(s):
    return len(str(s).split())

# Issues mit zu kurzen Texten entfernen
work = work[work["Summary"].apply(word_count) >= MIN_WORDS_SUMMARY]
work = work[work["Description"].apply(word_count) >= MIN_WORDS_DESC]

# nur Issues mit gültigen Prioritäten behalten
work = work[work["Priority"].isin(PRIORITY_ORDER)].copy()

# Verteilung der Prioritäten pro Repository analysieren
counts = (
    work.groupby(["source", "Priority"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=PRIORITY_ORDER, fill_value=0)
)

# kleinstes gemeinsames Minimum bestimmen
# dieses wird später für die Balancierung verwendet
global_min = int(counts.min().min())

# Liste zum Sammeln der balancierten Teilmengen
parts = []

# vorhandene Repositories ermitteln
sources = sorted(work["source"].dropna().unique().tolist())

# für jede Kombination aus Repository und Priorität
# werden gleich viele Issues zufällig ausgewählt
for x in sources:
    for y in PRIORITY_ORDER:

        subset = work[(work["source"] == src) & (work["Priority"] == prio)]

        sampled = subset.sample(n=global_min, random_state=RANDOM_SEED)

        parts.append(sampled)

# alle ausgewählten Teilmengen zu einem Datensatz zusammenführen
trainset = pd.concat(parts, ignore_index=True)

# Datensatz zufällig mischen
trainset = trainset.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

# finalen Trainingsdatensatz als CSV exportieren
trainset.to_csv(
    TARGET_TESTCASE_FILE,
    index=False,
    sep=";",
    encoding="utf-8-sig",
)

print("Trainingsdatensatz gespeichert:", TARGET_TESTCASE_FILE)
