// ============================================================
// Datenextraktion – Bachelorarbeit
// Ausführung: in mongosh starten mit:
// & "C:\Users\Mani\AppData\Local\Programs\mongosh\mongosh.exe"
// ============================================================

// Schritt 1: MongoDB-Dump wiederherstellen (in CMD ausführen, nicht in mongosh)
// cd "C:\Users\Mani\Documents\BWI Jahr 3\Bachelorarbeit\mongodb-database-tools-windows-x86_64-100.13.0\bin"
// .\mongorestore.exe --gzip --archive="C:\Users\Mani\Documents\BWI Jahr 3\Bachelorarbeit\2025-06-23 The-PublicJiraDataset\ThePublicJiraDataset\3. DataDump\mongodump-JiraReposAnon.archive"

// Schritt 2: MongoDB-Instanz starten (in CMD ausführen)
// net start MongoDB

// Schritt 3: Datenbank auswählen
use JiraReposAnon

// Schritt 4: Verfügbare Collections anzeigen
show collections

// Schritt 5: Testabfrage eines Eintrags
db.JiraEcosystem.findOne()

// Schritt 6: Extraktion – kompatible Issues aus Spring, MariaDB und JiraEcosystem
const fs = require('node:fs');

const collectionsToUse = ["Spring", "MariaDB", "JiraEcosystem"];
const allowedPriorities = ["Blocker", "Critical", "Major", "Minor", "Trivial"];

const outPath = "C:/Users/Mani/Documents/BWI Jahr 3/Bachelorarbeit/all_compatible_with_source.jsonl";

let lines = [];

collectionsToUse.forEach(coll => {
  db.getCollection(coll).find(
    { "fields.Priority.name": { $in: allowedPriorities } },
    { Key: 1, "fields.Summary": 1, "fields.Description": 1, "fields.Priority.name": 1 }
  ).forEach(doc => {
    const row = {
      source: coll,
      Key: doc.Key || "",
      Summary: doc.fields?.Summary || "",
      Description: doc.fields?.Description || "",
      Priority: doc.fields?.Priority?.name || ""
    };
    lines.push(JSON.stringify(row));
  });
});

fs.writeFileSync(outPath, lines.join("\n"), "utf8");

print("JSONL exportiert:", outPath);
print("Gesamt extrahierte Issues:", lines.length);