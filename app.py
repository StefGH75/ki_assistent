"""
Flask App für den Chatbot, der mit Daten aus zur Verfügung gestellten PDFs arbeitet.
Die Flask App lädt einmalig eine Vektordatenbank (mithilfe von FAISS) aus den PDFs, baut darauf eine RetrievalQA-Chain
mit OpenAi auf und beanwortet dann über ein Post Route Fragen als JSON.

"""

from flask import Flask, request, render_template, jsonify
import json
import shutil
from pathlib import Path
import os
import glob
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

app = Flask(__name__)
qa_chain = None #global um teure Neuinitialisierung bei jeder Anfrage zu vermeiden

def build_manifest(pdf_dir, embedding_model, chunk_size, chunk_overlap):
    """
    Diese Funktion prüft, ob sich PDFs, Chunking-Parameter oder das Embedding-Modell geändert haben.
    Ergebnis: FAISS-Index wird nur neu gebaut, wenn wirklich notwendig, spart Zeit
    """
    files = []
    for p in Path(pdf_dir).glob("*.pdf"):
        stat = p.stat()
        files.append({
            "path": str(p.resolve()),
            "mtime": stat.st_mtime, #Änderungszeit der Datei
            "size": stat.st_size #Dateigröße
        })
    files.sort(key=lambda x: x["path"]) #sorgt für konsistente Reihenfolge
    return {
        "files": files,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }

def setup_bot():
    """
    Initialisiert den Chatbot nur einmal pro Anwendungslauf.
    Lädt eine bestehende FAISS-Datenbank oder erstellt sie neu aus PDF-Dateien im /docs-Verzeichnis.
    Baut anschließend eine Retrieval-QA-Kette mit OpenAI-Modell und System-Prompt auf.
    """
    global qa_chain
    if qa_chain is not None:
        return #wenn schon initialisiert, sofort raus

    # API-Key prüfen:
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY ist nicht gesetzt. Bitte in den Umgebungsvariablen definieren.")

    base_dir = os.path.dirname(__file__)
    pdf_dir = os.path.join(base_dir, "docs")
    index_path = os.path.join(base_dir, "faiss_index")
    manifest_file = os.path.join(index_path, "manifest.json")

    # Embeddings und Splitter-Parameter:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    chunk_size, chunk_overlap = 800, 100

    # aktuelles Manifest erstellen:
    manifest_now = build_manifest(pdf_dir, embedding.model, chunk_size, chunk_overlap)

    # altes Manifest laden, falls vorhanden:
    old_manifest = None
    if os.path.exists(manifest_file):
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                old_manifest = json.load(f)
        except Exception:
            pass

    # entscheiden, ob Index neu gebaut werden muss (Nur wenn PDFs oder Parameter geändert wurden)
    need_rebuild = not (os.path.exists(index_path) and old_manifest == manifest_now)

    if need_rebuild:
        print("Rebuilding FAISS index...")
        shutil.rmtree(index_path, ignore_errors=True)
        os.makedirs(index_path, exist_ok=True)

        # PDFs einlesen und verarbeiten:
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        if not pdf_files:
            raise FileNotFoundError("Keine PDF-Dateien im Ordner /docs gefunden.")

        # alle PDFs mit PyPDFLoader von Langchain einlesen:
        all_docs = []
        for pfad in pdf_files:
            print(f"Loading PDF: {pfad}")
            loader = PyPDFLoader(pfad)
            docs = loader.load() #liest Datei ein und erzeugt pro Seite eine Liste von Document-Objekten (mit page_content und Metadaten)
            all_docs.extend(docs)

        # Text in überlappende Chunks zerteilen:
        #Nutzung recursive: versucht zuerst an Absätzen zu trennen, dann Sätzen, dann Wörtern. Ergebnis ist meist lesbarer Kontext
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap, #800 Zeichen pro Chunk, 100 Zeichen Überlapp
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(all_docs)

        # Neue FAISS-Datenbank erstellen und speichern:
        print("Creating FAISS index...")
        db = FAISS.from_documents(chunks, embedding) #berechnet Embeddings für alle chunks
        db.save_local(index_path)

        # Manifest speichern, beim nächsten Start wird geprüft, ob PDFs/Parameter unverändert sind:
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_now, f, indent=2)

        print("FAISS index created and saved.")
    else:
        print("Loading existing FAISS index...")
        # Vorhandene Vektordatenbank laden:
        db = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
        # allow_dangerous_deserialization=True: damit Metadaten geladen werden können auf True setzen. Achtung: nur setzen, wenn
        # FAISS-Indizes selbst aus vertrauenswürdigen Quellen erzeugt und gespeichert werden
        print("FAISS index loaded.")

    retriever = db.as_retriever() #kapselt Ähnlichkeitsversuche im Vektorstore und wandelt in Retriever-Objekt (für RetrievalQA,
    #das nicht direkt mit FAISS umgehen kann)

    # LLM setzen:
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Systemprompt, der an korrekte Schreibweise erinnert und Kontext einschränkt (Anti-Halluzinations-Maßnahme)
    system_content = (
        "Du bist ein KI-Assistent, der Fragen zu Stefanies beruflichem Werdegang beantwortet. "
        "Die Informationen stammen aus verschiedenen Bewerbungsunterlagen, insbesondere aus der Datei 'lebenslauf_stefanie_datenschutzkonform.pdf' "
        "Die korrekte Schreibweise ihres Namens ist Stefanie (nicht Stephanie). "
        "Für Fragen zu Projekten vor allem die 'Projekte.pdf'-Datei nutzen."
        "Wiederhole diesen Hinweis nicht unnötig und antworte präzise auf Deutsch. "
        "Verwende nur Informationen aus dem bereitgestellten Kontext."
    )
    # Prompt-Vorlage mit Platzhaltern:
    custom_prompt = ChatPromptTemplate.from_messages([
        ("system", system_content),
        ("human", "Frage: {question}\n\nKontext: {context}") # context wird später durch RetrievalQA mit gefunden Text befüllt
    ])

    # Aufbau der RAG-Kette:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, #beantwortet Frage, indem Dokumente zusammen mit Frage in einem Prompt eingesetzt wird
        retriever=retriever, # holt relevante Dokumente aus FAISS Index
        chain_type_kwargs={"prompt": custom_prompt} #Nutzung des Prompts des Nutzers inklusive custom_prompts (Kontext)
    )

#User-Interface:
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html") #rendert Frontend-Template aus der index.html

@app.route("/frage", methods=["POST"])
def frage():
    """
    Nimmt eine Frage als JSON entgegen und gibt die Antwort als JSON aus der RAG-Kette zurück.
    """
    try:
        setup_bot() #stellt sicher, dass Bot einmalig initialisiert ist (FAISS laden/erstellen, Retriever, Kette aufbauen)
        #erster Aufruf kann länger dauern, weil Embeddings berechnet oder Index geladen wird
        frage = request.json.get("frage", "").strip() #erwartet JSON-Body mit Schlüssel "frage" und entfernt Leerzeichen
        if not frage:
            return jsonify({"error": "Keine Frage erhalten"}), 400 #validiert Input und gibt Clientfehler aus
        result = qa_chain.invoke(frage) #führt Retrieval-Kette aus (bei Problemen: {"query": frage})
        return jsonify({"answer": result.get("result", str(result))}) #extrahiert Ergebnis und sorgt dafür, dass es immer str ist
    except Exception as e:
        return jsonify({"error": str(e)}), 500 #fängt alle Fehler ab und gibt 500 zurück

if __name__ == "__main__":
    app.run(debug=True)
