from pathlib import Path

from flask import Flask, jsonify, request, send_file

from config import get_settings
from agent import Agent

app = Flask(__name__)

agent = Agent()


@app.get("/")
def home():
    return send_file(Path(__file__).resolve().parent / "ollama.html")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object"}), 400

    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({"error": "A non-empty prompt is required"}), 400

    try:
        # Get the evidence package
        evidence_package = agent.ask_corpus(prompt.strip())
        
        # If it's already a structured response (like from OECD analysis), return as-is
        if isinstance(evidence_package, dict) and "route" in evidence_package:
            # This is already a structured response with route information
            return jsonify(evidence_package)
        
        # Otherwise, wrap the simple string answer in a structured format
        # for backward compatibility
        return jsonify({"answer": evidence_package})
        
    except Exception as error:
        app.logger.exception("Chat request failed")
        return jsonify({"error": str(error)}), 500


@app.post("/index")
def index_corpus():
    try:
        report = agent.index_folder(str(get_settings().corpus_dir))

        return jsonify(report)

    except Exception as error:
        app.logger.exception("PDF indexing request failed")
        return jsonify({
            "error": str(error)
        }), 500


def main():
    settings = get_settings()
    app.run(
        host=settings.host,
        port=settings.port,
        debug=settings.debug,
    )


if __name__ == "__main__":
    main()

