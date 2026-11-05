from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from agent import Agent

app = Flask(__name__)
CORS(app)

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
        answer = agent.ask_corpus(prompt.strip())
    except Exception as error:
        app.logger.exception("Chat request failed")
        return jsonify({"error": str(error)}), 500

    if isinstance(answer, dict):
        response_data = {**answer}
        response_data.setdefault("answer", None)
    else:
        response_data = {"answer": answer}

    return jsonify(response_data)



@app.post("/index")
def index_corpus():
    try:
        report = agent.index_folder(
            r"C:\Temp\corpus"
        )

        return jsonify(report)

    except Exception as error:
        app.logger.exception("PDF indexing request failed")
        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
