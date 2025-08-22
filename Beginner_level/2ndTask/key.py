from flask import Flask, request, jsonify, render_template
from spellchecker import SpellChecker
from collections import defaultdict, Counter
import re
app = Flask(__name__)
spell = SpellChecker()
bigrams = defaultdict(Counter)
corpus = """
database connectivity and integration are crucial. database management systems help organize data. 
database optimization improves performance and data retrieval.
"""
def train_bigrams(text):
    words = re.findall(r'\b\w+\b', text.lower())
    for i in range(len(words) - 1):
        bigrams[words[i]][words[i + 1]] += 1
train_bigrams(corpus)
def predict_next_words(word, top_n=3):
    word = word.lower()
    if word in bigrams:
        return [w for w, _ in bigrams[word].most_common(top_n)]
    return []
@app.route('/')
def index():
    return render_template("smartboard.html")
@app.route("/autocorrect", methods=["POST"])
def autocorrect():
    data = request.get_json()
    text = data.get("text", "").strip()
    words = text.split()
    if not words:
        return jsonify({"suggestion": "", "predictions": []})
    last_word = words[-1]
    correction = spell.correction(last_word)
    predictions = predict_next_words(correction)
    return jsonify({
        "suggestion": correction if correction != last_word else "",
        "predictions": predictions
    })
if __name__ == "__main__":
    app.run(debug=True)