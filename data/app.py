from flask import Flask, render_template, request
model = pickle.load(open("models/moodforge_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
# home route
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    response = None

    if request.method == "POST":
        user_input = request.form["text"]

        # transform input
        vectorized_input = vectorizer.transform([user_input])

        # predict mood
        prediction = model.predict(vectorized_input)[0]

        # mood-based responses
        if prediction == "sad":
            response = "hey… i’m here with you 🫂 it’s okay to feel this way"
        elif prediction == "happy":
            response = "aww i love this energy 😭✨ stay like this"
        elif prediction == "angry":
            response = "hey… breathe a little, don’t let it consume you 🌿"
        elif prediction == "tired":
            response = "you’ve done enough today… rest a bit 💤"
        else:
            response = "i’m listening… tell me more 💭"

    return render_template("index.html", prediction=prediction, response=response)
# run app
if __name__ == "__main__":
    app.run(debug=True)