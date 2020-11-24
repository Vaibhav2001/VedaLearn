from flask import *
from nlp import generate
# initialize
app = Flask(__name__)

# home-page
@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        start = request.form["start"]
        return start
        generated_txt = generate(start)
        return render_template("index.html", txt=generated_txt)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
