from flask import Flask, render_template, request, redirect



app = Flask(__name__)

db = [
    {
        "jmeno":"Gulas",
        "ingredience":["Paprika", "Maso", "Brmabory"],
        "postup":"vsechno to nejak zamichame",
    },
    {
        "jmeno":"rohlik s maslem",
        "ingredience":["Rohlik"],
        "postup":"vis ne",
    }
]

@app.route("/")
@app.route("/home")
def home_page():
    return render_template("index.html")


@app.route("/pozdrav")
def pozdrav_nastevnika():
    return "<p>Ahoj navstevniku"

@app.route("/recepty")
def zobraz_recepty():
    return render_template("recepty.html", recepty=db)

@app.route("/formular", methods=["GET","POST"])
def zasli_recept():
    if request.method == "GET":
        return render_template("formular.html")
    else:
        jmeno = request.form.get("jmeno")
        ingredience = request.form.get("ingredience").split("\n")
        postup = request.form.get("postup")
        recept = {"jmeno":jmeno, "ingredience": ingredience, "postup": postup}
        db.append(recept)
        return redirect("/recepty")

if __name__ == "__main__":
    app.run(port=8500, debug=True)