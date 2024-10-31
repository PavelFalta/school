from flask import Flask, render_template, request, redirect
import redis
import threading
from rq import Queue


r = redis.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)

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
    r.incr('visit_count')
    cached_home = r.get('home_page')
    if cached_home:
        return cached_home
    rendered_home = render_template("index.html", visit_count=r.get('visit_count'))
    r.setex('home_page', 60, rendered_home)
    return rendered_home

@app.route("/pozdrav")
def pozdrav_nastevnika():
    return "<p>Ahoj navstevniku"

@app.route("/recepty")
def zobraz_recepty():
    rendered_recepty = render_template("recepty.html", recepty=db)
    return rendered_recepty

@app.route("/formular", methods=["GET","POST"])
def zasli_recept():
    if request.method == "GET":
        return render_template("formular.html")
    else:
        jmeno = request.form.get("jmeno")
        ingredience = request.form.get("ingredience").split("\n")
        postup = request.form.get("postup")
        recept = {"jmeno":jmeno, "ingredience": ingredience, "postup": postup}
        r.rpush('recept_queue', str(recept))
        return redirect("/recepty")


if __name__ == "__main__":
    app.run(port=8500, debug=True)