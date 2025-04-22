from flask import Flask, request, redirect, render_template_string
app = Flask(__name__)

todos = []

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        task = request.form.get("task")
        if task:
            todos.append(task)
        return redirect("/")
    return render_template_string("""
    <h1>TODO Seznam</h1>
    <form method="post">
    <input name="task" placeholder="Zadej úkol">
    <input type="submit" value="Přidat">
    </form>
    <ul>
    {% for t in todos %}
    <li>{{ t }}</li>
    {% endfor %}
    </ul>
    """, todos=todos)
if __name__ == "__main__":
    app.run(debug=True)