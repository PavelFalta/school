from flask import Flask, request, render_template_string

app = Flask(__name__)

# In-memory "databáze" komentářů

comments = []

@app.route('/', methods=['GET', 'POST'])

def guestbook():

    if request.method == 'POST':

        comment = request.form['comment']

        comments.append(comment)

    return render_template_string('''

        <h2>📖 Guestbook</h2>

        <p>Leave a comment – anything you write will be shown below.</p>

        <form method="POST">

            <textarea name="comment" rows="4" cols="40"></textarea><br>

            <input type="submit" value="Post Comment">

        </form>

        <hr>

        <h3>🗨️ Comments:</h3>

        {% for comment in comments %}

            <div style="padding:5px; border:1px solid #ccc; margin:5px;">

                {{ comment | safe }}

            </div>

        {% endfor %}

    ''', comments=comments)

if __name__ == '__main__':

    app.run(debug=True)
