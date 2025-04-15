from flask import Flask, request, render_template_string, redirect

import sqlite3

app = Flask(__name__)

# Inicializace databáze

def init_db():

    conn = sqlite3.connect('users.db')

    c = conn.cursor()

    c.execute('DROP TABLE IF EXISTS users')

    c.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)')

    c.execute('INSERT INTO users (username, password) VALUES ("admin", "admin123")')

    c.execute('INSERT INTO users (username, password) VALUES ("user", "user123")')

    conn.commit()

    conn.close()

init_db()

@app.route('/', methods=['GET', 'POST'])

def login():

    error = None

    if request.method == 'POST':

        username = request.form['username']

        password = request.form['password']

        conn = sqlite3.connect('users.db')

        c = conn.cursor()

        query = "SELECT * FROM users WHERE username = %s AND password = %s"

        print(f"DEBUG: {query}")

        c.execute(query, (username, password))

        result = c.fetchall()
        print(result)

        conn.close()

        if result:

            return f"Welcome, {username}!"

        else:

            error = "Invalid credentials"

    return render_template_string('''

        <h2>Login</h2>

        {% if error %}<p style="color:red;">{{ error }}</p>{% endif %}

        <form method="POST">

            Username: <input type="text" name="username"><br>

            Password: <input type="text" name="password"><br>

            <input type="submit" value="Login">

        </form>

    ''', error=error)

if __name__ == '__main__':

    app.run(debug=True)
