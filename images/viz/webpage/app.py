from flask import Flask, render_template, request, url_for, redirect, session

app = Flask(__name__)

app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'Secret'


@app.route('/')
def index():
    session.pop('name', None)
    return '<h1>Machine Learning for Algorithmic Learning. An Example with AAPLE stock.</h1>'


@app.route('/home/<string:name>', methods=['GET', 'POST'])
def home(name):
    session["name"] = name
    return render_template('home.html', name=name)


@app.route('/theform', methods=['GET', 'POST'])
def theform():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        name = request.form['name']
        return redirect(url_for('home', name=name))


if __name__ == '__main__':
    app.run(debug=True)
