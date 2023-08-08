from recommender import random_recommender, nmf_recommender
from flask import Flask, render_template, request

app = Flask(__name__, static_url_path='/static')

@app.route("/")
def homepage():
    return render_template("homepage2.html")

@app.route("/recommendations")
def recommendations():
    form = request.args
    form_data = dict(form) #you can use this as input to your recommender function in the next line
    #results=nmf_recommender(form_data)
    results = random_recommender()
    return render_template("recommendations.html", movies=results, votes=form)

if __name__ == "__main__":
    app.run(debug=True)