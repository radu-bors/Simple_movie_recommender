import pandas as pd

from recommender import nmf_recommender
from flask import Flask, render_template, request

app = Flask(__name__, static_url_path='/static')

@app.route("/")
def homepage():
    return render_template("homepage.html")

@app.route("/recommendations", methods=["GET", "POST"])
def recommendations():
    # Load the movies.csv file to create a mapping from title to ID
    movies_df = pd.read_csv("./data/ml-latest-small/movies.csv")
    title_to_id = movies_df.set_index('title')['movieId'].to_dict()
    
    form = request.form.to_dict()
    
    # Extract movie names and ratings and construct the desired dictionary
    movies = {form[f"movie{i}"]: int(form[f"rating{i}"]) for i in range(1, 4)}
    
    # Replace movie titles with their corresponding IDs
    movies = {title_to_id.get(title, title): rating for title, rating in movies.items()}
    
    results = nmf_recommender(movies)
    
    return render_template("recommendations.html", movies = results, votes=form)

if __name__ == "__main__":
    app.run(debug=False, port=5000)