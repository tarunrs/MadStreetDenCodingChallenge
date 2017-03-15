from flask import Flask, render_template, request
from index import EnronInvertedIndex

app = Flask(__name__)
Index = EnronInvertedIndex()


@app.route("/")
def home():
    query = request.args.get('query')
    page = request.args.get('page')
    if not page:
        page = 1
    else:
        page = int(page)
    if query:
        results = Index.search_results(query.strip().split(),page=page)
        return render_template('index.html', results=results, query=query)
    else:
        return render_template('index.html')
if __name__ == "__main__":
  app.run(host="0.0.0.0")

