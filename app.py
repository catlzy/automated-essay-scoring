from flask import Flask, request, render_template
import predict

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/input_essay', methods=['POST'])
def input_essay():
    essay = request.form['text']
    setnum = request.form['set']
    maxscores = ['12', '20', '3', '3', '4', '4', '30', '60']
    score = predict.predict(essay, setnum)[0]
    if float(score)  > int(maxscores[int(setnum)-1]):
        score = maxscores[int(setnum)-1]
    return 'You predicted score is: {} out of {}'.format(str(score), maxscores[int(setnum)-1])


if __name__ == "__main__":
    app.run(debug=True)
