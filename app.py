from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
test_data = pickle.load(open("test.pkl", "rb"))
cv = pickle.load(open("cv.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():

    if 'dataset' not in request.files:
        return 'No file part'

    file = request.files['dataset']

    if file.filename == '':
        return 'No selected file'
    if file:
        # Assuming the dataset is in CSV or TXT with character sequences
        try:
            df = pd.read_csv(file, sep="\t")
        except Exception as e:
            return f"Error reading file: {e}"
            # Strip any whitespace from column names
        df.columns = df.columns.str.strip()
        df1=df.select_dtypes(include=['number'])
        plt.figure(figsize=(6, 4))  # control size
        sns.countplot(x='class', data=df)
        plt.suptitle(' Dataset Class Distribution', y=1.02)
        # Save to static folder
        plot_path = os.path.join('static', 'df.png')
        plt.savefig(plot_path)
        plt.close()
        def Kmers_funct(seq, size=6):
            return [seq[x:x + size].lower() for x in range(len(seq) - size + 1)]
        df['words'] = df.apply(lambda x: Kmers_funct(x['sequence']), axis=1)
        df = df.drop('sequence', axis=1)
        df_texts = list(df['words'])
        for item in range(len(df_texts)):
            df_texts[item] = ' '.join(df_texts[item])
        #cv = CountVectorizer(ngram_range=(4, 4))
        data1= cv.transform(df_texts)
        pred= model.predict(data1)
        y_true = df['class']
        y_true = y_true.astype(str)
        pred = pred.astype(str)
        accuracy = round(accuracy_score(y_true, pred),3)
        precision = round(precision_score(y_true, pred,average='weighted'),3)
        recall = round(recall_score(y_true, pred,average='weighted'),3)
        f1 = round(f1_score(y_true, pred,average='weighted'),3)

        return render_template("result.html", graph_url=plot_path, accuracy=100*accuracy,precision=100*precision,
                               f1=100*f1)


if __name__ == '__main__':
    app.run(debug=True)
