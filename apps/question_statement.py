import os
import pandas as pd

tutoringdata_folderpath = "/content/drive/MyDrive/Colab Notebooks/MASTER/TutoringData/annotation_data/annotation1/"

question_filenames = ["pair1-1.csv", "pair2-1.csv", "pair3-1.csv", "pair4-1.csv"]

questions = list()

for filename in question_filenames:
    df = pd.read_csv(os.path.join(tutoringdata_folderpath, filename), skiprows=1, header=0)
    _questions = df["発問"].values.tolist()
    questions = questions + _questions

for i, q in enumerate(questions):
    print(f"{i}: {q}")