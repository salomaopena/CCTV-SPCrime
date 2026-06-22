import pandas as pd
from sklearn.metrics import cohen_kappa_score

df = pd.read_csv("anotacoes_amostra.csv")

df_pair = df.pivot_table(
    index="image_id",
    columns="annotator",
    values="class",
    aggfunc=lambda x: x.iloc[0],
)

kappa = cohen_kappa_score(df_pair["anotador1"], df_pair["anotador2"])
print("Cohen's kappa:", kappa)