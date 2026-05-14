import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_imp = pd.read_csv("all_feature_importances.csv")
df_og = pd.read_csv(r"C:\Users\ashok\OneDrive\Desktop\iiit\dataset.csv", skiprows=1)
top10 = df_imp.nlargest(10, "importance")["feature"].tolist()

plt.figure(figsize=(20, 10))
for i, feat in enumerate(top10):
    plt.subplot(2, 5, i + 1)
    sns.boxplot(y=df_og[feat], color="salmon")
    plt.title(feat)
plt.tight_layout()
plt.savefig("all_boxplots.png")
plt.show()

plt.figure(figsize=(20, 10))
for i, feat in enumerate(top10):
    plt.subplot(2, 5, i + 1)
    sns.histplot(df_og[feat], kde=True, color="teal")
    plt.title(feat)
plt.tight_layout()
plt.savefig("all_histograms.png")
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(df_og[top10].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()