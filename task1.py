import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

p_imagenet_csv = Path("resnet18_25_07_23_17_56/descriptions.csv")
p_places_csv   = Path("resnet18_places_25_07_23_18_32/descriptions.csv")
out_dir        = Path("analysis_figures")
out_dir.mkdir(exist_ok=True)

df_im = pd.read_csv(p_imagenet_csv)
df_pl = pd.read_csv(p_places_csv)
df_im["model"] = "ResNet18_ImageNet"
df_pl["model"] = "ResNet18_Places365"
df_all = pd.concat([df_im, df_pl], ignore_index=True)

top_k = 30
count_im = df_im["description"].value_counts().head(top_k)
count_pl = df_pl["description"].value_counts().head(top_k)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
count_im.plot(kind="barh", ax=axes[0])
axes[0].invert_yaxis()
axes[0].set_title("Top concepts (ImageNet)")
axes[0].set_xlabel("#neurons")

count_pl.plot(kind="barh", ax=axes[1])
axes[1].invert_yaxis()
axes[1].set_title("Top concepts (Places365)")
axes[1].set_xlabel("#neurons")

plt.tight_layout()
plt.savefig(out_dir / "top_concepts_comparison.png", dpi=300)
plt.close()

set_im = set(df_im["description"])
set_pl = set(df_pl["description"])
unique_im = len(set_im)
unique_pl = len(set_pl)
overlap   = len(set_im & set_pl)
union     = len(set_im | set_pl)
jaccard   = overlap / union if union else 0

print(f"#Unique concepts (ImageNet): {unique_im}")
print(f"#Unique concepts (Places365): {unique_pl}")
print(f"#Overlap concepts: {overlap}")
print(f"Jaccard similarity: {jaccard:.3f}")

layer_stats = (df_all
               .groupby(["model", "layer"])
               .agg(unique_concepts=("description", "nunique"),
                    avg_similarity=("similarity", "mean"),
                    median_similarity=("similarity", "median"),
                    max_similarity=("similarity", "max"),
                    n_units=("description", "count"))
               .reset_index())

layer_stats.to_csv(out_dir / "layer_stats.csv", index=False)

pivot_uc = layer_stats.pivot(index="layer", columns="model", values="unique_concepts")
pivot_uc.plot(kind="bar", figsize=(8,5))
plt.ylabel("#unique concepts")
plt.title("Unique concepts per layer")
plt.tight_layout()
plt.savefig(out_dir / "unique_concepts_per_layer.png", dpi=300)
plt.close()

sns.kdeplot(data=df_all, x="similarity", hue="model")
plt.title("Similarity distribution")
plt.tight_layout()
plt.savefig(out_dir / "similarity_distribution.png", dpi=300)
plt.close()

cnt_im = df_im["description"].value_counts()
cnt_pl = df_pl["description"].value_counts()
all_concepts = set(cnt_im.index) | set(cnt_pl.index)

diff_df = pd.DataFrame({
    "concept": list(all_concepts),
    "freq_im": [cnt_im.get(c,0) for c in all_concepts],
    "freq_pl": [cnt_pl.get(c,0) for c in all_concepts]
})
diff_df["diff_im_minus_pl"] = diff_df["freq_im"] - diff_df["freq_pl"]
diff_df["diff_pl_minus_im"] = -diff_df["diff_im_minus_pl"]

top_im_bias = diff_df.sort_values("diff_im_minus_pl", ascending=False).head(10)
top_pl_bias = diff_df.sort_values("diff_pl_minus_im", ascending=False).head(10)

top_im_bias.to_csv(out_dir / "top_imagenet_bias.csv", index=False)
top_pl_bias.to_csv(out_dir / "top_places365_bias.csv", index=False)

print("Results saved to:", out_dir.resolve())
