import json
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from random import randint
from models.autoEncoder import turbAI

def plot_losses_chart():
    with open("best_model_history.json") as f:
        history_dict = json.load(f)
        df = pd.DataFrame(history_dict)
        # Shift index so epoch starts at 1
        df.index = df.index + 1
        # Find the best val_loss
        best_epoch = len(df["val_loss"])-10
        best_val = df["val_loss"].min()
        df = df.rename(columns={'val_loss': 'Validação',
                        'loss':'Treinamento'})
        # Plot loss curves
        ax = df[["Treinamento", "Validação"]].plot.bar(
            figsize=(10, 6),
            xlabel="Época",
            ylabel="MSE",
            grid=False
        )

        # Highlight best val_loss
        ax.scatter(best_epoch-1, best_val, color="red", zorder=5, label=f"Melhor Epóca: {best_epoch} - MSE: {best_val:.4f}")

        # Add vertical dotted line
        ax.axvline(x=best_epoch-1, linestyle="--", color="red", alpha=0.6)

        ax.legend()
        plt.tight_layout()
        plt.savefig(r'output/charts/Training Charts.jpg', dpi=600, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    plot_losses_chart()
    turb = turbAI()
    turb.demo_model(model_name='best_model_f108542.keras')
    # turb.train_model()


