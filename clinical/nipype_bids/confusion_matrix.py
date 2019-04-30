import sys
import yaml
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import unicodedata
from sklearn.metrics import confusion_matrix

def plot_confusion(y_true, y_pred, n_epoch, output_file):
    fig = plt.figure(figsize = (1,1))
    plt.rcParams["font.size"] = 6
    
    cmx_data = confusion_matrix(y_true, y_pred, labels=[0, 1])
    df_cmx = pd.DataFrame(cmx_data, index=[0, 1], columns=[0, 1])
    
    sns.heatmap(df_cmx, annot=True, fmt='g', cmap='Blues')
    
    plt.title("Confusion matrix\n%s (%d epoch)\n" % ('3D VGG-like', n_epoch))
    plt.xlabel("Predict")
    plt.ylabel("Actual")
    
    fig.tight_layout()
    plt.savefig(output_file)
    print("Saved confusion matrix to \"%s\"" % output_file)

#plot_confusion([1,0,0,1], [1,0,1,0], 3, 'confusion.pdf')
