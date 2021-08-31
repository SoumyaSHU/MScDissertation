import seaborn as sns
import numpy as np
from matplotlib import pyplot, pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix


def get_heatmap(df):
    # This function gives heatmap of all NaN values
    pyplot.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    pyplot.tight_layout()
    return pyplot.show()


def show_sentiments(data):
    sentiment_plot = data['sentiment'].value_counts(sort=True, ascending=False).plot(kind='bar', figsize=(4, 4),
                                                                                     title='Total occurance of sentiments')
    sentiment_plot.set_xlabel('sentiment')
    sentiment_plot.set_ylabel('frequency')
    pyplot.show()


def roc_curves(y_test, lr_probs):
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    # calculate roc curve for model
    fpr, tpr, _ = roc_curve(y_test, lr_probs)
    # plot model roc curve
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, marker='.', label='AUC = %0.2f' % roc_auc)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


def cf_matrix(y_test, y_pred):
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,
                fmt='.2%', cmap='Blues')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.tight_layout()
    return pyplot.show()


