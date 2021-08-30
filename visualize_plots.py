import seaborn as sns
from matplotlib import pyplot, pyplot as plt
from sklearn import metrics
from yellowbrick.classifier.classification_report import classification_report
from yellowbrick.classifier.rocauc import roc_auc


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


def roc_curves(model_name, y_test, y_pred):
    # calculate roc curves
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    print('ROC AUC=%.3f' % roc_auc)
    plt.title('ROC Plot for Model: {}'.format(model_name))
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def report_classification(pipeline, X_train, y_train, X_test, y_test):
    # Plot Classification Report
    index = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
    report_classification = classification_report(pipeline,
                                                  X_train, y_train,
                                                  X_test, y_test,
                                                  support="percent",
                                                  cmap="Reds",
                                                  classes=index,
                                                  fig=pyplot.figure(figsize=(8, 6))
                                                  )
    return report_classification