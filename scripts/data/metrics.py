import pandas
from sklearn.metrics import classification_report, roc_auc_score


def report(y_true, y_pred):

    class_report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pandas.DataFrame(class_report).transpose()

    return report_df


def roc_auc(y_true, y_pred):
    roc = roc_auc_score(y_true, y_pred)
