from sklearn.metrics import auc, roc_curve, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def print_performance_metrics(truth, pred, alg_text = ""):
    
    cr = classification_report(truth, pred > 0.5)
    print(cr)
    
    print(confusion_matrix(truth, pred>0.5))
    
    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(truth, pred, pos_label=1)
    print(f"N thresholds: {len(thresholds)}")
        
    roc_auc = roc_auc_score(truth, pred)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(alg_text + ' ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
