from read_dataset import read_mnist_images, read_mnist_labels
from model import MLP
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

test_images = read_mnist_images('fashion/t10k-images-idx3-ubyte.gz')
test_images = test_images.reshape(test_images.shape[0], -1).astype(np.float32) / 255.0
test_labels = read_mnist_labels('fashion/t10k-labels-idx1-ubyte.gz')
test_labels = test_labels.astype(np.int32)

model1 = MLP([784,256,64,10], local_para_path="grid_search_results/best_model.npz")
model2 = MLP([784,256,64,10], local_para_path="grid_search_results_2/best_model.npz")
model3 = MLP([784,128,32,10], local_para_path="grid_search_results_3/best_model.npz", activation_function="sigmoid")

pred1, loss1 = model1.infer(test_images, test_labels)
pred2, loss2 = model2.infer(test_images, test_labels)
pred3, loss3 = model3.infer(test_images, test_labels)

def print_evaluation(model_name, pred, true_labels, loss):
    print(f"[{model_name}]")
    acc = accuracy_score(true_labels, pred)
    print(f"1. Accuracy: {acc:.4f}")
    avg_loss = np.mean(loss) if hasattr(loss, '__len__') else loss
    print(f"2. Loss: {avg_loss:.4f}")
    cm = confusion_matrix(true_labels, pred)
    print("3. Confusion Matrix:")
    print(cm)
    error_idx = np.where(pred != true_labels)[0]
    if len(error_idx) > 0:
        n = min(10, len(error_idx))
        sample_idx = np.random.choice(error_idx, size=n, replace=False)
        sample_idx.sort()
        print(f"4. random 10 errors: {sample_idx}")
        print(f"pred: {pred[sample_idx]}")
        print(f"true: {true_labels[sample_idx]}")
    else:
        print("4. no error")
        
    print("=" * 45)

print_evaluation("Model 1 (best_model.npz)", pred1, test_labels, loss1)
print_evaluation("Model 2 (best_model.npz)", pred2, test_labels, loss2)
print_evaluation("Model 3 (best_model.npz)", pred3, test_labels, loss3)