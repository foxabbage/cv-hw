import numpy as np
from typing import Literal, Union, List
from numpy.typing import ArrayLike
import os
import itertools
import matplotlib.pyplot as plt
from read_dataset import read_mnist_images, read_mnist_labels

class SGD:
    def __init__(self, lr:float, weight_decay:float, epoch:int):
        self.epoch = epoch
        self.lr = lr
        self.lr_decay:Literal["cos", "step"] = "cos"
        self.lr_decay_cos_max = lr
        self.lr_decay_cos_min = 0
        self.lr_decay_step = 5
        self.lr_decay_rate = 0.4
        self.weight_decay = weight_decay
        self.loss_record_step = 1000
        self.early_stop_patience = 5
        self.early_stop_delta = 1e-3
        self.grad_clip = False
        self.grad_clip_norm = 1
    
    def lr_update(self, epoch:int):
        """epoch should start from 1"""
        if self.lr_decay == "cos":
            self.lr = self.lr_decay_cos_min + 0.5*(self.lr_decay_cos_max-self.lr_decay_cos_min)*(1+np.cos(epoch/self.epoch*np.pi))
        elif self.lr_decay == "step":
            if epoch % self.lr_decay_step == 0:
                self.lr = self.lr*self.lr_decay_rate


class MLP:
    def __init__(self, layer_size:ArrayLike, activation_function:Literal["relu","sigmoid"]="relu", local_para_path:Union[str, None]=None):
        self.layer_size = np.array(layer_size)
        self.para_num = self.layer_size.size - 1
        self.activation_function = activation_function
        self.w_list:List[np.ndarray] = []
        self.b_list:List[np.ndarray] = []
        self.grad_cache = []
        self.result_cache = []
        if activation_function == "relu":
            self.f = self.relu
            self.df = self.d_relu
        elif activation_function == "sigmoid":
            self.f = self.sigmoid
            self.df = self.d_sigmoid
        if not local_para_path:
            self._init_parameters()
        else:
            self._load_model(local_para_path)
    
    def _init_parameters(self):
        """use x@w + b to calculate, b with shape (n, )"""
        if self.activation_function == "relu":
            for i in range(self.para_num):
                std = np.sqrt(2.0/(self.layer_size[i]))
                self.w_list.append(np.random.normal(0, std, (self.layer_size[i], self.layer_size[i+1]))) #(in, out) 
                self.b_list.append(np.ones(self.layer_size[i+1])*0.01)
        elif self.activation_function == "sigmoid":
            for i in range(self.para_num):
                std = np.sqrt(2.0/(self.layer_size[i]+self.layer_size[i+1]))
                self.w_list.append(np.random.normal(0, std, (self.layer_size[i], self.layer_size[i+1]))) #(in, out) 
                self.b_list.append(np.ones(self.layer_size[i+1])*0.01)
    
    def save_model(self, path='model_params.npz'):
        save_dict = {}
        for i, w in enumerate(self.w_list):
            save_dict[f'w_{i}'] = w
        for i, b in enumerate(self.b_list):
            save_dict[f'b_{i}'] = b
        
        np.savez(path, **save_dict)

    def _load_model(self, path):
        data = np.load(path)
        self.w_list = []
        self.b_list = []
        
        i = 0
        while f'w_{i}' in data.files:
            self.w_list.append(data[f'w_{i}'])
            i += 1
            
        i = 0
        while f'b_{i}' in data.files:
            self.b_list.append(data[f'b_{i}'])
            i += 1

    def train(self, X: np.ndarray, y: np.ndarray, para: SGD, X_val: np.ndarray = None, y_val: np.ndarray = None, verbose: bool = True):
        self.hyper_para = para
        loss_list = []

        best_val_acc = -1.0
        patience_counter = 0
        val_acc_history = []
        val_loss_history = []
        best_w = [w.copy() for w in self.w_list]
        best_b = [b.copy() for b in self.b_list]

        for epoch in range(self.hyper_para.epoch):
            indices = np.random.permutation(len(X))
            for j in range(len(X)):
                loss = self._train_one_sample(X[indices[j]], y[indices[j]])
                if j % self.hyper_para.loss_record_step == 0:
                    loss_list.append(loss)
            self.hyper_para.lr_update(epoch + 1)

            val_pred, val_loss = self.infer(X_val, y_val)
            val_acc = self.calculate_accuracy(val_pred, y_val)
            val_acc_history.append(val_acc)
            val_loss_history.append(val_loss)


            if val_acc > best_val_acc + self.hyper_para.early_stop_delta:
                best_val_acc = val_acc
                patience_counter = 0
                for idx in range(len(self.w_list)):
                    best_w[idx] = self.w_list[idx].copy()
                    best_b[idx] = self.b_list[idx].copy()
                if verbose:
                    print(f"Epoch {epoch+1:03d} | Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                if verbose:
                    print(f"Epoch {epoch+1:03d} | Val Acc: {val_acc:.4f} | Patience: {patience_counter}/{self.hyper_para.early_stop_patience}")

            if patience_counter >= self.hyper_para.early_stop_patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}. Restoring best weights (Val Acc: {best_val_acc:.4f})")
                self.w_list = best_w
                self.b_list = best_b
                break

        return loss_list, val_acc_history, val_loss_history

    def infer(self, X: np.ndarray, y: np.ndarray):
        u = X
        for i in range(self.para_num - 1):
            u = u @ self.w_list[i] + self.b_list[i]
            u = self.f(u)
        u = u @ self.w_list[-1] + self.b_list[-1]
        probs = self.softmax(u)
        
        y_pred = np.argmax(probs, axis=1)
        batch_size = X.shape[0]
        loss = -np.mean(np.log(probs[np.arange(batch_size), y] + 1e-15))
        return y_pred, loss

    def _train_one_sample(self, x, y_label):
        p = self._infer_one_sample(x)
        loss = self.cross_entropy_loss(p, y_label)
        y_hot = np.zeros(self.layer_size[-1])
        y_hot[y_label] = 1.0
        now_grad = p - y_hot
        next_grad = now_grad@self.w_list[-1].T
        grad_w = np.outer(self.grad_cache[-1], now_grad)
        self.w_list[-1] = (1-self.hyper_para.lr*self.hyper_para.weight_decay)*self.w_list[-1] - self.hyper_para.lr*grad_w
        self.b_list[-1] = self.b_list[-1] - self.hyper_para.lr*now_grad
        now_grad = next_grad
        for i in reversed(range(self.para_num-1)):
            next_grad = now_grad@self.w_list[i].T
            grad_w = np.outer(self.result_cache[i], (self.grad_cache[i]*now_grad))
            grad_b = self.grad_cache[i]*now_grad
            if self.hyper_para.grad_clip:
                norm_w = np.linalg.norm(grad_w)
                norm_b = np.linalg.norm(grad_b)
                if norm_w > self.hyper_para.grad_clip_norm:
                    norm_w = (self.hyper_para.grad_clip_norm/norm_w)*grad_w
                if norm_b > self.hyper_para.grad_clip_norm:
                    norm_b = (self.hyper_para.grad_clip_norm/norm_b)*grad_b
            self.w_list[i] = (1-self.hyper_para.lr*self.hyper_para.weight_decay)*self.w_list[i] - self.hyper_para.lr*grad_w
            self.b_list[i] = self.b_list[i] - self.hyper_para.lr*grad_b
            now_grad = next_grad
        self._clear_cache()
        return loss

    def _infer_one_sample(self, x):
        u = x
        self.result_cache.append(u)
        for i in range(self.para_num-1):
            u = u@self.w_list[i] + self.b_list[i]
            self.grad_cache.append(self.df(u))
            u = self.f(u)
            self.result_cache.append(u)
        self.grad_cache.append(u)
        u = u@self.w_list[-1] + self.b_list[-1] # last layer only linear, z=uW+b, output p=softmax(z)
        return self.softmax(u) # p, dLoss/dz = p-y_true_one_hot
    
    def cross_entropy_loss(self, p, y_labels):
        p_safe = np.clip(p, 1e-12, 1.0)
        return -np.log(p_safe[y_labels])

    def relu(self, x:ArrayLike) -> np.ndarray:
        return np.maximum(0, x)
    
    def d_relu(self, x:ArrayLike) -> np.ndarray:
        dx = np.zeros_like(x)
        dx[x > 0] = 1
        return dx
    
    def sigmoid(self, x:ArrayLike) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def d_sigmoid(self, x:ArrayLike) -> np.ndarray:
        exp_neg_x = np.exp(-x)
        return exp_neg_x / (1 + exp_neg_x)**2
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        if x.ndim==1:
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _clear_cache(self):
        self.grad_cache.clear()
        self.result_cache.clear()

    def calculate_accuracy(self, y_pred:np.ndarray, y_true:np.ndarray):
        return np.mean(y_pred==y_true)


class GridSearchTrainer:
    def __init__(self, X_raw: np.ndarray, y_raw: np.ndarray, val_ratio: float = 0.2, num_classes: int = 10, output_dir: str = "grid_search_results"):
        X = X_raw.reshape(X_raw.shape[0], -1).astype(np.float32) / 255.0
        y = y_raw.astype(np.int32)
        
        indices = np.random.permutation(len(X))
        split_idx = int(len(X) * (1 - val_ratio))
        
        self.X_train, self.X_val = X[indices[:split_idx]], X[indices[split_idx:]]
        self.y_train, self.y_val = y[indices[:split_idx]], y[indices[split_idx:]]
        
        self.input_dim = self.X_train.shape[1]
        self.num_classes = num_classes
        self.best_val_acc = -1.0
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run_grid_search(self, 
                        lr_list: List[float], 
                        wd_list: List[float], 
                        hidden_layers_list: List[List[int]], 
                        epochs: int = 30, 
                        activation: str = "relu",
                        grad_clip = False):
        param_combos = list(itertools.product(lr_list, wd_list, hidden_layers_list))

        for i, (lr, wd, hidden) in enumerate(param_combos):
            print(f"\n[{i+1}/{len(param_combos)}] LR={lr} | WD={wd} | Hidden={list(hidden)}")
            layer_sizes = [self.input_dim] + list(hidden) + [self.num_classes]
            
            model = MLP(layer_sizes, activation_function=activation)
            optimizer = SGD(lr=lr, weight_decay=wd, epoch=epochs)
            optimizer.grad_clip = grad_clip

            loss_list, val_acc_history, val_loss_history = model.train(
                self.X_train, self.y_train, optimizer,
                X_val=self.X_val, y_val=self.y_val, verbose=True
            )
            
            current_val_acc = val_acc_history[-1] if val_acc_history else 0.0

            self._save_curves(loss_list, val_acc_history, val_loss_history, lr, wd, hidden, current_val_acc, run_id=i, grad_clip=grad_clip)

            if current_val_acc > self.best_val_acc:
                self.best_val_acc = current_val_acc
                best_path = os.path.join(self.output_dir, "best_model.npz")
                model.save_model(best_path)
                print(f"better accuracy: {self.best_val_acc:.4f}")
                
        print(f"\nbest accuracy: {self.best_val_acc:.4f}")

    def _save_curves(self, loss_list, val_acc_history, val_loss_history, lr, wd, hidden, final_acc, run_id, grad_clip):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        ax1.plot(loss_list, marker='o', markersize=3, color='tab:blue')
        ax1.set_title("Training Loss")
        ax2.set_xlabel("Iterations / 1000 steps")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

        ax2.plot(val_acc_history, marker='s', markersize=3, color='tab:red')
        ax2.set_title("Validation Accuracy")
        ax2.set_xlabel("Iterations / Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, alpha=0.3)

        ax3.plot(val_loss_history, marker='s', markersize=3, color='tab:blue')
        ax3.set_title("Validation Loss")
        ax3.set_xlabel("Iterations / Epochs")
        ax3.set_ylabel("Loss")
        ax3.grid(True, alpha=0.3)

        param_str = f"LR: {lr} | Weight Decay: {wd} | Hidden Layers: {list(hidden)} | Final Val Acc: {final_acc:.4f}"
        fig.suptitle(param_str, fontsize=13, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        if not grad_clip:
            save_path = os.path.join(self.output_dir, f"run_{run_id}_lr{lr}_wd{wd}.png")
        else:
            save_path = os.path.join(self.output_dir, f"run_{run_id}_lr{lr}_wd{wd}_gradclip.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

if __name__ == "__main__":
    train_images = read_mnist_images('fashion/train-images-idx3-ubyte.gz')
    train_labels = read_mnist_labels('fashion/train-labels-idx1-ubyte.gz')

    trainer = GridSearchTrainer(X_raw=train_images, y_raw=train_labels, val_ratio=0.2, num_classes=10, output_dir="grid_search_results_3")

    lr_list = [1e-3, 5e-3]
    wd_list = [1e-4, 1e-3]
    hidden_layers_list = [[128,32]]

    trainer.run_grid_search(lr_list=lr_list, wd_list=wd_list, hidden_layers_list=hidden_layers_list, activation="sigmoid")
