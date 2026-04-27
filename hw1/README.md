# readme

## requirements

- python >= 3.12
- requirements: requirements.txt

## model

放在grid_search_results/best_model.npz, grid_search_results_2/best_model.npz, grid_search_results_3/best_model.npz下

## train

修改model.py中main中下方需要调整的的参数并运行
trainer = GridSearchTrainer(..., val_ratio=0.2,..., output_dir="grid_search_results_3")
lr_list = [1e-3, 5e-3]
wd_list = [1e-4, 1e-3]
hidden_layers_list = [[128, 32]]
trainer.run_grid_search(..., activation="sigmoid")

## test

运行test.py可以看到控制台输出结果
可以修改如下参数加载不同模型，需要手动控制输入层，隐藏层，输出层大小
model1 = MLP([784, 256, 64, 10], local_para_path="grid_search_results/best_model.npz")
