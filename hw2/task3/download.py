from torchvision import datasets
datasets.OxfordIIITPet(root='./data', split='trainval', 
                                     target_types=['segmentation'], download=True)