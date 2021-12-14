CONFIG = {
    "image_size":64,
    "mean":[0.485, 0.456, 0.406],
    "std":[0.229, 0.224, 0.225],
    "batch_size": 128,
    "shuffle": True,
    "num_workers": 4,
    "d_hidden": 128,
    "num_classes": 5,
    "number_channel": 3,
    "learning_rate": 0.0001,
    "num_epochs": 40,
    "ckpt_save_interval": 40,
    "data_folder": "../data/image_folder",
    "data_path": "../data/image_info_data.json",
    "outlier_ids_path": "../data/preprocessing/indices2remove_allcategory.json"
}