CONFIG = {
    "image_size":64,
    "mean":[0.5, 0.5, 0.5],
    "std":[0.5, 0.5, 0.5],
    "batch_size": 32,
    "shuffle": True,
    "num_workers": 6,
    "d_hidden": 128,
    "num_classes": 3,
    "number_channel": 3,
    "learning_rate": 0.0002,
    "num_epochs": 10,
    "ckpt_save_interval": 10,
    "data_folder": "../data/image_folder",
    "data_path": "../data/image_info_data_3categories.json",
    "outlier_ids_path": "../data/preprocessing/indices2remove_3categories.json",
    "ckpt_folder": "../data/ckpt/16122021",
    "resume_ckpt": "../data/ckpt/16122021/ckpt_epoch10.pth"
}
