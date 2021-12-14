import torch, torchvision, json, os, PIL
from torchvision import transforms
from miscs import load_image



class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data,
                 data_folder,
                 config):
        self.data = data
        self.DATA_FOLDER = data_folder
        self.error_index = set([])

        #mean=[0.485, 0.456, 0.406]
        #std=[0.229, 0.224, 0.225]
        imsize, mean, std = config["image_size"], config["mean"], config["std"]
        self.imsize = imsize
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((imsize, imsize)),
                                             transforms.Normalize(mean=mean, std=std)])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index in self.error_index: return torch.zeros((3, self.imsize, self.imsize))

        image_path = os.path.join(self.DATA_FOLDER, self.data[index]["id"] + ".jpg")
        image = load_image(image_path)

        if image is None or image.shape[-1] != 3:
            self.error_index.add(index)
            return torch.zeros((3, self.imsize, self.imsize))

        return self.transform(image)



class ImageDatasetWithCategory(torch.utils.data.Dataset):
    def __init__(self, 
                 data: dict, 
                 outlier_ids: dict, 
                 config: dict):
        super(ImageDatasetWithCategory, self).__init__()
        self.preprocess_data(data, outlier_ids)
        self.error_index = set([])

        imsize, mean, std, num_classes = config["image_size"], config["mean"], config["std"], config["num_classes"]
        self.num_classes = num_classes
        self.imsize = imsize
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((imsize, imsize)),
                                             transforms.Normalize(mean=mean, std=std)])

    def preprocess_data(self, data, outlier_ids):
        data = {int(k):v for k,v in data.items()}
        outlier_ids = {int(k):v for k,v in outlier_ids.items()}

        preprocessed_data = []
        for category in data:
            for index, dp in enumerate(data[category]):
                if index in outlier_ids[category]: continue
                to_append = dp.copy()
                to_append["category"] = category
                preprocessed_data.append(to_append)
        self.data = preprocessed_data


    def get_one_hot(self, category):
        # example: category = 3, self.num_classes = 5 => return [0,0,0,1,0]
        return torch.nn.functional.one_hot(torch.tensor(category), self.num_classes)


    def __getitem__(self, index):
        if index in self.error_index: return torch.zeros((3, self.imsize, self.imsize))

        image_path = os.path.join(self.DATA_FOLDER, self.data[index]["id"] + ".jpg")
        image = load_image(image_path)

        if image is None or image.shape[-1] != 3:
            self.error_index.add(index)
            return torch.zeros((3, self.imsize, self.imsize))

        return self.transform(image), self.get_one_hot(self.data[index]["category"])


def get_loader(dataset, config):
    batch_size = config["batch_size"],
    shuffle = config["shuffle"]
    num_workers = config["num_workers"]

    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size = batch_size,
                                            shuffle = shuffle,
                                            num_workers = num_workers)

    return dataloader