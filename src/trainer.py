import json, torch, torchvision, os
import numpy as np
from PIL.Image import Image
from src.model import Generator, Discriminator
from src.dataloader import get_loader, ImageDatasetWithCategory
from src.miscs import get_linear_schedule_with_warmup
from tqdm import tqdm


class Trainer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available() == True: self.device = torch.device("cuda")
        else: self.device = torch.device("cpu")

        with open(config["data_path"]) as f:
            data = json.load(f)
        with open(config["outlier_ids_path"]) as f:
            outlier_ids = json.load(f)
        dataset = ImageDatasetWithCategory(data, outlier_ids, config)
        self.loader = get_loader(dataset, config)


        self.generator = Generator(config).to(self.device)
        self.discriminator = Discriminator(config).to(self.device)

        self.optimizer_g = torch.optim.RMSprop(self.generator.parameters(), lr=config["learning_rate"])
        self.optimizer_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=config["learning_rate"])

        self.criteria = torch.nn.BCELoss(reduction = "mean")


        self.fixed_noise = torch.randn(5, config["d_hidden"]).float()
        self.fixed_labels = torch.eye(5)




    def train_one_epoch(self, epoch):
        d_hidden = self.config["d_hidden"]
        num_classes = self.config["num_classes"]
        image_size = self.config["image_size"]

        train_progress_bar = tqdm(self.loader, desc = f"Epoch {epoch}")

        self.generator.train()
        self.discriminator.train()

        for batch_index, (images_real, labels_real) in enumerate(train_progress_bar):
            images_real = images_real.to(self.device)
            labels_real = labels_real.to(self.device).float()

            # generate noise and random labels to generate fake images
            noise = torch.randn(images_real.size(0), d_hidden).float().to(self.device)
            labels_fake = torch.nn.functional.one_hot(torch.from_numpy(np.random.choice(range(0,num_classes), labels_real.size(0))), num_classes).to(self.device).float()

            images_fake = self.generator(noise, labels_fake)

            assert images_real.shape == images_fake.shape
            assert labels_real.shape == labels_fake.shape


            # train discriminator
            self.discriminator.zero_grad()
            discriminator_output_real = self.discriminator(images_real, 
                                                           labels_real.view(labels_real.size(0), \
                                                                labels_real.size(1), 1, 1).repeat(1, 1, image_size, image_size))

            discriminator_output_fake = self.discriminator(images_fake.detach(),
                                                           labels_fake.view(labels_fake.size(0), \
                                                                labels_fake.size(1), 1, 1).repeat(1, 1, image_size, image_size))

            discriminator_loss = -(torch.mean(discriminator_output_real) - torch.mean(discriminator_output_fake))
            self.optimizer_d.step()

            for p in self.discriminator.parameters():
                p.data.clamp_(-self.config["clip_value"], self.config["clip_value"])

            # train generator
            if batch_index % self.config["n_critic"] == 0:
                self.generator.zero_grad()
                discriminator_output_fake = self.discriminator(images_fake,
                                                            labels_fake.view(labels_fake.size(0), \
                                                                    labels_fake.size(1), 1, 1).repeat(1, 1, image_size, image_size))
                generator_loss = -torch.mean(discriminator_output_fake)
                generator_loss.backward()
                self.optimizer_g.step()


            train_progress_bar.set_postfix({
                "Generator loss": generator_loss.item(),
                "Discriminator loss": discriminator_loss.item()
            })

    def save_model(self, epoch):
        if epoch % self.config["ckpt_save_interval"] != 0: return
        print("Saving model")
        ckpt_path = f'{self.config["ckpt_folder"]}/ckpt_epoch{epoch}.pth'
        torch.save({
            "discriminator": self.discriminator.state_dict(),
            "generator": self.generator.state_dict()
        }, ckpt_path)
        print("Done saving model")

    def save_images(self, epoch):
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise, self.fixed_labels)
        torchvision.utils.save_image(fake_images, os.path.join(self.config["ckpt_folder"], f"epoch{epoch}.jpg"), normalize = True)

    def train(self, start = 0):
        assert os.path.exists(self.config["ckpt_folder"])
        num_epochs = self.config["num_epochs"]
        for epoch in range(start, start + num_epochs):
            self.train_one_epoch(epoch + 1)
            self.save_model(epoch + 1)
            self.save_images(epoch + 1)