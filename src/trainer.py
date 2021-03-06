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

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=config["learning_rate"], betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=config["learning_rate"], betas=(0.5, 0.999))

        self.criteria = torch.nn.BCELoss(reduction = "mean")
        num_steps_one_epoch = len(self.loader)
        self.scheduler_g = get_linear_schedule_with_warmup(self.optimizer_g, 
                                                            num_steps_one_epoch, 
                                                            num_steps_one_epoch * self.config["num_epochs"],
                                                            last_epoch = - 1)
        self.scheduler_d = get_linear_schedule_with_warmup(self.optimizer_d, 
                                                            num_steps_one_epoch, 
                                                            num_steps_one_epoch * self.config["num_epochs"],
                                                            last_epoch = - 1)


        self.fixed_noise = torch.randn(self.config["num_classes"], config["d_hidden"]).float()
        self.fixed_labels = torch.eye(self.config["num_classes"])




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
            discriminator_loss_real = self.criteria(discriminator_output_real, torch.ones_like(discriminator_output_real).to(self.device))
            discriminator_loss_real.backward()

            discriminator_output_fake = self.discriminator(images_fake.detach(),
                                                           labels_fake.view(labels_fake.size(0), \
                                                                labels_fake.size(1), 1, 1).repeat(1, 1, image_size, image_size))
            discriminator_loss_fake = self.criteria(discriminator_output_fake, torch.zeros_like(discriminator_output_fake).to(self.device))
            discriminator_loss_fake.backward()

            averaged_discriminator_loss = (discriminator_loss_real.item() + discriminator_loss_fake.item()) / 2
            self.optimizer_d.step()
            self.scheduler_d.step()

            # train generator
            self.generator.zero_grad()
            discriminator_output_fake = self.discriminator(images_fake,
                                                           labels_fake.view(labels_fake.size(0), \
                                                                labels_fake.size(1), 1, 1).repeat(1, 1, image_size, image_size))
            generator_loss = self.criteria(discriminator_output_fake, torch.ones_like(discriminator_output_fake).to(self.device))
            generator_loss.backward()

            averaged_generator_loss = generator_loss.item()
            self.optimizer_g.step()
            self.scheduler_g.step()


            train_progress_bar.set_postfix({
                "Generator loss": averaged_generator_loss,
                "Discriminator loss": averaged_discriminator_loss
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

    def resume_training(self):
        if os.path.exists(self.config["resume_ckpt"]):
            checkpoint = torch.load(self.config["resume_ckpt"])
            self.discriminator.load_state_dict(checkpoint["discriminator"])
            self.generator.load_state_dict(checkpoint["generator"])
            print("Loaded checkpoint")
        else: print("Train from scratch")

    def save_images(self, epoch):
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise.to(self.device), self.fixed_labels.to(self.device))
        torchvision.utils.save_image(fake_images, os.path.join(self.config["ckpt_folder"], f"epoch{epoch}.jpg"), normalize = True)

    def train(self, start = 0):
        num_epochs = self.config["num_epochs"]
        assert os.path.exists(self.config["ckpt_folder"])
        self.resume_training()
        self.save_images(0)
        for epoch in range(start, start + num_epochs):
            self.train_one_epoch(epoch + 1)
            self.save_model(epoch + 1)
            self.save_images(epoch + 1)
