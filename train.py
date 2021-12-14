from src.trainer import Trainer
from config import CONFIG


if __name__ == "__main__":
    trainer = Trainer(config = CONFIG)
    trainer.train()