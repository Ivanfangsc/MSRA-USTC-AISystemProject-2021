import nni.retiarii
import nni.retiarii.strategy
import nni.retiarii.evaluator.pytorch.lightning as pl
import torchvision
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import model

if __name__ == "__main__":
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = nni.retiarii.serialize(torchvision.datasets.CIFAR10,
                                           root='./data', train=True, download=True, transform=transform)

    test_dataset = nni.retiarii.serialize(torchvision.datasets.CIFAR10,
                                          root='./data', train=False, download=True, transform=transform)

    trainer = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=100),
                                val_dataloaders=pl.DataLoader(test_dataset, batch_size=100),
                                max_epochs=10)

    model = model.CNN(32, 3, 16, 10, 8)
    strategy = nni.retiarii.strategy.Random()

    exp = RetiariiExperiment(model, trainer, [], strategy)

    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'darts'
    exp_config.trial_concurrency = 1
    exp_config.max_trial_number = 10
    exp_config.trial_gpu_number = 1
    exp_config.training_service.use_active_gpu = True
    exp_config.training_service.gpu_indices = [0, 1]

    exp.run(exp_config, 8080)
