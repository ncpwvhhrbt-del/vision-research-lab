"""Training engine tests on synthetic tensors."""

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.engine import evaluate, train_one_epoch


class TrainingEngineTests(unittest.TestCase):
    def test_evaluate_matches_full_dataset_mean_loss(self) -> None:
        logits = torch.tensor(
            [[3.0, 0.0], [0.0, 3.0], [1.0, 2.0]],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 1, 0])
        loader = DataLoader(TensorDataset(logits, targets), batch_size=2)
        criterion = nn.CrossEntropyLoss()

        metrics = evaluate(
            nn.Identity(),
            loader,
            criterion,
            torch.device("cpu"),
        )
        expected_loss = criterion(logits, targets).item()

        self.assertAlmostEqual(metrics.loss, expected_loss, places=6)
        self.assertEqual(metrics.samples, 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_train_moves_images_to_cuda(self) -> None:
        images = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(TensorDataset(images, targets), batch_size=2)
        model = nn.Linear(2, 2).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        metrics = train_one_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            optimizer,
            torch.device("cuda"),
        )

        self.assertEqual(metrics.samples, 4)


if __name__ == "__main__":
    unittest.main()
