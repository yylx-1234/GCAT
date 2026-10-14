"""Dependency-light functional checks; no dataset or checkpoint is fabricated."""

import unittest

import numpy as np
import pandas as pd
import torch

from data.progressive_collate import progressive_collate_fn
from evaluation.calibration import fit_temperature, softmax_temperature
from evaluation.operational import decision_summary, lead_time_gain
from evaluation.policies import GRID, exhaustive_quality_policy_counts
from evaluation.sequential_selective_risk import clopper_pearson_upper, sequential_first_accept
from models.brandenburg_external import BrandenburgGCATRPTClassifier
from models.gcat_rp_t import GCATRPTClassifier
from models.time_encoding import calendar_day_of_year
from utils.config import load_config


class CoreTests(unittest.TestCase):
    def test_doy_and_padding(self):
        dates = torch.tensor([[20200229, 20200301, 0]])
        self.assertEqual(calendar_day_of_year(dates).tolist(), [[60, 61, 0]])
        sample = {
            "label": 0, "source_label": 1, "fold": 2, "patch_id": 1,
            "parcel_id": 10, "instance_id": 1, "pixel_count": 4,
            "cutoff_date": 20190430, "cutoff": "Apr30",
            "s2_features": torch.zeros(2, 20), "s2_dates": torch.tensor([20190401, 20190412]),
            "s2_quality": torch.ones(2, 4),
            "s1a_features": torch.zeros(1, 6), "s1a_dates": torch.tensor([20190403]),
            "s1d_features": torch.zeros(3, 6), "s1d_dates": torch.tensor([20190402, 20190409, 20190416]),
        }
        other = dict(sample)
        other.update(parcel_id=11, s2_features=torch.zeros(1, 20),
                     s2_dates=torch.tensor([20190401]), s2_quality=torch.ones(1, 4),
                     s1d_features=torch.zeros(1, 6), s1d_dates=torch.tensor([20190402]))
        batch = progressive_collate_fn([sample, other])
        self.assertEqual(batch["s2_lengths"].tolist(), [2, 1])
        self.assertEqual(batch["s2_padding_mask"].tolist(), [[False, False], [False, True]])
        self.assertEqual(batch["s2_dates"].tolist()[1], [20190401, 0])

    def test_forward_variable_lengths(self):
        torch.manual_seed(7)
        model = GCATRPTClassifier(
            s2_input_dim=20, s1_input_dim=6, projection_dim=128, hidden_dim=128,
            attention={"embed_dim": 128, "num_heads": 4, "dropout": 0.1},
            gate={"type": "complementary_channel", "input_dim": 256, "output_dim": 128},
            num_classes=18,
        ).eval()
        args = (
            torch.randn(2, 3, 20), torch.tensor([3, 1]),
            torch.tensor([[20190401, 20190411, 20190420], [20190405, 0, 0]]),
            torch.randn(2, 2, 6), torch.tensor([2, 1]),
            torch.tensor([[20190402, 20190415], [20190406, 0]]),
            torch.randn(2, 4, 6), torch.tensor([4, 2]),
            torch.tensor([[20190403, 20190408, 20190416, 20190425],
                          [20190404, 20190412, 0, 0]]),
        )
        with torch.no_grad():
            logits, gate = model(*args, return_gate=True)
            changed = list(args)
            changed[0] = args[0].clone()
            changed[0][1, 1:] = 10000
            changed[3] = args[3].clone()
            changed[3][1, 1:] = 10000
            changed[6] = args[6].clone()
            changed[6][1, 2:] = 10000
            masked_logits = model(*changed)
        self.assertEqual(tuple(logits.shape), (2, 18))
        self.assertEqual(tuple(gate.shape), (2, 128))
        self.assertTrue(torch.allclose(logits, masked_logits, atol=1e-5))
        self.assertTrue(bool(torch.all((gate >= 0) & (gate <= 1))))
        brandenburg = BrandenburgGCATRPTClassifier().eval()
        with torch.no_grad():
            output = brandenburg(args[0], args[1], args[2],
                                 torch.randn(2, 2, 4), args[4], args[5],
                                 torch.randn(2, 4, 4), args[7], args[8])
        self.assertEqual(tuple(output.shape), (2, 9))

    def test_first_accept_lock_and_metrics(self):
        frames = [pd.DataFrame({"parcel_id": [1, 2], "label": [0, 1],
                                "checkpoint": ["Apr30", "Apr30"]}),
                  pd.DataFrame({"parcel_id": [1, 2], "label": [0, 1],
                                "checkpoint": ["May31", "May31"]})]
        probabilities = [np.array([[0.9, 0.1], [0.6, 0.4]]),
                         np.array([[0.1, 0.9], [0.1, 0.9]])]
        outcome, summary = decision_summary(frames, probabilities, 0.8)
        self.assertEqual(outcome["accepted_index"].tolist(), [0, 1])
        self.assertEqual(outcome["accepted_label"].tolist(), [0, 1])
        self.assertEqual(summary["n_accept"], 2)
        self.assertAlmostEqual(summary["mcac"], 0.75)
        self.assertAlmostEqual(summary["empirical_selective_risk"], 0.0)
        self.assertAlmostEqual(clopper_pearson_upper(0, 10, 0.05), 1 - 0.05 ** 0.1)
        self.assertEqual(softmax_temperature(np.array([[1.0, 0.0]]), 1.0).shape, (1, 2))

    def test_lead_time_gain_requires_aligned_ids(self):
        frames = [pd.DataFrame({"parcel_id": [1], "label": [0]}),
                  pd.DataFrame({"parcel_id": [1], "label": [0]})]
        s2 = sequential_first_accept(frames, [np.array([[0.6, 0.4]]), np.array([[0.9, 0.1]])], 0.8)
        gcat = sequential_first_accept(frames, [np.array([[0.9, 0.1]]), np.array([[0.9, 0.1]])], 0.8)
        gain = lead_time_gain(s2, gcat, ["2019-04-30", "2019-05-31"])
        self.assertEqual(gain["mean_gain_days"], 31.0)

    def test_temperature_and_primary_candidate_counts(self):
        logits = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
        temperature = fit_temperature(logits, np.array([0, 1, 0]))
        self.assertTrue(np.isfinite(temperature) and temperature > 0)
        frames = [pd.DataFrame({"parcel_id": [1, 2], "label": [0, 1],
                                "quality_bin": ["Q1", "Q4"]}),
                  pd.DataFrame({"parcel_id": [1, 2], "label": [0, 1],
                                "quality_bin": ["Q1", "Q4"]})]
        probabilities = [np.array([[0.70, 0.30], [0.55, 0.45]]),
                         np.array([[0.20, 0.80], [0.05, 0.95]])]
        accepted, errors = exhaustive_quality_policy_counts(frames, probabilities)
        choice = (20, 0, 0, 30)  # Q1=0.70, Q4=0.80
        direct = sequential_first_accept(frames, probabilities, {
            "Q1": float(GRID[choice[0]]), "Q2": float(GRID[choice[1]]),
            "Q3": float(GRID[choice[2]]), "Q4": float(GRID[choice[3]]),
        })
        self.assertEqual(int(accepted[choice]), int(direct["accepted"].sum()))
        self.assertEqual(int(errors[choice]), int(direct["wrong"].sum()))

    def test_configs(self):
        pastis = load_config("configs/pastis_fold2.yaml")
        brandenburg = load_config("configs/brandenburg_2018_2019.yaml")
        self.assertEqual(pastis["data"]["train_folds"], [3, 4, 5])
        self.assertEqual(pastis["data"]["test_folds"], [2])
        self.assertIn("2019", brandenburg["data"]["splits"]["test"])
        self.assertNotIn("2019", brandenburg["data"]["splits"]["train"])


if __name__ == "__main__":
    unittest.main()
