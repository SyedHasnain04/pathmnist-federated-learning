import os
import sys
import copy
import math
import torch
import torch.nn as nn
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("PFLlib/system"))

from PFLlib.system.flcore.trainmodel.models import FedAvgCNN
from metrics_evaluator import MetricsEvaluator, compute_update_metrics, flatten_trainable_params

def test_evaluator_split_counts_and_parameter_immutability():
    evaluator = MetricsEvaluator()
    model = FedAvgCNN(in_features=3, num_classes=9, dim=1024)
    
    # Save parameter snapshot before evaluation
    params_before = [p.clone() for p in model.parameters()]
    
    val_acc, val_loss = evaluator.evaluate_official_val(model)
    test_acc, test_loss = evaluator.evaluate_official_test(model)
    
    assert 0.0 <= val_acc <= 1.0
    assert val_loss > 0.0
    assert 0.0 <= test_acc <= 1.0
    assert test_loss > 0.0
    
    # Check parameters were not modified
    for pb, pa in zip(params_before, model.parameters()):
        assert torch.equal(pb, pa), "Evaluation mutated model parameters!"
        
    cm = evaluator.compute_confusion_matrix(model)
    assert cm.shape == (9, 9)
    assert cm.sum() == 7180, f"Expected 7180 test samples in confusion matrix, got {cm.sum()}"

def test_update_metrics_and_zero_norm_handling():
    m_start = nn.Linear(4, 2, bias=False)
    m_start.weight.data = torch.zeros(2, 4)
    
    m1 = nn.Linear(4, 2, bias=False)
    m1.weight.data = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    
    m2 = nn.Linear(4, 2, bias=False)
    m2.weight.data = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    
    m3_zero = nn.Linear(4, 2, bias=False)
    m3_zero.weight.data = torch.zeros(2, 4)  # Zero update!
    
    sample_counts = [100, 100, 100]
    metrics = compute_update_metrics([m1, m2, m3_zero], m_start, sample_counts)
    
    assert math.isclose(metrics["norms"][0], 1.0, rel_tol=1e-5)
    assert math.isclose(metrics["norms"][1], 1.0, rel_tol=1e-5)
    assert math.isclose(metrics["norms"][2], 0.0, abs_tol=1e-7)
    
    # Zero norm client must have NaN cosine, not 0.0!
    assert math.isnan(metrics["standard_cosines"][2]), "Zero norm client standard cosine must be NaN"
    assert math.isnan(metrics["leave_one_out_cosines"][2]), "Zero norm client LOO cosine must be NaN"
    
    # Check LOO cosine between m1 and m2: m1 update is [1,0], m2 is [0,1] -> orthogonal, cosine = 0.0
    assert math.isclose(metrics["leave_one_out_cosines"][0], 0.0, abs_tol=1e-5)
    assert math.isclose(metrics["leave_one_out_cosines"][1], 0.0, abs_tol=1e-5)
    
    # Standard cosine includes own contribution:
    # delta_global = 1/3 [1,0] + 1/3 [0,1] = [1/3, 1/3]. Norm = sqrt(2)/3.
    # cos(m1, delta_global) = (1/3) / (1 * sqrt(2)/3) = 1/sqrt(2) = 0.7071
    assert math.isclose(metrics["standard_cosines"][0], 1.0 / math.sqrt(2), rel_tol=1e-4)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

