import torch

from bpm.model import CNNBiLSTM


def test_forward_shape_matches_batch_size():
    model = CNNBiLSTM()
    x = torch.randn(4, 1, 1000)

    out = model(x)

    assert out.shape == (4,)


from bpm.model import N_BINS, argmax_bpm, decode_bpm, dist_std, gaussian_targets


def test_head_output_shapes():
    x = torch.randn(3, 1, 1000)
    assert CNNBiLSTM(head="reg")(x).shape == (3,)
    assert CNNBiLSTM(head="dist")(x).shape == (3, N_BINS)
    assert CNNBiLSTM(head="gauss")(x).shape == (3, 2)


def test_default_state_dict_loads_into_reg():
    sd = CNNBiLSTM().state_dict()
    assert sd["head.3.weight"].shape == (1, 32)
    CNNBiLSTM(head="reg").load_state_dict(sd)


def test_gaussian_targets_sum_one_and_peak_at_label():
    t = gaussian_targets(torch.tensor([30.0, 72.0, 220.0]))
    assert t.shape == (3, N_BINS)
    assert torch.allclose(t.sum(1), torch.ones(3))
    assert t.argmax(1).tolist() == [0, 42, 190]


def test_decode_of_target_log_is_label():
    labels = torch.tensor([45.0, 72.0, 150.0])
    out = decode_bpm(gaussian_targets(labels).log())
    assert torch.allclose(out, labels, atol=0.05)
    assert torch.allclose(argmax_bpm(gaussian_targets(labels)), labels)


def test_decode_prior_floor_finite_and_tau0_equals_plain():
    logits = gaussian_targets(torch.tensor([60.0, 100.0])).log()
    counts = torch.zeros(N_BINS)
    counts[70] = 500.0
    log_prior = ((counts + 1) / (counts.sum() + N_BINS)).log()
    assert torch.isfinite(decode_bpm(logits, log_prior=log_prior, tau=1.0)).all()
    assert torch.equal(decode_bpm(logits, log_prior=log_prior, tau=0), decode_bpm(logits))


def test_dist_std_wider_for_flat():
    sharp = gaussian_targets(torch.tensor([80.0])).log()
    flat = torch.zeros(1, N_BINS)
    assert dist_std(flat) > dist_std(sharp)
    assert abs(dist_std(sharp).item() - 2.0) < 0.1
