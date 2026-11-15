import pytest
import torch

from contextad.models import (
    LOGVAR_MAX,
    LOGVAR_MIN,
    ContextNet,
    MaskedMLP,
    count_parameters,
    gaussian_nll,
)

DIM = 6


def batch(b=3, t=4, k=7, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(b, t, DIM, generator=gen)
    context = torch.randn(b, k, DIM, generator=gen)
    avail = torch.ones(b, DIM, dtype=torch.bool)
    avail[1, 4:] = False
    obs = torch.rand(b, t, DIM, generator=gen) < 0.5
    obs &= avail[:, None, :]
    return x, obs, context, avail


def test_output_shapes():
    x, obs, context, avail = batch()
    mean, logvar = ContextNet(DIM)(x, obs, context, avail)
    assert mean.shape == logvar.shape == x.shape
    mean, logvar = MaskedMLP(DIM)(x, obs)
    assert mean.shape == logvar.shape == x.shape


def test_log_variance_is_clamped():
    x, obs, context, avail = batch()
    net = ContextNet(DIM)
    with torch.no_grad():
        net.head.bias[DIM:] = 100.0
    assert net(x, obs, context, avail)[1].max() == LOGVAR_MAX
    with torch.no_grad():
        net.head.bias[DIM:] = -100.0
    assert net(x, obs, context, avail)[1].min() == LOGVAR_MIN


@pytest.mark.parametrize("model", [ContextNet(DIM), MaskedMLP(DIM)])
def test_hidden_values_do_not_reach_the_prediction(model):
    torch.manual_seed(0)
    x, obs, context, avail = batch()
    changed = x.clone()
    changed[~obs] = 100.0
    args = (context, avail) if isinstance(model, ContextNet) else ()
    a = model(x, obs, *args)
    b = model(changed, obs, *args)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_columns_missing_from_the_schema_do_not_reach_the_prediction():
    torch.manual_seed(1)
    net = ContextNet(DIM)
    x, obs, context, avail = batch()
    changed = context.clone()
    changed[1][:, ~avail[1]] = 50.0
    a = net(x, obs, context, avail)
    b = net(x, obs, changed, avail)
    assert torch.allclose(a[0], b[0], atol=1e-6)


def test_context_order_does_not_matter():
    torch.manual_seed(2)
    net = ContextNet(DIM)
    x, obs, context, avail = batch(k=9)
    shuffled = context[:, torch.randperm(9)]
    a = net(x, obs, context, avail)
    b = net(x, obs, shuffled, avail)
    assert torch.allclose(a[0], b[0], atol=1e-5)
    assert torch.allclose(a[1], b[1], atol=1e-5)


def test_context_changes_the_prediction_but_none_is_allowed():
    torch.manual_seed(3)
    net = ContextNet(DIM)
    x, obs, context, avail = batch()
    with_context = net(x, obs, context, avail)[0]
    other = net(x, obs, context + 1.0, avail)[0]
    assert not torch.allclose(with_context, other, atol=1e-4)
    without = net(x, obs)[0]
    empty = net(x, obs, context[:, :0], avail)[0]
    assert torch.equal(without, empty)


def test_episodes_in_a_batch_are_independent():
    torch.manual_seed(4)
    net = ContextNet(DIM)
    x, obs, context, avail = batch()
    together = net(x, obs, context, avail)[0]
    alone = net(x[1:2], obs[1:2], context[1:2], avail[1:2])[0]
    assert torch.allclose(together[1:2], alone, atol=1e-5)


def test_targets_do_not_see_each_other():
    torch.manual_seed(5)
    net = ContextNet(DIM)
    x, obs, context, avail = batch()
    changed = x.clone()
    changed[:, 1] = 9.0
    a = net(x, obs, context, avail)[0]
    b = net(changed, obs, context, avail)[0]
    assert torch.allclose(a[:, 0], b[:, 0], atol=1e-6)


def test_parameter_counts_are_small():
    assert 100_000 < count_parameters(ContextNet(10)) < 300_000
    assert 100_000 < count_parameters(MaskedMLP(10)) < 300_000


def test_gaussian_nll_value_and_mask():
    mean = torch.zeros(1, 2, 3)
    logvar = torch.zeros(1, 2, 3)
    x = torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]])
    hidden = torch.tensor([[[True, False, False], [False, False, True]]])
    assert torch.isclose(gaussian_nll(mean, logvar, x, hidden), torch.tensor(0.25))
    logvar = torch.full((1, 2, 3), 2.0)
    expected = (0.5 * (2.0 + 1.0 * torch.exp(torch.tensor(-2.0))) + 0.5 * 2.0) / 2
    assert torch.isclose(gaussian_nll(mean, logvar, x, hidden), expected)
