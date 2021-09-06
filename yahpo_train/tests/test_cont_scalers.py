def test_cont_scaler():
    x = torch.random(100)

    cr = ContTransformerLogRange(x)
    x_tf = cr(x)
    x_rec = cr.invert(x_tf)

    assert torch.mean(torch.abs(diff[mask])) / torch.max(xs[mask]) < lim
    assert xsn.shape == xs.shape