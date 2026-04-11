from txtcaptcha import fit_model, decrypt


def test_fit_model_runs_few_epochs(tmp_labeled_dir):
    model, history = fit_model(
        tmp_labeled_dir,
        epochs=2,
        batch_size=4,
        prop_valid=0.25,
        verbose=False,
    )
    assert len(history.train_loss) == 2
    assert len(history.valid_loss) == 2
    # decrypt should run end-to-end
    files = sorted(tmp_labeled_dir.glob("*.png"))
    out = decrypt(files[:2], model)
    assert len(out) == 2
