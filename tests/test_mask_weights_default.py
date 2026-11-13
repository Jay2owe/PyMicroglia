"""A dropped folder masks with the accepted base unless someone names another."""

import hashlib

import pytest

from pymicroglia import mask_methods
from pymicroglia.learned_mask import apply as masking


@pytest.fixture
def accepted(tmp_path, monkeypatch):
    monkeypatch.delenv(masking.WEIGHTS_VARIABLE, raising=False)
    path = tmp_path / "runs" / "stage3_disguise" / "model.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"accepted weights")
    monkeypatch.setattr(masking, "ACCEPTED_RUN", path)
    return path


def test_unset_uses_the_accepted_base_and_records_it(accepted):
    assert masking.weights() == accepted
    reference = mask_methods.model_reference(None, default=True)
    assert reference == {"weights": str(accepted),
                         "weights_sha256": hashlib.sha256(b"accepted weights").hexdigest()}


def test_a_named_run_wins_over_the_accepted_base(accepted, tmp_path, monkeypatch):
    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    monkeypatch.setenv(masking.WEIGHTS_VARIABLE, str(other))
    assert masking.weights() == other


def test_without_the_checkout_copy_it_still_asks(accepted, monkeypatch):
    monkeypatch.setattr(masking, "ACCEPTED_RUN", accepted.with_name("absent.pt"))
    with pytest.raises(FileNotFoundError, match=masking.WEIGHTS_VARIABLE):
        masking.weights()
    assert mask_methods.model_reference(None, default=True) == {"weights": None}


def test_the_checkout_path_is_the_run_the_readme_accepts():
    assert masking.ACCEPTED_RUN.parts[-2:] == ("stage3_disguise", "model.pt")
    assert masking.ACCEPTED_RUN.parents[4].name == "development"
