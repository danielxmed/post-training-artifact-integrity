import ptaie


def test_version_is_exposed() -> None:
    assert ptaie.__version__
    assert ptaie.ENV_VERSION.count(".") == 2
