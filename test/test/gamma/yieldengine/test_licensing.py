import importlib
import os

import pytest
import rsa

from gamma.yieldengine import (
    LICENSE_CLIENT,
    LICENSE_KEY,
    LICENSE_KEY_SIG,
    retrieve_license,
    safe_dumps,
)


def _reimport_yield_engine() -> None:
    import gamma.yieldengine

    importlib.reload(gamma.yieldengine)


def test_no_license_warns() -> None:
    with pytest.warns(expected_warning=UserWarning):
        # have to ensure it's not cached already
        _reimport_yield_engine()


def test_valid_license() -> None:
    from gamma.yieldengine import LICENSED_FOR

    assert LICENSED_FOR == "UNLICENSED"
    (pubkey, privkey) = rsa.newkeys(512)
    client = "low yield client"
    signature = rsa.sign(client.encode("ASCII"), privkey, "SHA-1")

    # activate the test-license through the environment
    os.environ[LICENSE_KEY] = safe_dumps(pubkey)
    os.environ[LICENSE_KEY_SIG] = safe_dumps(signature)
    os.environ[LICENSE_CLIENT] = client

    # test license retrieval on the other end:
    ret_key, ret_sig, ret_client = retrieve_license()

    # assert client name is equal
    assert ret_client == client
    assert ret_key == pubkey
    assert ret_sig == signature

    _reimport_yield_engine()
    from gamma.yieldengine import LICENSED_FOR

    assert LICENSED_FOR == client
