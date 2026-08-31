import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Run the async tests on asyncio only; the server has no trio-specific code."""
    return "asyncio"
