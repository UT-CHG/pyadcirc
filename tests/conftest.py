"""
    conftest.py for pyadcirc.
"""
import pytest
import tempfile
import os

@pytest.fixture
def temp_file_stream():
    file_stream = tempfile.TemporaryFile(mode='w+')
    yield file_stream
    file_stream.close()

@pytest.fixture
def temp_file_path(request):
    file_path = tempfile.mkstemp()[1]
    yield file_path
    if os.path.exists(file_path):
        os.remove(file_path)
