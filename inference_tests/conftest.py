import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--inference-stage', choices=['sampling', 'generation', 'cache', 'all'], default=None,
        help='Enable a student exercise stage; otherwise inference tests are skipped.',
    )


def pytest_collection_modifyitems(config, items):
    stage = config.getoption('--inference-stage')
    stages = {
        'test_sampling.py': 'sampling',
        'test_generation.py': 'generation',
        'test_cache_allocate.py': 'cache',
    }
    for item in items:
        if 'inference_tests' not in item.path.parts:
            continue
        if stage != 'all' and stage != stages.get(item.path.name):
            item.add_marker(pytest.mark.skip(reason='Enable this exercise with --inference-stage'))
