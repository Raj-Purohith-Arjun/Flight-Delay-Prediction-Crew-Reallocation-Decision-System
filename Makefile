.PHONY: setup data features train evaluate simulate test

setup:
python -m pip install -r requirements.txt

data:
PYTHONPATH=. python -m src.utils.cli data

features:
PYTHONPATH=. python -m src.utils.cli features

train:
PYTHONPATH=. python -m src.utils.cli train

evaluate:
PYTHONPATH=. python -m src.utils.cli evaluate

simulate:
PYTHONPATH=. python -m src.utils.cli simulate

test:
PYTHONPATH=. pytest -q
