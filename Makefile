install:
	pip install --upgrade pip
	pip install -r requirements.txt

lint:
	flake8 src/ tests/ --max-line-length=100

test:
	pytest -v

train:
	python -m src.train

clean:
	del /S /Q *.pyc 2>nul
	del /S /Q .pytest_cache\* 2>nul