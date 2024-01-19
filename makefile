venv:
	python -m venv .venv
	./ venv/bin/activate;pip install -r requirements.txt

clean:
	rm ml/out/*
	rm ml/data/*