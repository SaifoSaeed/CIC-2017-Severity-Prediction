.PHONY: all deps save-deps help aggregate drop-extra clear-unique 

SHELL = cmd.exe

all:
	python dex.py -a -cu -dx -dp
#	python dex.py -cu -dx -dp

save-deps:
	uv pip freeze > requirements.txt

deps:
	uv pip install -r requirements.txt

help:
	python dex.py -h

aggregate: 
	python dex.py -a

clear-unique:
	python dex.py -cu

drop-extra:
	python dex.py -dx

drop-perfect:
	python dex.py -dp

no-args:
	python dex.py