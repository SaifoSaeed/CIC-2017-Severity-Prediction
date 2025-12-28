.PHONY: all dex save-deps deps help no-args train_all neural xgb lr

SHELL = cmd.exe

MODELS_DIR = models
OUTPUT_DIR = $(MODELS_DIR)/weights
DRIVE  = $(MODELS_DIR)/drive.py

all: dex train_all
	
dex:
	python dex.py -a -cu -dx -dp -fe

save-deps:
	uv pip freeze > requirements.txt

deps:
	uv pip install -r requirements.txt

help:
	python dex.py -h
	python $(DRIVE) -h

no-args:
	python dex.py

train_all:
	python $(DRIVE) -l -x -n -o $(OUTPUT_DIR)

neural:
	python $(DRIVE) -n -o $(OUTPUT_DIR)

xgb:
	python $(DRIVE) -x -o $(OUTPUT_DIR)

lr:
	python $(DRIVE) -l -o $(OUTPUT_DIR)