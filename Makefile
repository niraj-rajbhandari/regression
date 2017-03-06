.PHONY: all debug clean menu help

all:
ifdef $(data_set)
	python main.py -d $(data_set)
else
	python main.py
endif

debug:
ifdef $(data_set)
	python -m pdb main.py -d $(data_set)  -m
else
	python -m pdb main.py -m
endif

menu:
ifdef $(data_set)
	python main.py -d $(data_set) -m
else
	python main.py -m
endif

help:
	python main.py -h

clean:
	rm *.pyc
