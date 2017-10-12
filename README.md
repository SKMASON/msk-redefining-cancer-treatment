# msk-redefining-cancer-treatment

1.Requirements

System: Linux

Version of Python : 2.7

Modules:

	yaml==3.12	
	numpy==1.13.0
	pandas==0.18.1
	tqdm==4.14.0
	sklearn==0.18.2
	gensim==2.3.0
	keras==2.0.4
	nltk==3.2.4

2.Getting started

Firstly install required modules:

	pip install -r requirements.txt

Run the scripts:

	python 1_genetreat.py
	python 2_LSTM.py
	python 3_XBG.py
	python #4_FinalStep.py

Remarks: 

	The default paths of all the files are the same for every script
	Current configuration of model parameters achieved a score of 0.06~0.12 on the public leaderboard. Parameters can be changed as required
