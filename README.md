# MalariaCellClassification
CAP 5516 Spring 2022 Final Project -- classifying a highly imbalanced set of malaria cells into the proper malaria life-cycle type

This code utilizes Pytorch and multiple other packages that need to be installed.

To run the code, the user simply needs to run the train.py python file*.

- In the command line, navigate to the appropriate directory, then type "python train.py". Or, run from a programming environment like PyCharm.
- Starting at line #397, the command line argument parser is found in train.py. The user can either chose to change these default values or use the appropriate commands while running.
- Example: to change the batch size from default 32 to 16, type "python train.py -b 16"

*NOTE: the dataset is not included in this GitHub repository. It needs to be installed from: https://github.com/QaziAmmar/A-Dataset-and-Benchmark-for-Malaria-Life-Cycle-Classification-in-Thin-Blood-Smear-Images.
Then, run plot_annotations.py to crop the individual cells from the blood slide images.
