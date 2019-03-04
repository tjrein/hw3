==== CS613 ====
  Author: Tom Rein
  Email: tr557@drexel.edu

==== Dependencies ====
  * python3
  * pip3
  * numpy

  In the event that dependencies are not installed, I have provided a "requirements.txt" file.
  To install dependencies, type "pip3 install -r requirements.txt"
  That being said, the only real dependency is numpy.

==== Files Included ====
  * spam_bayes.py
  * spam_id3.py
  * ctg_bayes.py
  * ctg_id3.py

  * bayes_operations.py
  * id3_operations.py
  * data_operations.py
  * display_operations.py

  * requirements.txt
  * spambase.data
  * CTG.csv

  NOTE: The classification scripts are configured to use spambase.data and CTG.csv from the present directory by default.
        I have included the datasets in the .zip, so the scripts should work out of the box.

==== General Overview ====
  That files that have an "operations" suffix are all shared functions used by various scripts.

  The scripts that actually implement the classifications are:
    * spam_bayes.py
    * spam_id3.py
    * ctg_bayes.py
    * ctg_id3.py

    The prefix of these files indicate what data set is used, i.e. 'spam' for spambase.data and 'ctg' for CTG.csv
    The suffix of these files indicate which classifier is used.


==== spam_bayes.py ====
    * Reads in data from spambase.data
    * Performs an initial pass on the data and filters out features with a standard deviation < 0.1
    * Implements Gaussian Naive Bayes classification on the test data.
    * Outputs Precision, Recall, F-measure and Accuracy

    To execute:
      > python3 spam_bayes.py

    Alternatively, an optional argument can be passed to specify the path to a file:
      > python3 spam_bayes.py {path_to_file}

    By default, this script will use './spambase.data' as the path if no argument is passed


==== spam_id3.py ====
  * Reads in data from spambase.data
  * Performs an initial pass on the data and filters out features with a standard deviation < 0.1
  * Builds an ID3 decision tree and classifies test data.
  * Outputs Precision, Recall, F-measure, and Accuracy

  To execute:
    > python3 spam_id3.py

  Alternatively, an optional argument can be passed to specify the path to a file:
    > python3 spam_id3.py {path_to_file}

  By default, this script will use './spambase.data' as the path if no argument is passed


==== ctg_bayes.py ====
  * Reads in data from CTG.csv, ignoring first two header rows and removing 2nd to last column
  * Performs an initial pass on the data and filters out features with a standard deviation < 0.1
  * Implements Gaussian Naive Bayes classification on the test data.
  * Outputs Accuracy

  To execute:
    > python3 ctg_bayes.py

  Alternatively, an optional argument can be passed to specify the path to a file:
    > python3 ctg_bayes.py {path_to_file}

  By default, this script will use './CTG.csv' as the path if no argument is passed


==== ctg_id3.py ====
  * Reads in data from CTG.csv, ignoring first two header rows and removing 2nd to last column
  * Performs an initial pass on the data and filters out features with a standard deviation < 0.1
  * Builds an ID3 decision tree and classifies the test data.
  * Outputs Accuracy

  To execute:
    > python3 ctg_id3.py

  Alternatively, an optional argument can be passed to specify the path to a file:
    > python3 ctg_id3.py {path_to_file}

  By default, this script will use './CTG.csv' as the path if no argument is passed


==== bayes_operations.py ====
  Contains shared functions for Gaussian Naive Bayes Classification.
  The functions are generalized to work on both binary and multiclass datasets


==== id3_operations.py ====
  Contains shared functions for constructing ID3 decision trees.
  The functions are generalized to work on both binary and multiclass datasets.


==== data_operations.py ====
  Contains shared functions to manipulate datasets, such as standardizing or splitting into training and testing sets.


==== display_operations.py ===
  Contains shared functions used for displaying classification metrics to the console.
