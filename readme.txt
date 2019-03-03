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
  * data_operations.py
  * display_operations.py
  * id3_operations.py
  * bayes_operations.py
  * spam_bayes.py
  * spam_id3.py
  * ctg_bayes.py
  * ctg_id3.py
  * requirements.txt
  * spambase.data
  * CTG.csv

  NOTE: The respective scripts are configured to use spambase.data and CTG.csv from the present directory as a default.
        I have included the datasets in the .zip, so the scripts should work out of the box.

==== General Overview ====
  That files that have a suffix of operations are all shared functions used by various scripts.

  The scripts that actually implement the classifications are:
    * spam_bayes.py
    * spam_id3.py
    * ctg_bayes.py
    * ctg_id3.py

    The prefix of these files indicate what data set is used, i.e. spam for spambase.data and ctg for CTG.csv
    The suffix of these files indicate which classifier is used.


==== spam_bayes.py ====
    Implements Gaussian Naive Bayes classificaiton on the spambase.data dataset.

    To execute, type "python3 spam_bayes.py"



==== data_operations.py ====
  Contains shared functions to manipulate datasets, such as standardizing or splitting into training and testing sets.
  Used by spam_bayes.py, spam_id3.py, ctg_bayes.py, ctg_id3.py

==== display_operations.py ===
  Contains shared functions used for displaying classification metrics to the console.

  For binary classification:
    * Precision
    * Recall
    * F-measure
    * Accuracy

   For multiclass classificaiton:
    * Accuracy

    Used by spam_bayes.py, spam_id3.py, ctg_bayes.py, ctg_id3.py

==== id3_operations ===
  Contains shared functions used for ID3 classification.
  Used by spam_id3.py, ctg_id3.py
