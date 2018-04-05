# SVBM on Python
The SVBM module is compiled for Python 3.6.
It implements the variational EM of Pierre Latouche. His paper can be found [here](https://drive.google.com/open?id=1TH90r7auLsqnAXUpRYTLH8PKmaFpXvkw).

To use it, you just need to add one file to your directory :
* For Linux, Mac OS X : SVBM.cpython-36m-x86_64-linux-gnu.so
* For Windows : 

This file are in the folder [module](https://github.com/Nicotous1/Enron/blob/master/module/).<br>
If this files do not work you can try the notebook : "Try with Cython.ipynb"

There is a wiki to explain how the module work.

The module was developped for the study of the Enron Scandal.<br>
There are two notebook related :
* Data_Cleaning.ipynb : Clean and reduced the size of the original data from [here](http://www.ahschulz.de/enron-email-data/) .<br>
You need to have the database within csv file to execute this one.

* ScandalPerson.ipynb : Descriptive statistics about the different type of traffic<br>
You need a folder named "data" at your root filled with [this files](https://drive.google.com/open?id=1O3YPJKMkcAz11q_7xz0X-W_Xt1q_EojT).
