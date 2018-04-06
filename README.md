# SVBM on Python
The SVBM module is compiled for Python 3.6. It implements the variational EM of Pierre Latouche.<br>
His paper can be found [here](https://drive.google.com/open?id=1TH90r7auLsqnAXUpRYTLH8PKmaFpXvkw).<br>
There is a [wiki](https://github.com/Nicotous1/Enron/wiki) to explain how to use the module.<br>

To import it, you just need to add one file from this [folder](https://github.com/Nicotous1/Enron/blob/master/module/) to your directory :
* For Linux, Mac OS X : [SVBM.cpython-36m-x86_64-linux-gnu.so](https://github.com/Nicotous1/Enron/blob/master/module/SVBM.cpython-36m-x86_64-linux-gnu.so)
* For Windows : [SVBM.cp36-win_amd64.pyd](https://github.com/Nicotous1/Enron/blob/master/module/SVBM.cp36-win_amd64.pyd)

Then ```from SVBM import *``` or execute [the main.py](https://github.com/Nicotous1/Enron/blob/master/module/main.py) 

If this files do not work you can try the notebook : [Try with Cython.ipynb](https://github.com/Nicotous1/Enron/blob/master/Try%20with%20Cython.ipynb) with "enron_network.npy".


The module was developped for the study of the Enron Scandal.<br>
There are three notebooks related :
* The main notebook, containing the analysis presented in the report, can be accessed [here](https://drive.google.com/open?id=1nc1Y1kL37SJtyRtZMlHj6jmG9B23FqsW). It contains a notebook allowing to compute the adjency matrix and to display the network. All dataset are already in the folder. Make sure to change the path variable to be able to execute the notebook properly.

* Two notebooks that are draft to study the network.
  * [Data_Cleaning.ipynb](https://github.com/Nicotous1/Enron/blob/master/Data_Cleaning.ipynb) : Cleaning and reducing size of the original data from [here](http://www.ahschulz.de/enron-email-data/) .<br>
You need to have the database within csv files to execute this one.

  * [ScandalPerson.ipynb](https://github.com/Nicotous1/Enron/blob/master/ScandalPerson.ipynb) : Descriptive statistics about the different type of traffic<br>
You need a folder named "data" at your root filled with [this files](https://drive.google.com/open?id=1O3YPJKMkcAz11q_7xz0X-W_Xt1q_EojT).


