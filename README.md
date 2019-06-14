# SARCOS regression

## Description
I applied four different regression models; Nearest neighbor, Linear regression, Regression forests, and Gaussian Processes. The data originates from the SARCOS dataset (http://gaussianprocess.org/gpml/data/), which consists of 21 features (arising from seven joint positions, seven joint velocities, and seven joint accelerations). The goal is to estimate the first of the seven corresponding joint torques. 

## Results
|                    | Training error [RMSE] | Test error [RMSE] |
| :---               |         :---:         |       :---:       |
| Nearest Neighbors  |           -           |       3.3386      |
| Linear Regression  |        5.5527         |       5.4600      |
| Regression forests |        4.1356         |       4.6956      |
| Gaussian Processes |           -           |       5.4386      |

## Instructions
In order to execute a model, run main.py file.

## Dependencies
* [Python (Anaconda 3.6.5)](https://anaconda.org/) 
* [NumPy (1.15.4)](http://www.numpy.org/) 
* [Matplotlib (3.0.2)](https://matplotlib.org/)

## Hardware
* 8 * Intel Core i7-8550U CPU @ 1.80GHz
* 16 GiB RAM
* Manjaro Linux 17.1 (Kernel: 4.14.91-1-MANJARO)
