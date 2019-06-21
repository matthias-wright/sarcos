# SARCOS regression

## Description
I applied four different regression models; Nearest neighbor, Linear regression, Regression forests, and Gaussian Processes. The data originates from the [SARCOS dataset](http://gaussianprocess.org/gpml/data/), which consists of 21 features (arising from seven joint positions, seven joint velocities, and seven joint accelerations). The goal is to estimate the first of the seven corresponding joint torques. 

## Instructions
In order to execute a model, run main.py file.

## Results
|                    | Training error [RMSE] | Test error [RMSE] |
| :---               |         :---:         |       :---:       |
| Nearest Neighbors  |           -           |       3.3386      |
| Linear Regression  |        5.5527         |       5.4600      |
| Regression forests |        4.1356         |       4.6956      |
| Gaussian Processes |           -           |       5.4386      |

## Dependencies
* [Python (Anaconda 3.6.5)](https://anaconda.org/) 
* [NumPy (1.15.4)](http://www.numpy.org/) 
* [Matplotlib (3.0.2)](https://matplotlib.org/)

## License
This project is licensed under the [MIT Licence](https://choosealicense.com/licenses/mit/)
