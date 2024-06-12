::**********************************************************************************************
::
::  @brief       Use this script to automatically create or update the python environment
::				 for sample_project and installs all the necessary packages.
::			     The python environment is created at this project level in the .venv folder.
::                  
::               Prerequisites:
::
::               Python 3.7.9 64 bit has to be installed:
::               https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe
::				 > "Install Now" and set "Add Python 3.7 to PATH"
::
::				 Usage: pythonCreateEnv.cmd <ignore_pause>
::				 - <ignore_pause>: 0: Does finish script with pause (default), 1: without pause
::
::---------------------------------------------------------------------------------------------
::
::  @author      Jonas Hofstetter, Andreas Caduff
::
::  @copyright   CC Intelligent Sensors and Networks 
::               Lucerne University of Applied Sciences
::               and Arts T&A, Switzerland.
::
::**********************************************************************************************
echo off
SETLOCAL EnableDelayedExpansion

::**********************************************************************************************

set ignore_pause=%1
if not defined ignore_pause set ignore_pause=0

echo Start creating python environment and install packages listed in Pipfile
echo ************************************************************************
echo ^> Install package pipenv
call pip install pipenv==2020.11.15
echo ^> Set user variable PIPENV_MAX_DEPTH to 100
call setx PIPENV_MAX_DEPTH 100
call setx PIPENV_VENV_IN_PROJECT 1

echo ^> Create and clean (uninstall not needed packages) environment in .venv from Pipfile.lock
pipenv install --ignore-pipfile
pipenv clean

if %ERRORLEVEL% NEQ 0 (
	echo ^> Error: Package installation failed.
	pause
	exit(1)
)

echo ^> Packages successfully installed.
if %ignore_pause% NEQ 1 (
	pause
)
exit(0)