@echo off
setlocal enabledelayedexpansion

rem Set the number of iterations
set iterations=3

rem Run the command multiple times and wait for each to finish
for /l %%i in (1,1,%iterations%) do (
    echo Running iteration %%i
    set /a seed=600+%%i
    set command=python scripts/population_evolution.py --num_envs 1024 --task Grasp-Objects --generations 30 --seed !seed!
    !command!
)

endlocal