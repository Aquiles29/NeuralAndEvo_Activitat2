Codes to execute different experiments:

python -m src.run_experiments --dataset data/raw/myciel3.col --k 5
python -m src.run_experiments --dataset data/raw/queen7_7.col --k 9
python -m src.run_experiments --dataset data/raw/fpsol2.i.1.col --k 80

Codes to execute the graphics:
python -m src.plot_best --dataset data/raw/myciel3.col --k 5 --config tour_1pt_reset
python -m src.plot_best --dataset data/raw/queen7_7.col --k 9 --config roulette_uniform_swap
python -m src.plot_best --dataset data/raw/fpsol2.i.1.col --k 80 --config roulette_uniform_swap
