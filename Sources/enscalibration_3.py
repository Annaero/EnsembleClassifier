import sys
import os.path

from utils import rmse, calibrate_ensemble, calibrate_peak_ensemble

from itertools import product
from numpy.linalg import lstsq

from regres import read_data
    
if __name__ == "__main__":
    path = sys.argv[1]
    path2 = sys.argv[2]

    MODEL = "S1"

    measurementsFile = os.path.join(path, "2011080100_measurements_{}_2623.txt".format(MODEL))
    noswanFile = os.path.join(path, "2011080100_noswan_{}_48x434.txt".format(MODEL))
    swanFile = os.path.join(path, "2011080100_swan_{}_48x434.txt".format(MODEL))
    hirombFile = os.path.join(path, "2011080100_hiromb_{}_60x434.txt".format(MODEL))
    coeffsFile = os.path.join(path, "ens_coefs.txt")
    
    measurements = read_data(measurementsFile)
    noswan = read_data(noswanFile)
    swan = read_data(swanFile)
    hiromb = read_data(hirombFile)
    hiromb = [[h-34 for h in forecast] for forecast in hiromb]

    models_forecasts = [hiromb, swan, noswan]    
    form_str = "\t".join(["{{{0}:.3f}}".format(i) for i in range(4)])     
     
    with open(os.path.join(path2, "ens_coefs.txt"), "w+") as ens_coef_file:
        for coefs in calibrate_ensemble(models_forecasts, measurements):
            coef_str = form_str.format(*coefs)
            ens_coef_file.write(coef_str+"\n")
            print(coef_str)
            
    with open(os.path.join(path2, "peak_ens_coefs.txt"), "w+") as peak_ens_coef_file:
        t, h = calibrate_peak_ensemble(models_forecasts, measurements)
        t_coef_str = form_str.format(*t)
        peak_ens_coef_file.write(t_coef_str + "\n")
        h_coef_str = form_str.format(*h)
        peak_ens_coef_file.write(h_coef_str + "\n")
            
        print(t_coef_str+"\n"+h_coef_str)