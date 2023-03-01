# @author: GuillaumeCld
#
# Script to quickly plot from the log files.
# This script only works for fast_gradAlign log (only log with test)
# The scrit takes in argument a .csv log file.
# Example:
#   gnuplot -e "filename='fast_gradAlign/MNIST_cnn_10_pgd.csv'" test.gnuplot 
#
# The columns of the csv are:
# 
# Epoch,Train loss,Train acc clean,Train acc PGD,Test acc clean,Test acc FGSM,Test acc PGD,Learning Rate, Training time,Time elapsed
# 1      2          3              4             5              6             7            8             9              10


set datafile separator ','
set key autotitle columnhead


plot filename using 1:9
pause -1
