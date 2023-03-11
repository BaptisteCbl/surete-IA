# @author: GuillaumeCld
#
# Script to quickly plot from the log files.
# This script only works for fast_gradAlign log (only log with test)
# The scrit takes in argument a .csv log file.

# The columns of the csv are:
# 
# Epoch,Train loss,Train acc clean,Train acc PGD,Test acc clean,Test acc FGSM,Test acc PGD,Learning Rate, Training time,Time elapsed
# 1      2          3              4             5              6             7            8             9              10


set datafile separator ','
set key autotitle columnhead

filename = ARG1
output_cond = int(ARG2)

x = int(ARG4)
y = int(ARG5)

labels = "'Epoch' 'Train loss' 'Train accuracy clean' 'Train accuracy PGD' 'Test accuracy clean' 'Test accuracy FGSM' 'Test accuracy PGD' 'Learning Rate' 'Time' 'Time elapsed'"

set xlabel word(labels,x)
set ylabel word(labels,y)
if(x==1){
    set xrange [1:] # first epoch at 1
}
if(output_cond == 1){
    set term png         # (will produce .png output)
    set output ARG3      # (output to any filename.png you want)
    print "Saved"
}

plot \
    "fast_gradAlign/FashionMNIST_cnn_small_10_none_processed.csv" using x:y title "Clean"with linespoints , \
    "fast_gradAlign/FashionMNIST_cnn_small_10_fgsm_processed.csv" using x:y title "FGSM" with linespoints , \
    "fast_gradAlign/FashionMNIST_cnn_small_10_pgd_processed.csv" using x:y title "PGD"with linespoints 


if(output_cond == 1){
    set term x11
    replot
}
pause -1

