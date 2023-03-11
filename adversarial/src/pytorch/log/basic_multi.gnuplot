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
# Epoch, Train loss,Train acc, Time, Time elapsed
# 1      2          3          4      5


set datafile separator ','
set key autotitle columnhead

filename = ARG1
output_cond = int(ARG2)
x = int(ARG4)
y = int(ARG5)

labels = "'Epoch' 'Train loss' 'Train accuracy' 'Time' 'Time elapsed'"

set xlabel word(labels,x)
set ylabel word(labels,y)

if(x==1){
    set xrange [1:] # first epoch at 1
}

if(output_cond == 1){
    set term svg         # (will produce .svg output)
    set output ARG3      # (output to any filename.svg you want)
    print "Saved"
}
plot \
    "basic/FashionMNIST_cnn_small_0.0392_none.csv" using x:y title "Clean"with linespoints , \
    "basic/FashionMNIST_cnn_small_0.0392_fgsm.csv" using x:y title "FGSM" with linespoints , \
    "basic/FashionMNIST_cnn_small_0.0392_pgd.csv" using x:y title "PGD"with linespoints 

if(output_cond == 1){
    set term x11
    replot
}
pause -1
