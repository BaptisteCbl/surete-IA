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


x = 1
y = 7

labels = "'Epoch' 'Train loss' 'Train accuracy clean' 'Train accuracy for  PGD' 'Test accuracy clean' 'Test accuracy FGSM' 'Test accuracy PGD' 'Learning Rate' 'Time' 'Time elapsed'"

set key outside spacing 1.5 #width 1# font ",7"
set xlabel word(labels,x)
set ylabel "Accuracy (%)"
set xtics #font ",7"
set pointsize 1.2

set yrange [20:100]
set ytics  20,20,100 #font ",10"
set term pdf size 11, 4.4 font ",20" 
#
set output "FahionMNIST_test_acc_basicTrain.pdf"

plot \
    "fast_gradAlign/FashionMNIST_cnn_small_10_none_processed.csv" using x:5 title "Clean", \
    "fast_gradAlign/FashionMNIST_cnn_small_10_none_processed.csv" using x:6 title "FGSM" , \
    "fast_gradAlign/FashionMNIST_cnn_small_10_none_processed.csv" using x:7 title "PGD" 

    # "basic/FashionMNIST_cnn_small_0.0392_none.csv" using x:3 title "Basic clean"   with points pointtype 26, \
    # "basic/FashionMNIST_cnn_small_0.0392_fgsm.csv" using x:3 title "Baisc FGSM"  with points pointtype 3

set term x11
replot

pause -1

