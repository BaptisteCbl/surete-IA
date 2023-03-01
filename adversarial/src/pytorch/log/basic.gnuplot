# @author: GuillaumeCld
#
# Script to quickly plot from the log files.
# This script only works for fast_gradAlign log (only log with test)
# The scrit takes in argument a .csv log file.
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
    
    set term png         # (will produce .png output)
    set output ARG3      # (output to any filename.png you want)
    print "Saved"
}

plot filename using x:y  notitle

if(output_cond == 1){
    set term x11
    replot
}
pause -1
