set datafile separator ','
set key autotitle columnhead

# Epoch,Train loss,Train acc clean,Train acc PGD,Test acc clean,Test acc FGSM,Test acc PGD,Learning Rate, Training time,Time elapsed
# 1      2          3              4             5              6             7            8             9              10

plot filename using 1:9
pause -1


# gnuplot -e "filename='fast_gradAlign/MNIST_cnn_10_pgd.csv'" test.gnuplot 