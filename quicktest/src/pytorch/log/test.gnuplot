set datafile separator ','
set key autotitle columnhead

# Epoch,Train loss,Train acc clean,Train acc PGD,Test acc clean,Test acc FGSM,Test acc PGD,Training time,Time elapsed
# 1      2          3              4             5              6             7            8             9

plot filename using 1:3
pause -1
