
set datafile separator ','
set key autotitle columnhead


set arrow from 10,68 to 10,90 nohead lc rgb 'gray' dt 2
plot \
 "FashionMNIST_none_epss.csv" using 1:2  title "Clean", \
 "FashionMNIST_fgsm_epss.csv" using 1:2  title "FGSM", \
 "FashionMNIST_pgd_epss.csv" using  1:2  title "PGD"
pause -1