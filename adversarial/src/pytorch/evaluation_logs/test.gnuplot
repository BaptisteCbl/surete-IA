
set datafile separator ','
set key autotitle columnhead


set arrow from 10,0 to 10,100 nohead lc rgb 'gray' dt 2
plot \
 "fast_gradAlign/FashionMNIST_none_epss.csv" using 1:2  title "Fast Clean", \
 "fast_gradAlign/FashionMNIST_fgsm_epss.csv" using 1:2  title "Fast FGSM", \
 "fast_gradAlign/FashionMNIST_pdg_epss.csv"  using 1:2  title "Fast PGD",\
 "basic/FashionMNIST_none_epss.csv" using 1:2  title "Basic Clean", \
 "basic/FashionMNIST_fgsm_epss.csv" using 1:2  title "Basic FGSM", \
 "basic/FashionMNIST_pgd_epss.csv"  using 1:2  title "Basic PGD"

 
pause -1