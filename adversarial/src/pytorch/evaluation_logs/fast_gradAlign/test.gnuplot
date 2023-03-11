
set datafile separator ','
set key autotitle columnhead


set ylabel "Epoch"
set xlabel "Test accuray for PGD"
set set xrange [1:40]
set xtics 0,10,40

set yrange [0:90]
set ytics  10,20,90


# set term svg
# set output "FashionMNIST_all_atks_eps.svg"


set arrow from 10,0 to 10,100 nohead lc rgb 'gray' dt 2
plot \
 "FashionMNIST_none_epss.csv" using 1:2  title "Clean", \
 "FashionMNIST_fgsm_epss.csv" using 1:2  title "FGSM", \
 "FashionMNIST_pdg_epss.csv" using  1:2  title "PGD"


pause -1