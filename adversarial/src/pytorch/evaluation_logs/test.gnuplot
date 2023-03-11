
set datafile separator ','
set key autotitle columnhead



set ylabel "Test accuray for PGD" font ",12"
set xlabel "Epsilon" font ",12"
set xrange [1:40]
set xtics 0,10,40 font ",8"

set yrange [0:90]
set ytics  10,20,90 font ",8"
set key font ",12"
set term pdf
set output "FashionMNIST_all_atks_eps.pdf"

set arrow from 10,0 to 10,90 nohead lc rgb 'red' dt 2
set label "Train epsilon" at 12,4  tc rgb "red"
plot \
"basic/FashionMNIST_none_epss.csv" using 1:2  title "Clean", \
 "basic/FashionMNIST_fgsm_epss.csv" using 1:2  title "FGSM", \
 "basic/FashionMNIST_pgd_epss.csv"  using 1:2  title "PGD"

set term x11
replot
pause -1

#  "fast_gradAlign/FashionMNIST_none_epss.csv" using 1:2  title "Fast Clean", \
#  "fast_gradAlign/FashionMNIST_fgsm_epss.csv" using 1:2  title "Fast FGSM", \
#  "fast_gradAlign/FashionMNIST_pdg_epss.csv"  using 1:2  title "Fast PGD",\
 