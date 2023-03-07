# Draw, using the given gnuplot ($1), the given file ($2)
# Executing from src/pytorch/log

# bash plot.sh SCRIPT(.gnuplot) FILE EXPORT(0,1) xcolumn (1-*) ycolumn (1-*)

gnuplot=$1  # script to use
file=$2     # either name of data file to use or name to use when saving
echo "With: $1"

## All checks require to have the arguments in the right order
# Check gnuplot script
if [ -z "$1" ]
then 
    echo "No gnuplot script given"
    exit
fi
# Check input file
if [ -z "$2" ]
then 
    echo "No input file"
else    
    echo "Ploting $file"
fi
# Check output saving condition
if [[ $3 == 0 ]]
then
   output_cond=0
   output_file=""
else
    output_cond=1
    output_file="../figures/"${file%.csv}".png"

    echo "Saving to: $output_file"
fi

gnuplot -c $gnuplot $file $output_cond $output_file $4 $5