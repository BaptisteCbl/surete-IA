# Script to compare the parameters described in the first line of the log files.
# Example:
# source src/pytorch/log/log_diff.sh src/pytorch/log/fast_gradAlign/MNIST_cnn_8_fgsm.csv src/pytorch/log/fast_gradAlign/MNIST_cnn_10_fgsm.csv

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color
printf "I love${NC} Stack Overflow\n"

echo "Comparing parameters between: "
printf "${RED} $1 ${NC}"
printf "${GREEN} $2 ${NC} \n"
diff <(head -n 1 $1 ) <(head -n 1 $2) | colordiff

