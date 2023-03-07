# Take a tex file in input whith a standalone picture and generate a png of this picture
# Convertion with imagemagick 
# if security policy issue check:
# https://stackoverflow.com/questions/52998331/imagemagick-security-policy-pdf-blocking-conversion

file=$1 
if test -f $file 
then
    if [[ $file == *.tex ]]
    then

        echo "Compile $file"
        pdflatex -interaction=batchmode $file
        echo "Convert pdf to png"
        convert -density 600x600 ${file%.tex}.pdf -quality 90 -resize 800x600 ${file%.tex}.png

        echo "Remove intermediate file created"
        for ext in aux log pdf
        do
            rm ${file%.tex}.$ext
            
        done
    else   
        echo "This is not a .tex file"
    fi
else

    echo "$file does not exist exists."
fi