for leaf in $(find $1 -type d -links 2); do
    mkdir -p $leaf/Dumps
    echo $leaf/Dumps
    mv $leaf/* $leaf/Dumps/
done
