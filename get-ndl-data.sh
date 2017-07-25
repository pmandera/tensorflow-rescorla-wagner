#!/bin/sh

TMP_DIR=`mktemp --directory`
OUT_DIR="./datasets/ndl/"

wget --directory-prefix=$TMP_DIR \
	https://cran.r-project.org/src/contrib/ndl_0.2.17.tar.gz

tar -zxvf $TMP_DIR/ndl_0.2.17.tar.gz \
	--directory $TMP_DIR

mkdir -p $OUT_DIR

SOURCE_DIR="$TMP_DIR/ndl/data/"

rda_to_csv () {
	R -e "load('$SOURCE_DIR/$1.rda'); write.csv($1, '$OUT_DIR/$2', row.names=FALSE)"
}

rda_to_csv danks danks_ndl.csv
rda_to_csv numbers numbers.csv

rm -rf $TMP_DIR
