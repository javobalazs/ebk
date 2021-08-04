#!/usr/bin/bash

if [ $# != 2 ]
then
  echo "ket parameter kell!"
  exit 1
fi

prefix1="$1"_
prefix2="$2"_

{
  find . -mindepth 1 -maxdepth 1 -name "$prefix1"'*'".json" | sed "s/^\.\/$prefix1//"
  find . -mindepth 1 -maxdepth 1 -name "$prefix2"'*'".json" | sed "s/^\.\/$prefix2//"
} | sort | uniq |
while read a
do
  echo diff "$prefix1$a" "$prefix2$a"
  diff "$prefix1$a" "$prefix2$a"
done

