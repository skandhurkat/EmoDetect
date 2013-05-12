#!/bin/sh

# parse.sh is a shell script for creating the input file in necessary
# format for the programs included in EmoDetect. This shell script
# recursively goes through all subdirectories, and lists all .jpg files
# along with a label to indicate the associated emotion. This file has been
# optimised to work with the MUG facial expressions data set mentioned in
# the README. This file may need to be changed if another data set is used.
#
# Shell script to find out all the files under a directory and its
# subdirectories. This also takes into consideration those files or
# directories which have spaces or newlines in their names 

DIR="."
category=-1
list_files()
{
    if !(test -d "$1") 
    then echo $1; return;
    fi

    cd "$1"
    #echo; echo `pwd`:; #Display Directory name

    for i in *
    do
	if test -d "$i" #if dictionary
	then
	    case $i in
		"anger")
		    category=0;
		    ;;
		"disgust")
    		    category=1;
		    ;;
		"fear")
		    category=2;
		    ;;
		"happiness")
		    category=3;
		    ;;
		"neutral")
		    category=4;
		    ;;
		"sadness")
		    category=5;
		    ;;
		"surprise")
		    category=6;
		    ;;
		"")
		
		;;
		*)
		
		esac
	    list_files "$i" #recursively list files
	    cd ..
	else
	    case $i in
		*jpg*)		    
		    echo "$category\t"`pwd`\/"$i";
		    ;;
	    esac
	fi

    done
}

if [ $# -eq 0 ]
then 
    list_files .
    exit 0
fi

for i in $*
do
    DIR="$1"
    list_files "$DIR"
    shift 1 #To read next directory/file name
done
