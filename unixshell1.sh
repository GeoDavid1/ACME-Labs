#!/bin/bash

# remove any previously unzipped copies of Shell1/
if [ -d Shell1 ];
then
  echo "Removing old copies of Shell1/..."
  rm -r Shell1
  echo "Done"
fi

# unzip a fresh copy of Shell1/
echo "Unzipping Shell1.zip..."
unzip -q Shell1
echo "Done"

: ' Problem 1: In the space below, write commands to change into the
Shell1/ directory and print a string telling you the current working
directory. '

cd Shell1/
pwd



: ' Problem 2: Use ls with flags to print one list of the contents of
Shell1/, including hidden files and folders, listing contents in long
format, and sorting output by file size. '

ls -alS

: ' Problem 3: Inside the Shell1/ directory, delete the Audio/ folder
along with all its contents. Create Documents/, Photos/, and
Python/ directories. Rename the Random/ folder as Files/. '

rm -rv Audio/
mkdir Documents/
mkdir Photos/
mkdir Python/
mv Random Files

echo "Checking random rename"
ls


: ' Problem 4: Using wildcards, move all the .jpg files to the Photos/
directory, all the .txt files to the Documents/ directory, and all the
.py files to the Python/ directory. '

mv *.jpg Photos/
mv *.txt Documents/
mv *.py Python/

echo "Problem 4"

: ' Problem 5: Move organize_photos.sh to Scripts/, add executable
permissions to the script, and run the script. '
cd Files/Feb/

mv organize_photos.sh ../../Scripts/

cd ../../


cd Scripts

chmod u+x organize_photos.sh

./organize_photos.sh

: ' Problem 6: Copy img_649.jpg from UnixShell1/ to Shell1/Photos, making
sure to leave a copy of the file in UnixShell1/.'

# scp geodavid@acme21.byu.edu:/sshlab/5e/c8/img_649.jpg /Users/user/Documents/ACME-Junior-Fall/byu_vol1/byu_vol1/UnixShell1

cd ../../
pwd
ls
cp img_649.jpg  Shell1/Photos


# remove any old copies of UnixShell1.tar.gz
if [ ! -d Shell1 ];
then
  cd ..
fi

if [ -f UnixShell1.tar.gz ];
then
  echo "Removing old copies of UnixShell1.tar.gz..."
  rm -v UnixShell1.tar.gz
  echo "Done"
fi

# archive and compress the Shell1/ directory
echo "Compressing Shell1/ Directory..."
tar -zcpf UnixShell1.tar.gz Shell1/*
echo "Done"
