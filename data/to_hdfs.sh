#!/bin/bash

f_dir="$HOME/hadoop_setup/data/"
hdfs_dir="/phase2/"

# Check to see if hdfs directory exists
hdfs dfs -test -d "$hdfs_dir"
if [ $? -ne 0 ]; then
    echo "${hdfs_dir} does not exist in HDFS. Creating it..."
    hdfs dfs -mkdir "$hdfs_dir"
fi

hdfs_dir="/phase2/data/"

# Check to see if hdfs directory exists
hdfs dfs -test -d "$hdfs_dir"
if [ $? -ne 0 ]; then
    echo "${hdfs_dir} does not exist in HDFS. Creating it..."
    hdfs dfs -mkdir "$hdfs_dir"
fi

# Unzip all of the zip files in the data directory
echo 'Unzipping files...'
for f in "$f_dir"*.zip; do
        unzip -o "$f" -d "$f_dir"
done

mv "${f_dir}out.json" "${f_dir}tweets.json"

# Remove the mac os empty dir
rm -r "${f_dir}__MACOSX"

# Move all files that are not .zip, .sh, or .md to hdfs
echo 'Moving data to hdfs...'
find "$f_dir" -type f ! \( -name "*.zip" -o -name "*.sh" -o -name "*.md" -o -name ".DS_Store" \) | while read item; do
    echo "Uploading $item to HDFS"
    hdfs dfs -moveFromLocal "$item" "$hdfs_dir"
done

# Delete all of the unzipped contents
echo 'Deleting unzipped files...'
find "$f_dir" -type f ! -name "*.md" ! -name "*.sh" ! -name "*.zip" -exec rm -f {} \;
find "$f_dir" -type d ! -path "$f_dir" -empty -exec rmdir {} \;

echo "Upload complete!"