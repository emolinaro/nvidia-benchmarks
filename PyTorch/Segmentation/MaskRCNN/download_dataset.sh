DATA_DIR=$1

wget -c https://dl.fbaipublicfiles.com/detectron/coco/coco_annotations_minival.tgz
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip

if md5sum -c hashes.md5
then
	echo "DOWNLOAD PASSED"
	# mkdir $DATA_DIR
	mv coco_annotations_minival.tgz $DATA_DIR
	mv train2014.zip $DATA_DIR
	mv val2014.zip $DATA_DIR
	mv annotations_trainval2014.zip $DATA_DIR

	cd $DATA_DIR
	dtrx --one=here coco_annotations_minival.tgz
	dtrx --one=here annotations_trainval2014.zip
	mv annotations.1/* annotations/

	dtrx train2014.zip
	dtrx val2014.zip
    
    ## create symlinks to datasets
    sudo mkdir -p /datasets/data
    sudo chown -R $USER /datasets
    cd /datasets/data
    ln -s $DATA_DIR/train2014 train2014
    ln -s $DATA_DIR/val2014 val2014
    ln -s $DATA_DIR/annotations annotations
    cd $DATA_DIR

	echo "EXTRACTION COMPLETE"
else
	echo "DOWNLOAD FAILED HASHCHECK"
fi
