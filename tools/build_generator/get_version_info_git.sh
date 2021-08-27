CUR_PATH=$1
if [ "$CUR_PATH" = "" ]; then
	CUR_PATH=$PWD
fi
git -C $CUR_PATH describe --long | sed -e "s/-/:/g" &> /dev/null
if [ "$?" != 0 ]; then
	git -C $CUR_PATH describe --long | sed -e "s/-/:/g"
else
    echo "NOT_DEFINED"
fi
