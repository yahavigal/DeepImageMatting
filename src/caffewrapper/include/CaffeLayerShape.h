namespace caffe
{

	enum ShapeIndex
	{
		Num = 0,
		Channels,
		Height,
		Width
	};

	struct CaffeLayerShape
	{
		int num;
		int channels;
		int height;
		int width;
	};

}