#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <string>
#include <fstream>
#include <cassert>

#include <opencv2/opencv.hpp>
#include "track.h"

using namespace std;
using namespace cv;

int main()
{
	vector<string> imgnames;
	ifstream imgf("imgs.txt");
	while (!imgf.eof())
	{
		string img;
		getline(imgf, img);
		if(img.empty())
			continue;
		imgnames.push_back(img);
	}
	
	cout << "[Main] Readed " << imgnames.size() << " images\n" << endl;

	vector<cv::Mat> imgs;
	imgs.resize(imgnames.size());

	for(unsigned i = 0; i < imgnames.size(); ++i)
	{
		imgs[i] = imread(imgnames[i], CV_LOAD_IMAGE_GRAYSCALE);
		//cv::resize(imgs[i], imgs[i], cv::Size(), 0.2, 0.2);
	}

	Tracks tracks;
	tracks.computeTracks(imgs);
	tracks.writeTracks();
	/*std::vector<ImgKeys> imgKeys = tracks.getImgKeys();
	std::vector<Track> ts = tracks.getTracks();

	for( unsigned i = 0; i < imgs.size(); ++i)
		cv::cvtColor(imgs[i], imgs[i], CV_GRAY2BGR);

	for(unsigned i = 0; i < ts.size(); ++i)
	{
		Track &t = ts[i];
		for(unsigned j = 0; j < t.size(); ++j)
		{
			unsigned im = t[j].first;
			unsigned key = t[j].second;
			

			cv::circle(imgs[im], imgKeys[im][key], 4 , cv::Scalar(128, 255, i * 10));
			stringstream ss;
			ss << im;
			string s;
			ss >> s;
			imshow(s, imgs[im]);
		}
		waitKey();
	}*/
	
	/*for( unsigned i = 0; i < imgs.size(); ++i)
	{
	stringstream ss;
	ss << i;
	string s;
	ss >> s;

	imshow(s, imgs[i]);
	}
	waitKey();*/
	return 0;
}
