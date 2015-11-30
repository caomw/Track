#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef std::pair<unsigned, unsigned> KeyPairMatch;

struct ImgPairMatch
{
	unsigned imgIdx1;	//query image index
	unsigned imgIdx2;	//train image index

	std::vector<KeyPairMatch> keyIdxMatches;
};

typedef std::vector<cv::Point2f> ImgKeys;
typedef std::vector<unsigned> MatchedKeyIdx;	//matched keypoint index in one image

class ImgKeyMatch
{
public:
	ImgKeyMatch(std::vector<cv::Mat> &imgs, float ratio = 0.7, unsigned minMatches = 8);

	void computeKeyPoints();

	void computeMatches();

	void writeKeyPoints(std::string basename = "output/keys/key");

	void writeMatches(std::string filename = "output/img_pairs_match.txt");

	std::vector<ImgPairMatch>& getImgPairMatches();

	std::vector<unsigned>& getMatchedImgNums();

	std::vector<ImgKeys>& getImgKeys();

	std::vector<MatchedKeyIdx>& getMatchedKeyIdxs();

private:

	std::vector<cv::Mat> m_imgs;
	
	std::vector<ImgKeys> m_keys;	// store keypoints use point2f

	std::vector<std::vector<KeyPoint> > m_keypoints;	//clearing after getMatches over 
	
	std::vector<cv::Mat> m_descriptors;

	std::vector<ImgPairMatch> m_fullImgPairMatches;

	float m_knnRatio;

	unsigned m_minMatches;	// minimum match number between two images

	std::vector<unsigned> m_matchedImgNum;

	std::vector<MatchedKeyIdx> m_matchedKeyIdxs;	//matched keypoints index in each image
};


struct AdjListElem
{
	AdjListElem():m_index(-1)
	{
	}
	int m_index;	//query image index
	std::vector<KeyPairMatch> m_matches;
};


typedef vector<AdjListElem> MatchAdjList;


class MatchTable
{
public:
	MatchTable();

	MatchTable(unsigned imgNum);
	
	int getAdjListElemIdx(MatchAdjList &list, unsigned idx);

	int getAdjListElemIdx(unsigned idx1, unsigned idx2);

	AdjListElem& getAdjListElem(MatchAdjList &list, int idx);

	AdjListElem& getAdjListElem(unsigned idx1, unsigned idx2);

	void createMatchTable(std::vector<cv::Mat> &imgs, bool symmetry = true);

	std::vector<MatchAdjList>& getMatchTable();

	std::vector<ImgKeys>& getImgKeys();

	std::vector<MatchedKeyIdx>& getMatchedKeyIdxs();


private:

	std::vector<MatchAdjList> m_match_lists;
	AdjListElem m_findElemFaild;
	std::vector<ImgKeys> m_imgKeys;
	std::vector<MatchedKeyIdx> m_matchedKeyIdxs;	//matched keypoints index in each image
};

typedef std::pair<unsigned, unsigned> ImgKeyIdx;
typedef std::vector<ImgKeyIdx> Track;

class CTracks
{
public:
	CTracks(unsigned minTrackSize = 3);
	
	void computeTracks(std::vector<cv::Mat> &imgs);

	void computeTracks();

	int getMatchedKeyIdx(AdjListElem &e, unsigned keyIdx);

	void computeMatchTable(std::vector<cv::Mat> &imgs);

	void writeTracks(std::string filename = "output/tracks.txt");

	std::vector<Track>& getTracks();

	std::vector<ImgKeys>& getImgKeys();
private:
	std::vector<Track> m_tracks;
	std::vector<MatchAdjList> m_matchTable;
	std::vector<ImgKeys> m_imgKeys;
	std::vector<MatchedKeyIdx> m_matchedKeyIdxs;	//matched keypoints index in each image
	unsigned m_minTrackSize;
};
