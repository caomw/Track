#include "track.h"

#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <queue>
#include <utility>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/gpu.hpp>
#include <opencv2/gpu/gpu.hpp>


MatchTable::MatchTable(unsigned imgNum)
{
	if(imgNum < 3)
	{
		cout << "[MatchTable] Error: Image number is less than 3.\n\n";
		exit(0);
	}
	m_match_lists.resize(imgNum);
	m_findElemFaild.m_index = -1;
}

MatchTable::MatchTable()
{
	m_findElemFaild.m_index = -1;
}

int MatchTable::getAdjListElemIdx(MatchAdjList &list, unsigned idx)
{
	for(unsigned i = 0; i < list.size(); ++i)
	{
		if(list[i].m_index == idx)
			return i;
	}
	return -1;
}

int MatchTable::getAdjListElemIdx(unsigned idx1, unsigned idx2)
{
	MatchAdjList &list = m_match_lists[idx1];
	return getAdjListElemIdx(list, idx2);
}

AdjListElem& MatchTable::getAdjListElem(MatchAdjList &list, int idx)
{
	for(unsigned i = 0; i < list.size(); ++i)
	{
		if(list[i].m_index == idx)
			return list[i];
	}
	return m_findElemFaild;
}

AdjListElem& MatchTable::getAdjListElem(unsigned idx1, unsigned idx2)
{
	MatchAdjList &list = m_match_lists[idx1];
	return getAdjListElem(list, idx2);
}

void MatchTable::createMatchTable(std::vector<cv::Mat> &imgs, bool symmetry /*= true*/)
{
	if(imgs.size() < 3)
	{
		cout << "[MatchTable] Error: Image number is less than 3.\n\n";
		exit(0);
	}
	if(imgs.size() != m_match_lists.size())
		m_match_lists.resize(imgs.size());
	
	cout << "[MatchTable] Creating MatchTable...\n" << endl;

	ImgKeyMatch imgkeymatch(imgs);
	imgkeymatch.computeKeyPoints();
	imgkeymatch.computeMatches();
	m_imgKeys = imgkeymatch.getImgKeys();
	m_matchedKeyIdxs = imgkeymatch.getMatchedKeyIdxs();

	std::vector<ImgPairMatch> fullImgPairMatch = imgkeymatch.getImgPairMatches();
	std::vector<unsigned> matchedImgNums = imgkeymatch.getMatchedImgNums();

	// not symmetric
	for(unsigned i = 0, count = 0; i < matchedImgNums.size() - 1; ++i)
	{
		MatchAdjList &mal = m_match_lists[i];
		unsigned inum = matchedImgNums[i];
		if(inum == 0)
			mal.clear();
		else
		{
			for(unsigned j = 0; j < inum; ++j, count++)
			{
				std::vector<ImgPairMatch>::iterator it = fullImgPairMatch.begin() + count;
				AdjListElem e, symmetrye;
				if(it->imgIdx1 != i)
				{
					cout << "[MatchTable] Error: createMatchTable(), match error.\n" << endl;
					exit(0);
				}
				e.m_index = it->imgIdx2;
				symmetrye.m_index = i;

				for(unsigned k = 0; k < it->keyIdxMatches.size(); ++k)
				{
					KeyPairMatch m(it->keyIdxMatches[k]);
					e.m_matches.push_back(m);

					// make matchtable symmetric
					if(symmetry)
					{
						KeyPairMatch sm(m.second, m.first);
						symmetrye.m_matches.push_back(sm);
					}
				}
				mal.push_back(e);
				// make matchtable symmetric
				if(symmetry)
					m_match_lists[e.m_index].push_back(symmetrye);
				
			}
		}
	}
	cout << "[MatchTable] Creating MatchTable is over.\n" << endl;

	// make matchtable symmetric
	/*
	if(symmetry)
	{
		for(unsigned i = 1; i < m_match_lists.size(); ++i)
		{
			MatchAdjList &mali = m_match_lists[i];
			for(unsigned j = 0; j < i; ++j)
			{
				MatchAdjList &malj = m_match_lists[j];
				AdjListElem &e = getAdjListElem(malj, i);
				if(e.m_index < 0)
					continue;
				std::vector<KeyPairMatch> &kpms = e.m_matches;
				AdjListElem me;
				me.m_index = j;
				for(unsigned k = 0; k < kpms.size(); ++k)
				{
					KeyPairMatch kpm(kpms[k]);
					me.m_matches.push_back(kpm);
				}
			}
		}
	}*/
	
}

std::vector<MatchAdjList>& MatchTable::getMatchTable()
{
	return m_match_lists;
}

std::vector<ImgKeys>& MatchTable::getImgKeys()
{
	return m_imgKeys;
}

std::vector<MatchedKeyIdx>& MatchTable::getMatchedKeyIdxs()
{
	return m_matchedKeyIdxs;
}

ImgKeyMatch::ImgKeyMatch(std::vector<cv::Mat> &imgs, float ratio /*= 0.5*/, unsigned minMatches /*= 8*/):m_imgs(imgs), m_knnRatio(ratio),m_minMatches(minMatches)
{
	if(m_imgs.size() < 3)
	{
		std::cout << "[ImgKeyMatch] Error: Image number is less than 3." << endl << endl;
		exit(0);
	}
	m_matchedImgNum = std::vector<unsigned>(m_imgs.size(), 0);
	m_matchedKeyIdxs.resize(imgs.size());
}

void ImgKeyMatch::computeKeyPoints()
{
	cout << "[ImgKeyMatch] Computing keypoints and descriptors...\n"<< endl;
	cv::SURF surf;
	std::vector<cv::KeyPoint> kpts;
	cv::Mat des;
	for(int i = 0; i < m_imgs.size(); ++i)
	{
		cout << "[ImgKeyMatch] Computing image " << i << " ...\n";
		surf(m_imgs[i], cv::Mat(), kpts, des);
		m_keypoints.push_back(kpts);
		m_descriptors.push_back(des);

		std::vector<cv::Point2f> points;
		for(int j = 0; j < kpts.size(); ++j)
		{
			points.push_back(kpts[j].pt);
		}
		m_keys.push_back(points);
	}
}

void ImgKeyMatch::computeMatches()
{
	cv::FlannBasedMatcher matcher;
	
	std::vector<std::set<unsigned> > matchedKeyIdxs;
	matchedKeyIdxs.resize(m_imgs.size());

	cout << "[ImgKeyMatch] Computing matches...\n" << endl;

	for(int i = 0; i < m_imgs.size() - 1; ++i)
	{
		unsigned matchedImg = 0;
		for(int j = i + 1; j < m_imgs.size(); ++j)
		{
			cv::Mat desc2;

			std::vector<std::vector<cv::DMatch> >knnmatches;
			
			cout << "[ImgKeyMatch] Computing matches between image " << i << " and image " << j << endl;
			matcher.knnMatch(m_descriptors[i], m_descriptors[j], knnmatches, 2);

			ImgPairMatch ipm;
			ipm.imgIdx1 = i;
			ipm.imgIdx2 = j;
			
			std::vector<std::pair<unsigned, unsigned> > &idxPairs = ipm.keyIdxMatches;

			cout << "[ImgKeyMatch] Removing bad matches" << endl;
			for(int k = 0; k < knnmatches.size(); ++k)
			{
				cv::DMatch m;
				if((knnmatches[k][0].distance + knnmatches[k][1].distance) != std::numeric_limits<float>::max())
				{
					m = knnmatches[k][0];
					if(m.distance / knnmatches[k][1].distance > m_knnRatio)
						continue;

					std::pair<unsigned, unsigned> idxpair(m.queryIdx, m.trainIdx);
					idxPairs.push_back(idxpair);
					matchedKeyIdxs[i].insert(m.queryIdx);
					matchedKeyIdxs[j].insert(m.trainIdx);
				}
			}
			if(ipm.keyIdxMatches.size() >= m_minMatches)
			{
				m_fullImgPairMatches.push_back(ipm);
				matchedImg++;
			}
			cout << endl;
		}
		m_matchedImgNum[i] = matchedImg;
	}
	if(m_fullImgPairMatches.size() < 3)
	{
		cout <<"[ImgKeyMatch] Error: Matched image pairs is less than 3.\n\n";
		exit(0);
	}
	for(unsigned im = 0; im < matchedKeyIdxs.size(); ++im)
	{
		m_matchedKeyIdxs[im].assign(matchedKeyIdxs[im].begin(), matchedKeyIdxs[im].end());
	}
	m_keypoints.clear();
	m_descriptors.clear();
	m_imgs.clear();
	cout << "[ImgKeyMatch] Matching is over.\n" << endl;
}

void ImgKeyMatch::writeKeyPoints(std::string basename /*= "output/keys/key"*/)
{
	
	for(int i = 0; i < m_keys.size(); ++i)
	{
		std::stringstream ss;
		ss >> i;
		string name;
		ss << name;
		name = basename + name + ".txt";

		std::ofstream ofs(name);
		if(!ofs.is_open())
		{
			cout << "[ImgKeyMatch] Error: File " << name << " open faild.\n\n";
			exit(0);
		}

		std::vector<cv::Point2f> & points = m_keys[i]; 
		ofs << points.size() << '\n';
		for(int j = 0; j < points.size(); ++j)
		{
			ofs << points[j].x << ' ' << points[j].y << ' ';
		}
	}
}

void ImgKeyMatch::writeMatches(std::string filename /*= "output/img_pairs_match.txt"*/)
{
	std::ofstream ofs(filename);
	if(!ofs.is_open())
	{
		cout << "[ImgKeyMatch] Error: File " << filename << " open failed." << endl << endl;
		exit(0);
	}

	cout << "[ImgKeyMatch]  Writing matches to file " << filename << endl << endl;

	for(int i = 0; i < m_fullImgPairMatches.size(); ++i)
	{
		ImgPairMatch &ipm = m_fullImgPairMatches[i];
		std::vector<std::pair<unsigned, unsigned> > &idxpair = ipm.keyIdxMatches;
		ofs << ipm.imgIdx1 << ' ' << ipm.imgIdx2 << ' ' << idxpair.size() << '\n';
		for(int j = 0; j < idxpair.size(); ++j)
		{
			std::pair<unsigned, unsigned> &idxs = idxpair[j];
			ofs << idxs.first << ' ' << idxs.second << '\n';
		}
	}
	ofs.close();
}

std::vector<ImgPairMatch>& ImgKeyMatch::getImgPairMatches()
{
	return m_fullImgPairMatches;
}

std::vector<unsigned>& ImgKeyMatch::getMatchedImgNums()
{
	return m_matchedImgNum;
}

std::vector<ImgKeys>& ImgKeyMatch::getImgKeys()
{
	return m_keys;
}

std::vector<MatchedKeyIdx>& ImgKeyMatch::getMatchedKeyIdxs()
{
	return m_matchedKeyIdxs;
}

CTracks::CTracks(unsigned minTrackSize /*= 3*/)
{
	m_minTrackSize = minTrackSize;
}

void CTracks::computeTracks(std::vector<cv::Mat> &imgs)
{
	computeMatchTable(imgs);
	computeTracks();
}

void CTracks::computeTracks()
{
	cout << "[Tracks] Computing Tracks...\n" << endl;

	std::vector<std::map<unsigned, bool> > keyFlags;
	keyFlags.resize(m_matchedKeyIdxs.size());

	for(unsigned im = 0; im < m_matchedKeyIdxs.size(); ++im)
	{
		MatchedKeyIdx &mki = m_matchedKeyIdxs[im];

		std::vector<bool> flags(mki.size(), false);
		std::map<unsigned, bool> &flag = keyFlags[im];

		for(unsigned i = 0; i < mki.size(); ++i)
		{
			flag[mki[i]] = false;
		}
	}

	for(unsigned im = 0; im < m_matchTable.size(); ++im)
	{
		MatchAdjList &list = m_matchTable[im];
		if(list.size() == 0)
			continue;

		std::map<unsigned, bool> &keyflag = keyFlags[im];

		for(map<unsigned, bool>::iterator it = keyflag.begin(); it != keyflag.end(); ++it)
		{
			if(it->second)
				continue;
			it->second = true;
			unsigned keyIdx = it->first;

			std::vector<bool> imgflag(m_imgKeys.size(), false);
			imgflag[im] = true;

			Track track;
			std::queue<ImgKeyIdx> track_queue;
			ImgKeyIdx ikIdx(im, keyIdx);
			track.push_back(ikIdx);
			track_queue.push(ikIdx);

			while (!track_queue.empty())
			{
				ImgKeyIdx idx = track_queue.front();
				track_queue.pop();

				unsigned imgIdx = idx.first;
				unsigned keyIdx = idx.second;

				MatchAdjList &l = m_matchTable[imgIdx];
				for(unsigned i = 0; i < l.size(); ++i)
				{
					AdjListElem &e = l[i];
					unsigned eImgIdx = e.m_index;
					if(imgflag[eImgIdx])
						continue;
					imgflag[eImgIdx] = true;

					map<unsigned, bool> &f = keyFlags[eImgIdx];
					int matchKeyIdx = getMatchedKeyIdx(e, keyIdx);
					if(matchKeyIdx == -1 || f[matchKeyIdx])
						continue;
					f[matchKeyIdx] = true;
					ImgKeyIdx k(eImgIdx, matchKeyIdx);
					track.push_back(k);
					track_queue.push(k);
				}
			}
			if(track.size() >= m_minTrackSize)
				m_tracks.push_back(track);
		}
		
	}

	cout << "[Tracks] Computing Tracks is over.\n" << endl;
}

int CTracks::getMatchedKeyIdx(AdjListElem &e, unsigned keyIdx)
{
	vector<KeyPairMatch> &m = e.m_matches;
	for(vector<KeyPairMatch>::iterator it = m.begin(); it != m.end(); ++it)
	{
		if(it->first == keyIdx)
			return it->second;
	}

	return -1;
}

void CTracks::computeMatchTable(std::vector<cv::Mat> &imgs)
{
	MatchTable mtb(imgs.size());
	mtb.createMatchTable(imgs);

	m_matchTable = mtb.getMatchTable();
	if(m_matchTable.empty())
	{
		cout << "[Tracks] Error: computeMatchTable(), matchTable is empty.\n" << endl;
		exit(0);
	}

	m_imgKeys = mtb.getImgKeys();
	if(m_imgKeys.empty())
	{
		cout << "[Tracks] Error: computeMatchTable(), imgKeys is empty.\n" << endl;
		exit(0);
	}

	m_matchedKeyIdxs = mtb.getMatchedKeyIdxs();
	if(m_matchedKeyIdxs.empty())
	{
		cout << "[Tracks] Error: computeMatchTable(), matchedKeyIdxs is empty.\n" << endl;
		exit(0);
	}
}

void CTracks::writeTracks(std::string filename /*= "output/tracks.txt"*/)
{
	ofstream ofs(filename);
	if(!ofs.is_open())
	{
		cout << "[Tracks} Error: File " << filename << " open failed\n\n";
		exit(0);
	}
	cout << "[Tracks] Writing tracks...\n" << endl;

	for(unsigned i = 0; i < m_tracks.size(); ++i)
	{
		Track &track = m_tracks[i];
		for(int j = 0; j < track.size(); ++j)
		{
			ofs << track[j].first << ' ' << track[j].second << ' ';
		}
		ofs << '\n';
	}
	ofs.close();
	cout << "[Tracks} Writing is over.\n" << endl;
}

std::vector<Track>& CTracks::getTracks()
{
	return m_tracks;
}

std::vector<ImgKeys>& CTracks::getImgKeys()
{
	return m_imgKeys;
}
