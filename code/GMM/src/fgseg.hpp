/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB1.0: Background Subtraction - Unix version
 *	fgesg.hpp
 *
 * 	Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es) & Juan Carlos San Miguel (juancarlos.sanmiguel@uam.es)
 *	VPULab-UAM 2020
 * 	Maria Fernanda Herrera, David Savary
 */


#include <opencv2/opencv.hpp>

#ifndef FGSEG_H_INCLUDE
#define FGSEG_H_INCLUDE

using namespace cv;
using namespace std;

namespace fgseg {


	//Declaration of FGSeg class based on BackGround Subtraction (bgs)
	class bgs{
	public:

		//constructor with parameter "threshold"
		bgs(double threshold, bool rgb);

		//constructor with parameter "threshold", alpha
		bgs(double threshold, double alpha, bool rgb);

		bgs(double threshold, double alpha, bool selective_bkg_update, int threshold_ghosts2, bool rgb);

		//destructor
		~bgs(void);

		//method to initialize bkg (first frame - hot start)
		void init_bkg(cv::Mat Frame);

		//method to perform BackGroundSubtraction
		void bkgSubtraction(cv::Mat Frame);

		//method to detect and remove shadows in the binary BGS mask
		void removeShadows();

		//returns the BG image
		cv::Mat getBG(){return _means;};

		//returns the DIFF image
		cv::Mat getDiff(){return _diff;};

		//returns the BGS mask
		cv::Mat getBGSmask(){return _bgsmask;};

		//returns the binary mask with detected shadows
		cv::Mat getShadowMask(){return _shadowmask;};

		//returns the binary FG mask
		cv::Mat getFGmask(){return _fgmask;};


		//ADD ADITIONAL METHODS HERE
		//...
		void bkgUpdate();
		void ghostSupression();
		//cv::Mat getBGSmask3ch(){return _bgsmask3ch;};


	private:
		cv::Mat _bkg; //Background model
		cv::Mat	_frame; //current frame
		cv::Mat _diff; //abs diff frame
		cv::Mat _gaussmask; //binary image for bgssub (FG)
		cv::Mat _thmask; //binary image for bgssub (FG)
		cv::Mat _bgsmask; //binary image for bgssub (FG)
		cv::Mat _shadowmask; //binary mask for detected shadows
		cv::Mat _fgmask; //binary image for foreground (FG)
		cv::Mat _bgsmask3ch; //binary image for bgssub (FG)
		cv::Mat _stds; //Background model
		cv::Mat _means; //Background model
		cv::Mat _weights; //Background model
		cv::Mat _bkgcounter; //Background model

		bool _rgb;
		double _threshold;
		
		//ADD ADITIONAL VARIABLES HERE
		//...
		double _alpha;
		bool _selective_bkg_update;
		int _ghosttau;

	};//end of class bgs

}//end of namespace

#endif




