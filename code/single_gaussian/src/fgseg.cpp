/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB1.0: Background Subtraction - Unix version
 *	fgesg.cpp
 *
 * 	Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es) & Juan Carlos San Miguel (juancarlos.sanmiguel@uam.es)
 *	VPULab-UAM 2020
 * 	Maria Fernanda Herrera, David Savary
 */

#include <opencv2/opencv.hpp>
#include "fgseg.hpp"

using namespace fgseg;

//default constructor
bgs::bgs(double threshold, bool rgb)
{
	_rgb = rgb;
	_threshold = threshold;
}

//second constructor defined when background update is performed
bgs::bgs(double alpha, bool selective_bkg_update, bool rgb)
{
	_rgb = rgb;
	_alpha = alpha;
	_selective_bkg_update = selective_bkg_update;
}

//third constructor defined when supression of stationary objects is required
bgs::bgs(double threshold, double alpha, bool selective_bkg_update, int threshold_ghosts2, bool rgb)
{
	_rgb = rgb;
	_threshold = threshold;
	_alpha = alpha;
	_selective_bkg_update = selective_bkg_update;
	_ghosttau = threshold_ghosts2;
}

//default destructor
bgs::~bgs(void)
{
}

//method to initialize bkg (first frame - hot start)
void bgs::init_bkg(cv::Mat Frame)
{																						// GRAY METHOD
	if (!_rgb){
		cvtColor(Frame, Frame, COLOR_BGR2GRAY); 										// To work with gray even if input is color
		_bkg_std = Mat(Frame.rows, Frame.cols, CV_8UC1, cv::Scalar(30,30,30));			// Defines std matrix in case of single gaussian
	}
	else {																				// RGB METHOD
		_bkg_std = Mat(Frame.rows, Frame.cols, CV_8UC3, cv::Scalar(30,30,30));			// Defines std matrix in case of single gaussian
	}
	Frame.copyTo(_bkg);																	// Defines background model
}

//method to perform BackGroundSubtraction
void bgs::bkgSubtraction(cv::Mat Frame)
{	
	if (!_rgb){																			// GRAY METHOD
		cvtColor(Frame, Frame, COLOR_BGR2GRAY); 										// To work with gray even if input is color
		Frame.copyTo(_frame);															// Saves current fame 
		absdiff(_frame, _bkg,_diff);													// Compute difference between bkg and current frame and 
		_bgsmask = _diff > 3*_bkg_std;													// 		measure if value is out of 3 standard deviations
																						// Value is set to 1 if foreground and to 0 if background
	}
	else{																				// RGB METHOD

		cv::Mat chans[3];																// Auxiliary variables used to split and merge bkg masks
		cv::Mat aux;																	//		according to RGB channels
		Frame.copyTo(_frame);															// Saves current fame 
		absdiff(Frame, _bkg, _diff);													// Compute difference between bkg and current frame and 
		aux = _diff > 3*_bkg_std;														// 		measure if value is out of 3 standard deviations
																						// Value is set to 1 if foreground and to 0 if background
		split(aux, chans);																// Masks obtained from RGB channels are ORed together and 
		bitwise_or(chans[0], chans[1], aux);											// 		saved to bkg mask
		bitwise_or(aux, chans[2], _bgsmask);
	}
}

void bgs::bkgUpdate()
{
	cv::Mat aux;																		// Auxiliary matrixes
	cv::Mat auxb;

	if (!_selective_bkg_update){														// BLIND UPDATE 
		absdiff(_frame,_bkg,aux);														// Perform operations necessary to update standar deviations
		multiply(_bkg_std,_bkg_std, auxb, 1, CV_64FC1);
		multiply(aux,aux, aux, 1, CV_64FC1);
		addWeighted(aux, _alpha, auxb, 1-_alpha, 0.0, aux);
		cv::sqrt(aux,aux);
		aux.convertTo(aux,CV_8UC1);
		aux.copyTo(_bkg_std);
		addWeighted(_frame, _alpha, _bkg, 1-_alpha, 0.0, aux);							// Perform operations necessary to update bkg model
		aux.copyTo(_bkg);
	}
	else{
		absdiff(_frame,_bkg,aux);														// SELECTIVE UPDATE
		multiply(_bkg_std,_bkg_std, auxb, 1, CV_64FC1);									// Perform operations necessary to update standar deviations
		multiply(aux,aux, aux, 1, CV_64FC1);											//		according to the bkg mask
		addWeighted(aux, _alpha, auxb, 1-_alpha, 0.0, aux);
		cv::sqrt(aux,aux);
		aux.convertTo(aux,CV_8UC1);
		aux.copyTo(_bkg_std,~_bgsmask);
		addWeighted(_frame, _alpha, _bkg, 1-_alpha, 0.0, aux);							// Perform operations necessary to update bkg model
		aux.copyTo(_bkg,~_bgsmask);														//		according to the bkg mask
	}
}


// Method to detect and remove shadows in the BGS mask to create FG mask
void bgs::removeShadows(double alphaV,double betaV,double TH, double TS,int active)		// alphaS and betaS as thresholds for Value channel							
{																						// TH and TS as hue and saturation thresholds
	if(!_rgb){
		_bgsmask.copyTo(_shadowmask);													 // Defines shadowmask as a mask of zeros since 
		absdiff(_bgsmask, _bgsmask, _shadowmask);										 // 	it should not be used when grayscale
	}
	else{
		cv::Mat hsvBKG;																	 // Defines auxiliary matrixes 														
		cv::Mat hsvFRAME;
		cv::Mat t1;
		cv::Mat t2;
		cv::Mat t3;
		cv::Mat chansBKG[3];
		cv::Mat chansFRAME[3];

		cvtColor(_bkg, hsvBKG, COLOR_BGR2HSV);
		split(hsvBKG, chansBKG);
		cvtColor(_frame, hsvFRAME, COLOR_BGR2HSV);
		split(hsvFRAME, chansFRAME);

		divide(chansFRAME[2], chansBKG[2], t1, 1, 5);									// Value channel
		absdiff(chansFRAME[1], chansBKG[1], t2);										// Saturation channel
		absdiff(chansFRAME[0], chansBKG[0], t3);										// Hue channel
		min(t3,360-t3,t3);

		inRange(t1, alphaV, betaV, t1);													// Value channel compared against its thresholds
		threshold(t2, t2, TS, 1, 1);													// Satutation channel compared against its thresholds
		threshold(t3, t3, TH, 1, 1);													// Hue channel compared against its thresholds
		t1.convertTo(t1, 0);
		
		bitwise_and(t1,t2, t1);															// AND to the 3 masks obtained previously which are 							
		bitwise_and(t1,t3, _shadowmask);												// 		the shadowmask
		_shadowmask = _shadowmask*255;									
	}
	if(active==0){
		_bgsmask.copyTo(_shadowmask);													 // Defines shadowmask as a mask of zeros since 
		absdiff(_bgsmask, _bgsmask, _shadowmask);										 // 	it should not be used when grayscale
	}
	absdiff(_bgsmask, _shadowmask, _fgmask);											// eliminates shadows from bgsmask
}





