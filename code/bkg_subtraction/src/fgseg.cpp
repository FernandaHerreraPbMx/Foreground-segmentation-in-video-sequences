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
bgs::bgs(double threshold, double alpha, bool selective_bkg_update, bool rgb)
{
	_rgb = rgb;
	_threshold = threshold;
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
	}
	Frame.copyTo(_bkg);																	// Defines background model
	_bkgcounter = Mat::zeros(Size(Frame.cols,Frame.rows), CV_8UC1);						// Defines a counter used for stationary objects supression
}

//method to perform BackGroundSubtraction
void bgs::bkgSubtraction(cv::Mat Frame)
{	
	if (!_rgb){																			// GRAY METHOD
		cvtColor(Frame, Frame, COLOR_BGR2GRAY); 										// To work with gray even if input is color
		Frame.copyTo(_frame);															// Saves current fame 
		absdiff(_frame, _bkg,_diff);													// Compute difference between bkg and current frame and 
		_bgsmask = _diff > _threshold;													// 		measure if value is out of 3 standard deviations
																						// Value is set to 1 if foreground and to 0 if background
	}
	else{																				// RGB METHOD

		cv::Mat chans[3];																// Auxiliary variables used to split and merge bkg masks
		cv::Mat aux;																	//		according to RGB channels
		Frame.copyTo(_frame);															// Saves current fame 
		absdiff(Frame, _bkg, _diff);													// Compute difference between bkg and current frame and 
		aux = _diff > _threshold;														// 		measure if value is out of 3 standard deviations
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
		addWeighted(_frame, _alpha, _bkg, 1-_alpha, 0.0, _bkg);							// Perform operations necessary to update bkg model
	}
	else{																				// SELECTIVE UPDATE
		addWeighted(_frame, _alpha, _bkg, 1-_alpha, 0.0, aux);							// Perform operations necessary to update bkg model
		aux.copyTo(_bkg,~_bgsmask);														//		according to the bkg mask
	}
}

void bgs::ghostSupression()																// SUPRESSION OF STATIONARY OBJECTS
{
	cv::Mat ghostmask;																	// Auxiliary variables
	cv::Mat aux;						
	_bgsmask.copyTo(aux);	
	aux = aux/255;														
	addWeighted(_bkgcounter, 1, aux, 1, 0.0, _bkgcounter);								// Add 1 to the counter if classified as foreground
	_bgsmask.copyTo(_bkgcounter,255-_bgsmask);											// Set to 0 in counter if classified as bkg
	threshold(_bkgcounter, ghostmask, _ghosttau, 1, 0);									// Search for values above a threshold and return a mask
	_frame.copyTo(_bkg,ghostmask);														// Update the bkf using current frame according to previous mask
}

// Method to detect and remove shadows in the BGS mask to create FG mask
void bgs::removeShadows(double alphaV,double betaV,double TH, double TS,int active)				// alphaS and betaS as thresholds for Value channel							
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
		_bgsmask.copyTo(_shadowmask);
		absdiff(_bgsmask, _bgsmask, _shadowmask);	
	}
	absdiff(_bgsmask, _shadowmask, _fgmask);											// eliminates shadows from bgsmask
}





