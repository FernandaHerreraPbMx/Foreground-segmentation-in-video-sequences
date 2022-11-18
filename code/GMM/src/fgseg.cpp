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

//second constructor
bgs::bgs(double threshold, double alpha, bool rgb)
{
	_rgb = rgb;
	_threshold = threshold;
	_alpha = alpha;
}

//third constructor
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
{	
	if (!_rgb){
		cvtColor(Frame, Frame, COLOR_BGR2GRAY); 										// to work with gray even if input is color
	}

	vector<Mat> auxM(3);																// Auxiliary matrixes to split and merge channels
	vector<Mat> auxS(3);
	vector<Mat> auxW(3);
	
	_means = Mat::zeros(Size(Frame.cols,Frame.rows),CV_8UC3); 							// Matrix of means that define each gaussian
	_stds = Mat::zeros(Size(Frame.cols,Frame.rows),CV_8UC3); 							// Matrix of standard deviations that define each gaussian
	_weights = Mat::zeros(Size(Frame.cols,Frame.rows),CV_64FC3); 						// Matrix of weights that define each gaussian
	_bgsmask3ch = Mat::zeros(Size(Frame.cols,Frame.rows),CV_8UC3); 						// Matrix of masks that define each gaussian
	
	split(_means, auxM);														
	Frame.copyTo(auxM[0]);																// Set the means of the first gaussian according to initial frame
	randu(auxM[1], 0, 256);																// Set means of other gaussians randomly
	randu(auxM[2], 0, 256);
	merge(auxM,_means);
	//cout << "bkg: " << (int) _means.at<Vec3b>(100,100)[1] << endl;

	split(_stds, auxS);
	auxS[0] = 10 + Mat::zeros(Size(Frame.cols,Frame.rows),CV_8UC1);						// Set the standard deviation of the first gaussian w/low value
	auxS[1] = 30 + Mat::zeros(Size(Frame.cols,Frame.rows),CV_8UC1);						// Set the standard deviation of the second gaussian w/high value
	auxS[2] = 30 + Mat::zeros(Size(Frame.cols,Frame.rows),CV_8UC1);						// Set the standard deviation of the third gaussian w/high value
	merge(auxS,_stds);
	//cout << "bkg: " << (int) _stds.at<Vec3b>(100,100)[0] << endl;

	split(_weights, auxW);	
	auxW[0] = 0.5 + Mat::zeros(Size(Frame.cols,Frame.rows),CV_64FC1);					// Set the weight  of the first gaussian w/high value
	auxW[1] = 0.25 + Mat::zeros(Size(Frame.cols,Frame.rows),CV_64FC1);					// Set the weight  of the first gaussian w/low value
	auxW[2] = 0.25 + Mat::zeros(Size(Frame.cols,Frame.rows),CV_64FC1);					// Set the weight  of the first gaussian w/low value
	merge(auxW,_weights);
	//cout << "bkg: " << (double) _weights.at<Vec3d>(100,100)[0] << endl;

	//Frame.copyTo(_bkg);
	_thmask = Mat::zeros(Size(Frame.cols,Frame.rows),CV_8UC3);							// Mask used to threshold background and foreground gaussians according to weights
}


//method to perform BackGroundSubtraction
void bgs::bkgSubtraction(cv::Mat Frame)
{	
	vector<Mat> aux(3);
	vector<Mat> auxm(3);
	vector<Mat> auxstd(3);
	vector<Mat> flags(3);
	Mat thmask;
	Mat bkgtemp;
	Mat BG;

	Mat auxz;																				// Auxiliar matrix of zeros
	Mat auxo;

	double mymin,mymax;
	int minc, maxc,medc;
	int idxmin;
	int idxmax;
	int c;

	if (!_rgb){
		cvtColor(Frame, Frame, COLOR_BGR2GRAY); 												// to work with gray even if input is color
		_frame = Mat::zeros(Size(Frame.cols,Frame.rows),CV_8UC3); 

		split(_frame, aux);
		Frame.copyTo(aux[0]);
		Frame.copyTo(aux[1]);
		Frame.copyTo(aux[2]);
		merge(aux,_frame);
		
		////////////////////////////////////////////////////////////////////////////////////////// Organize weights in decreasing order
		
		split(_weights, aux);
		split(_means, auxm);
		split(_stds, auxstd);

		auxz = aux[1] < aux[2];	
		aux[2].copyTo(auxo);
		aux[1].copyTo(aux[2],auxz);
		auxo.copyTo(aux[1],~auxz);

		auxm[2].copyTo(auxo);
		auxm[1].copyTo(auxm[2],auxz);
		auxo.copyTo(auxm[1],~auxz);

		auxstd[2].copyTo(auxo);
		auxstd[1].copyTo(auxstd[2],auxz);
		auxo.copyTo(auxstd[1],~auxz);
		//////////////////////////////////////////////////////////////////
		auxz = aux[0] < aux[1];	
		aux[1].copyTo(auxo);
		aux[0].copyTo(aux[1],auxz);
		auxo.copyTo(aux[0],~auxz);

		auxm[1].copyTo(auxo);
		auxm[0].copyTo(auxm[1],auxz);
		auxo.copyTo(auxm[0],~auxz);

		auxstd[1].copyTo(auxo);
		auxstd[0].copyTo(auxstd[1],auxz);
		auxo.copyTo(auxstd[0],~auxz);
		//////////////////////////////////////////////////////////////////
		auxz = aux[1] < aux[2];	
		aux[2].copyTo(auxo);
		aux[1].copyTo(aux[2],auxz);
		auxo.copyTo(aux[1],~auxz);

		auxm[2].copyTo(auxo);
		auxm[1].copyTo(auxm[2],auxz);
		auxo.copyTo(auxm[1],~auxz);

		auxstd[2].copyTo(auxo);
		auxstd[1].copyTo(auxstd[2],auxz);
		auxo.copyTo(auxstd[1],~auxz);

		//////////////////////////////////////////////////////////////////////////////////////////

		absdiff(_frame, _means,_diff);															// Obtain difference between  frame and mean of gaussians
		_gaussmask = _diff <= (2.5 * _stds);													// Check if difference is within 2.5 standard deviations


	    ////////////////////////////////////////////////////////////////////////////////////////// Define the B-UPPER gaussians according to the threshold and sum of weights
				 																				// Auxiliar matrix of ones
		auxz = Mat::zeros(Size(Frame.cols,Frame.rows),CV_8UC1);
		auxo = 255 + Mat::zeros(Size(Frame.cols,Frame.rows),CV_8UC1);
	
		split(_weights, aux);																	// Flag matrixes are used to acumulate the sum of weights
		flags[0] = aux[0] >= _threshold;														// Flag[0] contains 255 if the weight of the gaussian 0 (G w/higher weight) is larger than th
		addWeighted(aux[0],1,aux[1],1,0,auxz);													// Flag[1] contains 255 if the sum of weights of gaussian 0 and 1 is larger than th
		flags[1] = auxz >= _threshold;															// Flag[2] contains 255 if the sum of weights of all gaussians is larger than th
		addWeighted(auxz,1,aux[2],1,0,auxz);	
		flags[2] = 255 + Mat::zeros(Size(Frame.cols,Frame.rows),CV_8UC1);
		
		auxz = Mat::zeros(Size(Frame.cols,Frame.rows),CV_8UC1);									
		auxz.copyTo(flags[1],flags[0]);															// If gaussian 0 weight is higher than th send gaussian 1 to 0
		auxz.copyTo(flags[2],flags[0]);															// If gaussian 0 weight is higher than th send gaussian 2 to 0
		auxz.copyTo(flags[2],flags[1]);															// If gaussian 0 and 1 weight is higher than th send gaussian 2 to 0
		
		auxo.copyTo(flags[0],flags[2]);															// If lower gaussians is still 255, upper gaussians are also 255 
		auxo.copyTo(flags[1],flags[2]);
		auxo.copyTo(flags[0],flags[1]);
		merge(flags,_thmask);
		////////////////////////////////////////////////////////////////////////////////////////// Define the matched gaussian according to standard deviation and weights
		split(_gaussmask, aux);																	// Check if pixel value falls within the gaussians in decreasing order
		auxz.copyTo(aux[1],aux[0]);																// 			according to the weights and, if there is a match, send the 
		auxz.copyTo(aux[2],aux[0]);																//			other gaussians flag to 0
		auxz.copyTo(aux[2],aux[1]);
		merge(aux,_gaussmask);

		bitwise_and(_thmask,_gaussmask, _bgsmask3ch);											// Join standard deviation and threshold criterion to select bkg gaussians to update

		split(_bgsmask3ch, aux);
		bitwise_or(aux[0],aux[1], BG);
		bitwise_or(BG,aux[2], _bgsmask);
		absdiff(_bgsmask,255,_bgsmask);															// Define bkg mask by ORing the gaussiian masks
		_bgsmask3ch = _bgsmask3ch/255;	
	}
	else{
		cout << "RGB method not available" << endl;
	}
}

void bgs::bkgUpdate(){

	cv::Mat aux;
	cv::Mat stdsqr;
	cv::Mat newfg;
	
	vector<Mat> auxvec1(3);
	vector<Mat> auxvec2(3);
	vector<Mat> auxvec3(3);

																								// BACKGROUND UPDATE 
		absdiff(_frame,_means,aux);																// Update standard deviation of bkg gaussians
		aux.convertTo(aux,CV_64FC3);
		multiply(_stds,_stds, stdsqr, 1, CV_64FC3); 
		multiply(aux,aux, aux, 1, CV_64FC3); 		
		addWeighted(aux, _alpha, stdsqr, 1-_alpha, 0.0, aux);
		cv::sqrt(aux,aux);
		aux.convertTo(aux,CV_8UC3);
		aux.copyTo(_stds,_bgsmask3ch);

		addWeighted(_frame, _alpha, _means, 1-_alpha, 0.0, aux);								// Update mean of bkg gaussians
		aux.copyTo(_means,_bgsmask3ch);

		addWeighted(_weights, 1-_alpha, _bgsmask3ch, _alpha, 0.0, aux, CV_64FC3);				// Update weights of all bkg gaussians
		aux.copyTo(_weights,~_bgsmask);

																								// FOREGROUND UPDATE 
		split(_frame, auxvec1);
		split(_means, auxvec2);

		auxvec1[2].copyTo(auxvec2[2],_bgsmask);													// Replace gaussian w/lower weight by gaussian coming from frame
		merge(auxvec2,_means);

		split(_stds, auxvec2);
		newfg = 30 + Mat::zeros(Size(_stds.cols,_stds.rows),CV_8UC1);							// Set standard deviation to a high value
		newfg.copyTo(auxvec2[2],_bgsmask);
		merge(auxvec2,_stds);
		/*
		cout << "after uodate" << endl;
		cout << "frame:" << (int) _frame.at<Vec3b>(100,150)[0] << endl;
		cout << "mean 1:" << (int) _means.at<Vec3b>(100,150)[0] << endl;
		cout << "mean 2:" << (int) _means.at<Vec3b>(100,150)[1] << endl;
		cout << "mean 3:" << (int) _means.at<Vec3b>(100,150)[2] << endl;
		cout << "std 1:" << (int) _stds.at<Vec3b>(100,150)[0] << endl;
		cout << "std 2:" << (int) _stds.at<Vec3b>(100,150)[1] << endl;
		cout << "std 3:" << (int) _stds.at<Vec3b>(100,150)[2] << endl;
		cout << "w 1:" << (double) _weights.at<Vec3d>(150,100)[0] << endl;
		cout << "w 2:" << (double) _weights.at<Vec3d>(150,100)[1] << endl;
		cout << "w 3:" << (double) _weights.at<Vec3d>(150,100)[2] << endl;
		cout << "diff 1:" << (int) _diff.at<Vec3b>(100,150)[0] << endl;
		cout << "diff 2:" << (int) _diff.at<Vec3b>(100,150)[1] << endl;
		cout << "diff 3:" << (int) _diff.at<Vec3b>(100,150)[2] << endl;
		cout << "gaussian yes no 1:"  << (int) _gaussmask.at<Vec3b>(100,150)[0] << endl;
		cout << "gaussian yes no 2:" << (int) _gaussmask.at<Vec3b>(100,150)[1] << endl;
		cout << "gaussian yes no 3:" << (int) _gaussmask.at<Vec3b>(100,150)[2] << endl;
		cout << "th 1:"  << (int) _thmask.at<Vec3b>(100,150)[0] << endl;
		cout << "th 2:" << (int) _thmask.at<Vec3b>(100,150)[1] << endl;
		cout << "th 3:" << (int) _thmask.at<Vec3b>(100,150)[2] << endl;
		cout << "and 1:" << (int) _bgsmask3ch.at<Vec3b>(100,150)[0] << endl;
		cout << "and 2:" << (int) _bgsmask3ch.at<Vec3b>(100,150)[1] << endl;
		cout << "and 3:" << (int) _bgsmask3ch.at<Vec3b>(100,150)[2] << endl;
		cout << "bkg 0:" << (int) _bgsmask.at<uchar>(100,150) << endl; */		
	
}

//method to detect and remove shadows in the BGS mask to create FG mask
void bgs::removeShadows()
{
	// init Shadow Mask (currently Shadow Detection not implemented)
	_bgsmask.copyTo(_shadowmask); // creates the mask (currently with bgs)
	//ADD YOUR CODE HERE
	//...
	absdiff(_bgsmask, _bgsmask, _shadowmask);// currently void function mask=0 (should create shadow mask)
	//...
	absdiff(_bgsmask, _shadowmask, _fgmask); // eliminates shadows from bgsmask
}
