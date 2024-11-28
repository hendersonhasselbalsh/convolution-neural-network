#pragma once

#include "../Processing-Cells/Convolution-Cell/ConvolutionCell.h"
#include "../Processing-Cells/Activation-Cell/ActivationCell.h"
#include "../Processing-Cells/Pooling-Cell/pooling.h"
#include "../Data-Maneger/data-maneging.h"
#include "../FeedForward-Neural-Network/multy-layer-perceptron.h" 
#include "../Utils/basic-includes.h"


class CNNbuilder;



class CNN {

	private:
		std::vector<IProcessingUnit*> _processingUnits;
		MLP _mlp;

		std::function<void(size_t, double, double&)> _UpdateLeraningRate;                      //  double f(size_t epoch, double accuracy, double currentLearningRate)

		size_t _reshapeRows;
		size_t _reshapeCols;

		size_t _maxEpochs;

		CNN();

	public:
		
		std::vector<double> Forward(Eigen::MatrixXd& input);
		std::vector<double> Backward(std::vector<double>& predictedValues, std::vector<double>& correctValues);

		void Training(std::vector<CNN_DATA> trainingDataSet, std::function<void(void)> callback = [](){ });

		void UpdateLearningRate(size_t epoch, double error);


	friend class CNNbuilder;
};




#include "Builder/CNNbuilder.h"
