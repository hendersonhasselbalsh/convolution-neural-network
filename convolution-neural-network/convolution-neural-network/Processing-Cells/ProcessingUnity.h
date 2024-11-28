#pragma once


#include "../Utils/basic-includes.h" 
#include "../Utils/utils.h" 
#include "../FeedForward-Neural-Network/Activation-Func/activation-functions.h" 


class IProcessingUnit {
	public: 
		virtual Eigen::MatrixXd Forward(Eigen::MatrixXd& input) = 0;            /*pure virtual method*/
		virtual Eigen::MatrixXd Backward(Eigen::MatrixXd& dLoss_dOutput) = 0;   /*pure virtual method*/
		virtual void UpdateLearningRate(size_t epoch, double error, std::function<void(size_t, double, double&)> UpdateRule) { }
};


