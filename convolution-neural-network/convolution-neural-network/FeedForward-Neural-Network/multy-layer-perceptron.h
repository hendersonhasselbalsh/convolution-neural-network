#pragma once

#include "../Utils/basic-includes.h"
#include "../Utils/utils.h"
#include "Layer/layer.h"

#define INPUT first
#define LABEL second
using MLPTrainigData = std::pair<std::vector<double>, std::vector<double>>;


class MlpBuilder;
class CNNbuilder;
class CNN;



class MLP {

	private:
	//--- atributos importantes do mlp
		std::vector<Layer> _layers;
		ILostFunction* _lostFunction;

		std::function<std::vector<double>(size_t)> ParseLabelToVector;
		std::function<void(size_t, double, double&)> UpdateLeraningRate;                      //  double f(size_t epoch, double accuracy, double currentLearningRate)
		std::string _outFile;

		size_t _inputSize;
		size_t _maxEpochs;
		double _acceptableAccuracy;
		double _error;
		std::vector<double> _accumulatedGradients;


	//--- construtor privado (usado pelo builder)
		MLP();


	//--- backward and forward
		std::vector<double> Foward(std::vector<double> input);
		std::vector<double> Backward(std::vector<double> predictedValues, std::vector<double> correctValues);
		std::vector<double> Backward(std::vector<double> lossGradientWithRespectToOutput);
		
		void BuildJson();
		Json ToJson() const;

		void ChangeLearningRate(size_t epoch, double error);
		void CalculateError(std::vector<double> predictedValues, std::vector<double> correctValues);
		

	public:
		~MLP();

		void Training(std::vector<MLPTrainigData> trainigSet, std::function<void(void)> callback = [](){} );
		void Training(std::vector<MLP_DATA> trainigSet, std::function<void(void)> callback = [](){} );


		std::vector<double> Classify(std::vector<double> input);
		size_t Classify(std::vector<double> input, std::function<size_t(std::vector<double>)> ParseOutputToLabel);
		void Classify(std::vector<std::vector<double>> inputSet, std::function<void(std::vector<double>)> CallBack);
		void Classify(std::vector<MLP_DATA> inputSet, std::function<void(std::vector<double>)> CallBack);


		Layer& operator[](size_t layerIndex);
		Layer& LastLayer();


		friend class MlpBuilder;
		friend class CNNbuilder;
		friend class CNN;

};


#include "Builder/mlp-builder.h"

