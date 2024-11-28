#pragma once


#include <opencv2/opencv.hpp>
#include <gnuplot-include.h>
#include "../Utils/basic-includes.h"
#include "../CNN/Builder/CNNbuilder.h"




std::vector<CNN_DATA> LoadData_CNN(const std::string& folderPath)
{
    std::vector<CNN_DATA> set;

    int l = -1;

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (std::filesystem::is_regular_file(entry.path())) {

            std::string fileName = entry.path().filename().string();
            std::string labelStr = Utils::SplitString(fileName, "_")[0];
            size_t labelIndex = (size_t)std::stoi(labelStr);

            size_t outputSize = 10;
            auto label = std::vector<double>(outputSize, 0.0);
            label[labelIndex] = 1.0;

            std::string fullPathName = entry.path().string();
            Eigen::MatrixXd input = Utils::ImageToMatrix(cv::imread(fullPathName));

            CNN_DATA cnnData{ input, labelIndex };
            cnnData.label = label;

            set.push_back(cnnData);

            if (labelIndex != l) {
                l = labelIndex;
                std::cout << "load data: [" << (labelIndex+1)*10 << "%]\n";
            }
        }
    }

    return set;
};


Eigen::MatrixXd TestingModelAccuracy(CNN* cnn, std::vector<CNN_DATA> testSet, double* accuracy)  // "..\\..\\.resources\\test"
{
    Eigen::MatrixXd confusionMatrix = Eigen::MatrixXd::Zero(10, 10);
    int totalData = 0;
    int errors = 0;

    for (auto& testData : testSet) {

        std::vector<double> givenOutput = cnn->Forward(testData.input);

        auto it = std::max_element(givenOutput.begin(), givenOutput.end());
        size_t givenLabel = std::distance(givenOutput.begin(), it);

        confusionMatrix(givenLabel, testData.labelIndex) += 1.0;

        totalData++;

        if (givenLabel != testData.labelIndex) { errors++; }

    }

    (*accuracy) = 1.0 - ((double)errors/totalData);

    return confusionMatrix;
}


void DecreaseLearningRate(size_t epoch, double error, double& learnRate)
{
    std::cout << "\nmodel accuracy: " << 1.0 - error << "\n";
    if (1.0-error >= 0.965) {
        learnRate  =  0.95*learnRate;
    }
}


