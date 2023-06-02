#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// Linear model class
class LinearModel {
private:
    std::vector<double> weights;
    double learningRate;
    double threshold;
    bool isClassification; // Flag to indicate classification or regression

public:
    LinearModel(unsigned long long inputSize, double lr, double th, bool classification = false) {
        weights.resize(inputSize, 0.0);
        learningRate = lr;
        threshold = th;
        isClassification = classification;
    }

    double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
        double result = 0.0;
        for (int i = 0; i < a.size(); i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    int classify(const std::vector<double>& input) {
        double activation = dotProduct(input, weights);
        double output = sigmoid(activation);
        return output >= threshold ? 1 : 0;
    }

    double predict(const std::vector<double>& input) {
        return dotProduct(input, weights);
    }

    void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs) {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_int_distribution<int> distribution(0, inputs.size() - 1);

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.size(); i++) {
                int index = distribution(generator);
                const std::vector<double>& input = inputs[index];
                const std::vector<double>& target = targets[index];

                std::vector<double> prediction(target.size(), 0.0);
                for (int j = 0; j < target.size(); j++) {
                    prediction[j] = isClassification ? classify(input) : predict(input);
                    double error = target[j] - prediction[j];

                    for (int k = 0; k < weights.size(); k++) {
                        weights[k] += learningRate * error * input[k];
                    }
                }
            }
        }
    }
};

int function(int a){
    std::cout << "some texte and else " << a << std::endl;
    return a;
}

int main() {
    printf("%d\n", function(2));
    // Define classification data and targets
    std::vector<std::vector<double>> classificationData = {

            {224, 76, 64, 222, 37, 24, 203, 91, 84, 220, 226, 227, 197, 197, 198, 208, 208, 208, 238, 245, 245, 225, 150, 145, 206, 15, 6, 205, 24, 16, 224, 72, 61, 205, 11, 0, 204, 91, 82, 216, 223, 224, 193, 194, 194, 215, 215, 215, 234, 242, 242, 221, 148, 142, 205, 9, 0, 205, 20, 9, 230, 76, 65, 210, 14, 2, 208, 97, 89, 212, 225, 226, 206, 160, 156, 230, 161, 156, 227, 238, 239, 213, 144, 139, 209, 14, 2, 206, 24, 11, 240, 92, 82, 238, 49, 36, 212, 100, 92, 213, 210, 210, 228, 116, 109, 243, 109, 101, 211, 203, 203, 203, 132, 127, 213, 21, 7, 208, 28, 15, 245, 106, 99, 244, 69, 60, 211, 94, 86, 228, 101, 92, 244, 84, 75, 245, 95, 86, 204, 59, 51, 212, 123, 117, 215, 27, 13, 213, 36, 22, 239, 90, 81, 242, 67, 57, 213, 101, 93, 237, 120, 112, 232, 36, 23, 239, 66, 55, 202, 51, 41, 230, 147, 142, 214, 29, 16, 222, 47, 35, 233, 82, 71, 240, 65, 54, 220, 117, 109, 223, 207, 206, 202, 46, 37, 228, 66, 55, 203, 146, 143, 235, 167, 163, 212, 31, 17, 229, 58, 46, 231, 81, 72, 238, 66, 55, 227, 124, 116, 195, 203, 204, 199, 174, 173, 209, 183, 181, 215, 218, 218, 215, 147, 142, 212, 35, 20, 236, 67, 56, 233, 78, 67, 236, 55, 43, 221, 120, 113, 186, 195, 195, 205, 198, 197, 206, 194, 193, 221, 230, 231, 203, 135, 130, 215, 35, 21, 239, 74, 64, 230, 73, 62, 224, 38, 25, 207, 96, 89, 192, 199, 199, 194, 195, 195, 219, 220, 220, 216, 223, 223, 205, 139, 134, 217, 37, 23, 241, 78, 70}


    };

    std::vector<std::vector<double>> classificationTargets = {
            {1},
            {1},
            {2}

    };

    // Create and train linear model for classification
    LinearModel classificationModel(classificationData.size(), 0.3, 0.5, true);
    classificationModel.train(classificationData, classificationTargets, 1000);

    // Output classification predictions
    std::cout << "Classification Predictions:" << std::endl;
    for (const auto& input : classificationData) {
        int prediction = classificationModel.classify(input);
        std::cout << prediction << std::endl;
    }

    return 0;
}
