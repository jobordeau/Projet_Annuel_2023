
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

class MyMLP {
private:
    std::vector<std::vector<std::vector<double>>> W;
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> deltas;
    std::vector<int> d;
    unsigned long long L;

public:
    MyMLP(const std::vector<int>& npl) {
        d = npl;
        L = d.size() - 1;
        std::cout <<"start creating the mpl"<< std::endl;
        for (int l = 0; l <= L; l++) {
            W.emplace_back();

            if (l == 0) {
                continue;
            }

            for (int i = 0; i <=npl[l - 1]; i++) {
                W[l].emplace_back();

                for (int j = 0; j <= npl[l]; j++) {
                    W[l][i].push_back((j == 0) ? 0.0 : getRandomUniform(-1.0, 1.0));

                }
            }
        }

        for (int l = 0; l <= L; l++) {
            X.emplace_back();
            for (int j = 0; j <= npl[l]; j++) {
                X[l].push_back((j == 0) ? 1.0 : 0.0);
            }
        }


        for (int l = 0; l <= L; l++) {
            deltas.emplace_back();
            for (int j = 0; j <= npl[l]; j++) {
                deltas[l].push_back(0.0);

            }
        }
        std::cout << "mlp created" <<  std::endl;
    }
    void propagate(const std::vector<double>& inputs, bool is_classification) {

        for (int j = 0; j < d[0] ; j++) {
            X[0][j + 1] = inputs[j];
        }

        for (int l = 1; l < L+1; l++) {
            for (int j = 1; j < d[l] + 1; j++) {
                double total = 0.0;
                for (int i = 0; i < d[l - 1] + 1; i++) {
                    total += W[l][i][j] * X[l - 1][i];

                }

                if (l < L || is_classification) {
                    total = tanh(total);
                }

                X[l][j] = total;
            }
        }
    }

    std::vector<double> predict(const std::vector<double>& inputs, bool is_classification) {
        propagate(inputs, is_classification);
        std::cout << " propagate work" << std::endl;
        return std::vector<double>(X[L].begin() + 1, X[L].end());
    }

    void train(const std::vector<std::vector<double>>& all_samples_inputs,
               const std::vector<std::vector<double>>& all_samples_expected_outputs,
               bool is_classification, int iteration_count, double alpha) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, all_samples_inputs.size() - 1);

        std::cout << "start training" <<  std::endl;
        for (int it = 0; it < iteration_count; it++) {
            int k = dis(gen);
            const std::vector<double>& inputs_k = all_samples_inputs[k];
            const std::vector<double>& y_k = all_samples_expected_outputs[k];

            propagate(inputs_k, is_classification);

            for (int j = 1; j < d[L] + 1; j++) {
                deltas[L][j] = (X[L][j] - y_k[j - 1]);
                if (is_classification) {
                    deltas[L][j] *= (1 - pow(X[L][j], 2));
                }

            }

            for (int l = L ; l >= 1; l--) {
                for (int i = 1; i < d[l - 1] + 1; i++) {
                    double total = 0.0;
                    for (int j = 1; j < d[l]  + 1; j++) {
                        total += W[l][i][j] * deltas[l][j];

                    }
                    deltas[l-1][i] = (1 - X[l-1][i] * X[l-1][i]) * total;
                }
            }

            for (int l = 1; l < L+1; l++) {
                for (int i = 0; i < d[l - 1] + 1; i++) {
                    for (int j = 1; j < d[l] + 1; j++) {
                        W[l][i][j] -= alpha * X[l - 1][i] * deltas[l][j];



                    }
                }
            }
        }
    }

private:
    double static getRandomUniform(double min, double max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min, max);
        return dis(gen);
    }
};


int main() {
    std::vector<std::vector<double>> all_samples_inputs = {
            {43, 82, 147, 41, 81, 146, 42, 82, 147, 51, 91, 148, 58, 98, 150, 66, 102, 152, 91, 122, 162, 105, 128, 167, 92, 118, 162, 48, 89, 150, 44, 86, 150, 41, 84, 145, 45, 85, 149, 48, 89, 150, 64, 100, 153, 107, 130, 167, 129, 142, 172, 111, 137, 173, 88, 116, 161, 49, 88, 149, 43, 86, 153, 64, 98, 147, 57, 90, 149, 60, 98, 162, 152, 171, 195, 186, 174, 183, 203, 44, 54, 151, 81, 100, 109, 133, 169, 109, 133, 167, 39, 85, 154, 77, 114, 164, 66, 92, 151, 38, 60, 119, 197, 201, 209, 222, 218, 218, 158, 40, 41, 153, 9, 11, 134, 132, 153, 141, 160, 184, 58, 96, 158, 86, 118, 165, 82, 113, 171, 33, 67, 141, 94, 108, 143, 183, 190, 194, 179, 96, 99, 176, 0, 0, 160, 67, 81, 125, 151, 182, 81, 111, 165, 120, 140, 176, 103, 128, 177, 59, 91, 169, 132, 152, 197, 195, 202, 207, 146, 118, 127, 131, 4, 8, 135, 19, 31, 76, 105, 152, 99, 124, 170, 141, 158, 185, 120, 139, 180, 48, 80, 156, 119, 139, 185, 243, 244, 242, 188, 179, 182, 153, 77, 77, 138, 55, 57, 101, 99, 130, 112, 134, 175, 153, 167, 191, 115, 135, 179, 41, 77, 153, 104, 131, 179, 206, 219, 229, 208, 193, 199, 148, 56, 76, 107, 75, 114, 70, 99, 153, 94, 123, 172, 123, 145, 182, 94, 120, 170, 58, 97, 165, 62, 101, 165, 63, 103, 164, 72, 109, 167, 60, 100, 164, 56, 102, 166, 59, 100, 163, 86, 116, 168, 97, 128, 176, 78, 114, 170, 62, 102, 169, 63, 103, 168, 62, 102, 165, 61, 102, 165, 62, 103, 165, 62, 102, 164, 62, 102, 164},
            {1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2},
            {224, 76, 64, 222, 37, 24, 203, 91, 84, 220, 226, 227, 197, 197, 198, 208, 208, 208, 238, 245, 245, 225, 150, 145, 206, 15, 6, 205, 24, 16, 224, 72, 61, 205, 11, 0, 204, 91, 82, 216, 223, 224, 193, 194, 194, 215, 215, 215, 234, 242, 242, 221, 148, 142, 205, 9, 0, 205, 20, 9, 230, 76, 65, 210, 14, 2, 208, 97, 89, 212, 225, 226, 206, 160, 156, 230, 161, 156, 227, 238, 239, 213, 144, 139, 209, 14, 2, 206, 24, 11, 240, 92, 82, 238, 49, 36, 212, 100, 92, 213, 210, 210, 228, 116, 109, 243, 109, 101, 211, 203, 203, 203, 132, 127, 213, 21, 7, 208, 28, 15, 245, 106, 99, 244, 69, 60, 211, 94, 86, 228, 101, 92, 244, 84, 75, 245, 95, 86, 204, 59, 51, 212, 123, 117, 215, 27, 13, 213, 36, 22, 239, 90, 81, 242, 67, 57, 213, 101, 93, 237, 120, 112, 232, 36, 23, 239, 66, 55, 202, 51, 41, 230, 147, 142, 214, 29, 16, 222, 47, 35, 233, 82, 71, 240, 65, 54, 220, 117, 109, 223, 207, 206, 202, 46, 37, 228, 66, 55, 203, 146, 143, 235, 167, 163, 212, 31, 17, 229, 58, 46, 231, 81, 72, 238, 66, 55, 227, 124, 116, 195, 203, 204, 199, 174, 173, 209, 183, 181, 215, 218, 218, 215, 147, 142, 212, 35, 20, 236, 67, 56, 233, 78, 67, 236, 55, 43, 221, 120, 113, 186, 195, 195, 205, 198, 197, 206, 194, 193, 221, 230, 231, 203, 135, 130, 215, 35, 21, 239, 74, 64, 230, 73, 62, 224, 38, 25, 207, 96, 89, 192, 199, 199, 194, 195, 195, 219, 220, 220, 216, 223, 223, 205, 139, 134, 217, 37, 23, 241, 78, 70}


    };

    std::vector<std::vector<double>> all_samples_expected_outputs = {
            {1},
            {1},
            {2}
    };


    std::vector<int> npl = {100,50,1};
    MyMLP mlp(npl);
    std::cout << "mlp ok" <<  std::endl;
    std::vector<double> first;

    for (const auto& input : all_samples_inputs) {
        first =  mlp.predict(input, true);
        for (const auto& output:first){
            std::cout << "output before train: "<< output << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

    }

    bool is_classification = true;
    int iteration_count = 1000;
    double alpha = 0.2;
    mlp.train(all_samples_inputs, all_samples_expected_outputs, is_classification, iteration_count, alpha);


    for (const auto& input : all_samples_inputs) {
        for(const auto& output: mlp.predict(input,true)){
            std::cout << "output after train: " << output << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

    }


    return 0;
}


