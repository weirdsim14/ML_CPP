#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

double mahalanobisDistance(const VectorXd &point, const VectorXd &mean, const MatrixXd &covarianceMatrix) {
    VectorXd diff = point - mean;
    return sqrt(diff.transpose() * covarianceMatrix.inverse() * diff);
}

int main() {
    // Example point and distribution
    VectorXd point(3);
    point << 1, 2, 3;

    VectorXd mean(3);
    mean << 1, 2, 1;

    MatrixXd covarianceMatrix(3, 3);
    covarianceMatrix << 1, 0, 0,
                        0, 1, 0,
                        0, 0, 1;

    // Calculate Mahalanobis distance
    double distance = mahalanobisDistance(point, mean, covarianceMatrix);
    cout << "Mahalanobis Distance: " << distance << endl;

    return 0;
}
