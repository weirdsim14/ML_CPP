#include <Eigen/Dense>

class KalmanFilter {
private:
    Eigen::MatrixXd F, B, H, Q, R, P;
    Eigen::VectorXd u, x;

public:
    KalmanFilter(const Eigen::MatrixXd& F, const Eigen::MatrixXd& B, const Eigen::VectorXd& u,
                 const Eigen::MatrixXd& H, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                 const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0) : F(F), B(B), u(u), H(H), Q(Q), R(R), x(x0), P(P0) {}

    void predict() {
        x = F * x + B * u;
        P = F * P * F.transpose() + Q;
    }

    void update(const Eigen::VectorXd& z) {
        Eigen::VectorXd y = z - H * x;
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();
        x = x + K * y;
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(P.rows(), P.cols());
        P = (I - K * H) * P;
    }
};
