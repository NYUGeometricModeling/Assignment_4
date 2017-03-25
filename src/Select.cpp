#include "Select.h"
#include <igl/embree/unproject_onto_mesh.h>
#include <igl/viewer/ViewerCore.h>

#include <iostream>
#include <fstream>
#include <igl/matlab_format.h>
#include <igl/embree/line_mesh_intersection.h>
#include <igl/slice.h>
#include <igl/list_to_matrix.h>
#include <igl/unique.h>


using namespace igl;
using namespace std;

Select::Select(const Eigen::MatrixXd &V_,
               const Eigen::MatrixXi &F_,
               const Eigen::MatrixXd &MF_,
               const Eigen::MatrixXd &FN_,
               const std::vector<std::vector<int> > &VF_,
               const igl::viewer::ViewerCore &v,
               const int n)
      : V(V_),
        F(F_),
        MF(MF_),
        FN(FN_),
        VF(VF_),
        viewercore(v),
        nw(n)
{
    ei.init(V.cast<float>(), F, true);
}

int Select::strokeAdd(int mouse_x,
                      int mouse_y) {
    int fi = m_addStrokePoint(mouse_x, mouse_y);
    m_compute_new_smooth_points();
    return fi;
}

int Select::m_addStrokePoint(int mouse_x, int mouse_y) {
    // Cast a ray in the view direction starting from the mouse position
    double x = mouse_x;
    double y = viewercore.viewport(3) - mouse_y;

    int fi;
    Eigen::Vector3f baryCoords;
    bool hit = igl::embree::unproject_onto_mesh(Eigen::Vector2f(x,y),
            F,
            viewercore.view * viewercore.model,
            viewercore.proj,
            viewercore.viewport,
            ei,
            fi, baryCoords);
    if (!hit) return -1;

    assert(fi <= F.rows());
    strokePoints.push_back(
        V.row(F(fi, 0)) * baryCoords[0] +
        V.row(F(fi, 1)) * baryCoords[1] +
        V.row(F(fi, 2)) * baryCoords[2]);
    strokedFaces.push_back(fi);
    // std::cout << "hit triangle: " << fi << std::endl;
    // std::cout << "At point: " << strokePoints.back() << std::endl;
    // std::cout << "Barycoords" << baryCoords.transpose() << std::endl;
    return fi;
}

void Select::m_clearStroke() {
    strokePoints.clear();
    strokedFaces.clear();
    smoothStrokePoints.resize(0, 3);
    smoothStrokeFaces.resize(0);
}

void Select::strokeFinish(Eigen::VectorXi &cf,
                          Eigen::MatrixXd &cfVel,
                          Eigen::MatrixXd &cf_stroke)
{
    cf_stroke = smoothStrokePoints;

    // We need at least two points to determine a constraint vector
    if (smoothStrokePoints.rows() < 2) {
        cf.resize(0);
        cfVel.resize(0, 3);
        m_clearStroke();
        return;
    }

    // Get unique faces in stroke, but *do not* change the order
    //   IA  #C index vector so that unique_faces = A(IA);
    //   IC  #A index vector so that A = unique_faces(IC);
    Eigen::VectorXi unique_faces, IA, IC;
    igl::unique(smoothStrokeFaces, unique_faces, IA, IC);

    cf   .resize(smoothStrokeFaces.size(), 1);
    cfVel.resize(smoothStrokeFaces.size(), 3);
    int num = 0;
    for (int i = 0; i < smoothStrokeFaces.size(); ++i) {
        // Detect entrance into a unique face
        // (JP: faces will repeat if the path returns!)
        if ((i == 0) || (IC[i] != IC[i - 1])) {
            Eigen::RowVector3d vec;
            if (i < smoothStrokePoints.rows() - 1) {
                // prefer forward difference as it is probably coming from the same face
                vec = smoothStrokePoints.row(i + 1) - smoothStrokePoints.row(i);
            }
            else {
                assert(i > 0);
                vec = smoothStrokePoints.row(i) - smoothStrokePoints.row(i - 1);
            }

            int fi = smoothStrokeFaces[i];
            cfVel.row(num) = (vec - vec.dot(FN.row(fi)) * FN.row(fi)).normalized();
            cf(num) = fi;
            ++num;
        }
    }

    cf.conservativeResize(num, Eigen::NoChange);
    cfVel.conservativeResize(num, Eigen::NoChange);

    m_clearStroke();
}

void Select::m_compute_new_smooth_points() {
    double tol = 0.001;
    smoothStrokeFaces .conservativeResize(strokedFaces.size(), 1);
    smoothStrokePoints.conservativeResize(strokedFaces.size(), 3);

    // only points N-nw:end need recomputation
    for (int i = std::max<int>(0, int(strokePoints.size()) - nw - 1); i < int(strokePoints.size()); ++i) {
        // moving average
        Eigen::RowVector3d p(Eigen::RowVector3d::Zero());
        int num = 0;
        for (int k = -nw; k <= nw; ++k) {
            int j = i + k;
            if (j < 0 || j >= strokePoints.size())
                continue;
            p += strokePoints[j];
            ++num;
        }
        p /= double(num);

        // projection
        igl::Hit A, B;
        Eigen::RowVector3d A_dir = FN.row(strokedFaces[i]);
        Eigen::RowVector3d A_pos = p - tol * A_dir;
        bool A_hit = ei.intersectRay(A_pos.cast<float>(), A_dir.cast<float>(), A);

        Eigen::RowVector3d B_dir = -FN.row(strokedFaces[i]);
        Eigen::RowVector3d B_pos = p - tol * B_dir;
        bool B_hit = ei.intersectRay(B_pos.cast<float>(), B_dir.cast<float>(), B);

        int choice = -1;
        if (A_hit && !B_hit)
            choice = 0;
        else if (!A_hit && B_hit)
            choice = 1;
        else if (A_hit && B_hit)
            choice = A.t > B.t;

        if (choice == -1) {
            assert(false);
        }
        else if (choice == 0) {
            smoothStrokePoints.row(i) = (1.0f - A.u - A.v) * V.row(F(A.id, 0)) +
                                                      A.u  * V.row(F(A.id, 1)) +
                                                      A.v  * V.row(F(A.id, 2));
            assert(A.id < F.rows());
            smoothStrokeFaces[i] = A.id;
        }
        else if (choice == 1) {
            smoothStrokePoints.row(i) = (1.0f - B.u - B.v) * V.row(F(B.id, 0)) +
                                                      B.u  * V.row(F(B.id, 1)) +
                                                      B.v  * V.row(F(B.id, 2));
            assert(B.id < F.rows());
            smoothStrokeFaces[i] = B.id;
        }
    }
}
