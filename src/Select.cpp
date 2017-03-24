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
  int fi = addStrokePoint(mouse_x, mouse_y);
  if (fi >= 0) {
        // now we have a velocity constraint; add it as a constraint to the face
        // at the end of the strokedFaces list
        addConstraintToLast();
  }
  compute_new_smooth_points();

  //  last step: add the currently selected face and mouse position to the list
  return fi;
}

int Select::addStrokePoint(int mouse_x, int mouse_y) {
    // Cast a ray in the view direction starting from the mouse position
    double x = mouse_x;
    double y = viewercore.viewport(3) - mouse_y;

    Eigen::MatrixXd obj;
    Eigen::Vector3f baryCoords;

    int fi;
    bool hit = igl::embree::unproject_onto_mesh(Eigen::Vector2f(x,y),
            F,
            viewercore.view * viewercore.model,
            viewercore.proj,
            viewercore.viewport,
            ei,
            fi, baryCoords);
    assert(fi <= F.rows());

    if (hit) {
        strokePoints.push_back(
            (V.row(F(fi, 0)) * baryCoords[0],
             V.row(F(fi, 1)) * baryCoords[1],
             V.row(F(fi, 2)) * baryCoords[2]).eval());
        strokedFaces.push_back(fi);
        std::cout << "hit triangle: " << fi << std::endl;
        std::cout << "At point: " << strokePoints.back() << std::endl;
        return fi;
    }
    return -1;
}

void Select::strokeFinish(Eigen::VectorXi &cf,
                          Eigen::MatrixXd &cfVel,
                          Eigen::MatrixXd &cf_stroke) {
    cf.resize(0,1);
    cfVel.resize(0,3);
    cf_stroke.resize(0,3);

    // Get unique faces in cf, but *do not* change the order
    //   IA  #C index vector so that unique_faces = A(IA);
    //   IC  #A index vector so that A = unique_faces(IC);
    Eigen::VectorXi unique_faces, IA, IC;
    igl::unique(smoothStrokeFaces, unique_faces, IA, IC);

    cf.resize(smoothStrokeFaces.size(), 1);
    cfVel.resize(smoothStrokeFaces.size(), 3);
    int num = 0;
    for (int i = 0; i < smoothStrokeFaces.size(); ++i) {
        // Detect entrance into a unique face
        // (JP: faces repeat if path returns!)
        if (i == 0 || (IC[i] != IC[i - 1])) {
            int fi = smoothStrokeFaces[i];
            Eigen::RowVector3d vec;
            if (i < smoothStrokePoints.rows() - 1) {
                // prefer forward difference as it is probably coming from the same face
                vec = smoothStrokePoints.row(i + 1) - smoothStrokePoints.row(i);
            }
            else {
                assert(i > 0);
                vec = smoothStrokePoints.row(i) - smoothStrokePoints.row(i - 1);
            }
            vec = (vec - vec.dot(FN.row(fi)) * FN.row(fi)).normalized();
            cfVel.row(num) = vec;
            cf(num) = fi;
            ++num;
        }
    }

    cf.conservativeResize(num, Eigen::NoChange);
    cfVel.conservativeResize(num, Eigen::NoChange);
    cf_stroke = smoothStrokePoints;

    // Averaging is bad, as we might miss turns that pass through the same face twice.

    strokePoints.clear();
    strokedFaces.clear();
    strokedFacesVel.clear();
    smoothStrokePoints.resize(0, 3);
    smoothStrokeFaces.resize(0, 1);
}

void Select::addConstraintToLast() {
    size_t L = strokedFaces.size();
    if (L == 1)
        return;

    Eigen::RowVector3d velocity;
    velocity <<
        strokePoints[L - 1][0] - strokePoints[L - 2][0],
        strokePoints[L - 1][1] - strokePoints[L - 2][1],
        strokePoints[L - 1][2] - strokePoints[L - 2][2];

    strokedFacesVel.push_back(velocity);
}

void Select::compute_new_smooth_points() {
    double tol = 0.00001;
    igl::Hit A, B;
    smoothStrokePoints.conservativeResize(strokePoints.size(), 3);
    smoothStrokeFaces .conservativeResize(strokePoints.size(), 1);
    // only points N-n:end need recomputation
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
        Eigen::RowVector3d A_pos = p + tol * FN.row(strokedFaces[i]);
        Eigen::RowVector3d A_dir = -FN.row(strokedFaces[i]);
        std::cout << "checking ray (" << A_pos << ", " << A_dir << ")" << std::endl;
        bool A_hit = ei.intersectRay(A_pos.cast<float>(), A_dir.cast<float>(), A);

        Eigen::RowVector3d B_pos = p - tol * FN.row(strokedFaces[i]);
        Eigen::RowVector3d B_dir = FN.row(strokedFaces[i]);
        std::cout << "checking ray (" << B_pos << ", " << B_dir << ")" << std::endl;
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
