#ifndef __CurlFree__Select__
#define __CurlFree__Select__

#include <igl/embree/EmbreeIntersector.h>

//forward declaration of ViewerCore (needed for unprojection)
namespace igl { namespace viewer {
    class ViewerCore;
}}

class Select {
    public:
        Select(const Eigen::MatrixXd &V_,    // Vertices
                const Eigen::MatrixXi &F_,   // Faces
                const Eigen::MatrixXd &MF_,  // Face Barycenters
                const Eigen::MatrixXd &FN_,  // Face Normals
                const std::vector<std::vector<int> > &VF_, // Vertex->face adjacency
                const igl::viewer::ViewerCore &v,
                const int n); // Size of moving average for point smoothing.

        ~Select() { }

    private:
        const Eigen::MatrixXd &V;
        const Eigen::MatrixXi &F;
        const Eigen::MatrixXd &MF;
        const Eigen::MatrixXd &FN;
        const std::vector<std::vector<int> > &VF;
        igl::embree::EmbreeIntersector ei;
        const igl::viewer::ViewerCore &viewercore;

        std::vector<Eigen::RowVector3d> strokePoints;
        std::vector<int> strokedFaces;

        int m_addStrokePoint(int mouse_x, int mouse_y);
        void m_compute_new_smooth_points();
        void m_clearStroke();

    public:
        int strokeAdd(int mouse_x, int mouse_y);
        void strokeFinish(Eigen::VectorXi &cf,
                Eigen::MatrixXd &cfVel,
                Eigen::MatrixXd &cf_stroke);

        // Smoothed stroke
        Eigen::MatrixXd smoothStrokePoints;
        // Faces to which the smooth stroke points belong
        Eigen::VectorXi smoothStrokeFaces;
        int nw;
};

#endif
