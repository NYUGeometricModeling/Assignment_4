#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>
#include <igl/barycenter.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/per_face_normals.h>
#include <igl/slice.h>
#include <igl/avg_edge_length.h>
#include <igl/file_dialog_open.h>
/*** insert any libigl headers here ***/
#include <memory>

#include "Select.h"

using namespace std;
using Viewer = igl::viewer::Viewer;

// Vertex array, #V x3
Eigen::MatrixXd V(0,3);
// Face array, #F x3
Eigen::MatrixXi F(0,3);
// Face barycenter array, #F x3
Eigen::MatrixXd MF(0,3);
// Face normal array, #F x3
Eigen::MatrixXd FN(0,3);
// Vertex-to-face adjacency
std::vector<std::vector<int> > VF, VFi;

// Face constraint painting
std::unique_ptr<Select> selector;
bool    selection_mode = false;
bool activelySelecting = false;
Eigen::VectorXi selected_faces;
Eigen::MatrixXd selected_vec3(0, 3),
                selection_stroke_points(0, 3);

// Face vector constraints: face indices and prescribed vector constraints
Eigen::VectorXi constraint_fi;
Eigen::MatrixXd constraint_vec3(0, 3);

// Scale for displaying vectors
double vScale = 0;

// Texture image (grayscale)
Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> texture_I;

// Output: vector field (one vector per face), #F x3
Eigen::MatrixXd vfield(0, 3);
// Output: scalar function computed from Poisson reconstruction (one value per vertex), #V x1
Eigen::VectorXd sfield;
// Output: scalar function's gradient, #F x3
Eigen::MatrixXd sfield_grad(0, 3);
// Output: per-vertex uv coordinates, #V x2
Eigen::MatrixXd UV(0, 2);
// Output: boolean flag for flipped faces, #F x1
Eigen::VectorXi is_flipped;
// Output: per-face color array, #F x3
Eigen::MatrixXd face_colors;

// Function declarations (see below for implementation)
void clearSelection();
void applySelection();
void loadConstraints();
void saveConstraints();

bool callback_key_down  (Viewer &viewer, unsigned char key, int modifiers);
bool callback_mouse_down(Viewer &viewer, int button,  int modifier);
bool callback_mouse_move(Viewer &viewer, int mouse_x, int mouse_y);
bool callback_mouse_up  (Viewer &viewer, int button,  int modifier);
void line_texture();
Eigen::MatrixXd readMatrix(const std::string &filename);

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        // Draw selection and constraints only
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;

        // Initialize face colors
        face_colors = Eigen::MatrixXd::Constant(F.rows(), 3, 0.9);
        // Color selected faces...
        for (int i = 0; i < selected_faces.rows(); ++i)
            face_colors.row(selected_faces[i]) << 231. / 255, 99. / 255, 113. / 255.;

        // ... and constrained faces
        for (int i = 0; i < constraint_fi.size(); ++i)
            face_colors.row(constraint_fi(i)) << 69 / 255., 163 / 255., 232. / 255;
        viewer.data.set_colors(face_colors);

        // Draw selection vectors
        Eigen::MatrixXd MF_s;
        igl::slice(MF, selected_faces, 1, MF_s);
        viewer.data.add_edges(
                MF_s,
                MF_s + vScale * selected_vec3,
                Eigen::RowVector3d(0, 1, 0));

        // Draw constraint vectors
        igl::slice(MF, constraint_fi, 1, MF_s);
        viewer.data.add_edges(
                MF_s,
                MF_s + vScale * constraint_vec3,
                Eigen::RowVector3d(0, 0, 1));


        // Draw the stroke path
        int ns = selection_stroke_points.rows();
        if (ns) {
            viewer.data.add_points(selection_stroke_points, Eigen::RowVector3d(0.4, 0.4, 0.4));
            viewer.data.add_edges(selection_stroke_points.   topRows(ns - 1),
                                  selection_stroke_points.bottomRows(ns - 1),
                                  Eigen::RowVector3d(0.7, 0.7, 0.7));
        }
    }

    if (key == '2') {
        // Field interpolation
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;

        // Show constraints
        // Initialize face colors
        face_colors = Eigen::MatrixXd::Constant(F.rows(), 3, 0.9);
        // Color the constrained faces
        for (int i = 0; i < constraint_fi.size(); ++i)
            face_colors.row(constraint_fi(i)) << 69 / 255., 163 / 255., 232. / 255;
        viewer.data.set_colors(face_colors);

        // Draw constraint vectors
        Eigen::MatrixXd MF_s;
        igl::slice(MF, constraint_fi, 1, MF_s);
        viewer.data.add_edges(MF_s, MF_s + vScale * constraint_vec3, Eigen::RowVector3d(0, 0, 1));

        // Add your code for interpolating a vector field here
    }

    if (key == '3') {
        // Scalar field reconstruction
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;

        // Draw constraint vectors
        Eigen::MatrixXd MF_s;
        igl::slice(MF, constraint_fi, 1, MF_s);
        viewer.data.add_edges(MF_s, MF_s + vScale * constraint_vec3, Eigen::RowVector3d(0, 0, 1));

        // Add your code for fitting and displaying the scalar function here
    }

    if (key == '4') {
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;

        // Add your code for computing hamonic parameterization here, store in UV

        viewer.data.set_uv(10 * UV);
        viewer.core.show_texture = true;
    }

    if (key == '5') {
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;

        // Add your code for computing LSCM parameterization here, store in UV

        viewer.data.set_uv(10 * UV);
        viewer.core.show_texture = true;
    }

    if (key == '6') {
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;
        // Add your code for displaying the gradient of one of the
        // parameterization functions
    }

    if (key == '7') {
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;
        // Add your code for replacing one of the parameterization functions
        // with the interpolated field
    }

    if (key == '8') {
        // Add your code for detecting and displaying flipped triangles in the
        // UV domain here
    }

    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage assignment4_bin mesh.obj" << endl;
        exit(0);
    }

    // Read mesh
    igl::readOFF(argv[1],V,F);

    // Plot the mesh
    Viewer viewer;
    viewer.callback_key_down = callback_key_down;
    callback_key_down(viewer, '1', 0);
    viewer.callback_mouse_down = callback_mouse_down;
    viewer.callback_mouse_move = callback_mouse_move;
    viewer.callback_mouse_up   = callback_mouse_up;

    viewer.callback_init = [&](Viewer &v) {
        v.ngui->addGroup("Selection");
        v.ngui->addVariable("Selection Mode", selection_mode);
        v.ngui->addButton("Clear Selection", [&]() {
            clearSelection();
            callback_key_down(v, '1', 0);
        });
        v.ngui->addButton("Apply Selection", [&]() {
            applySelection();
            callback_key_down(v, '1', 0);
        });
        v.ngui->addButton("Load Constraints", [&]() {
            loadConstraints();
            callback_key_down(v, '1', 0);
        });
        v.ngui->addButton("Save Constraints", [&]() {
            saveConstraints();
            callback_key_down(v, '1', 0);
        });

        v.screen->performLayout();

        return false;
    };

    // Compute face barycenters
    igl::barycenter(V, F, MF);

    // Compute face normals
    igl::per_face_normals(V, F, FN);

    // Compute vertex to face adjacency
    igl::vertex_triangle_adjacency(V, F, VF, VFi);

    // Initialize selector
    selector = std::unique_ptr<Select>(new Select(V, F, FN, viewer.core));

    // Initialize scale for displaying vectors
    vScale = 0.5 * igl::avg_edge_length(V, F);

    // Initialize texture image
    line_texture();

    // Initialize texture coordinates with something
    UV.setZero(V.rows(), 2);

    viewer.data.set_texture(texture_I, texture_I, texture_I);
    viewer.core.point_size = 10;

    viewer.launch();
}

void clearSelection() {
    selected_faces.resize(0);
    selected_vec3.resize(0, 3);
    selection_stroke_points.resize(0, 3);
}

void applySelection() {
    // Add selected faces and associated constraint vectors to the existing set.
    // On conflicts, we take the latest stroke.
    std::vector<bool> hasConstraint(F.rows());

    Eigen::VectorXi uniqueConstraintFi  (selected_faces.rows() + constraint_fi.rows());
    Eigen::MatrixXd uniqueConstraintVec3(selected_faces.rows() + constraint_fi.rows(), 3);

    int numConstraints = 0;
    auto applyConstraints = [&](const Eigen::VectorXi &faces, 
                                const Eigen::MatrixXd &vecs) {
        // Apply constraints in reverse chronological order
        for (int i = faces.rows() - 1; i >= 0; --i) {
            const int fi = faces[i];
            if (!hasConstraint.at(fi)) {
                hasConstraint[fi] = true;
                uniqueConstraintFi      [numConstraints] = fi;
                uniqueConstraintVec3.row(numConstraints) = vecs.row(i);
                ++numConstraints;
            }
        }
    };

    applyConstraints(selected_faces,   selected_vec3);
    applyConstraints(constraint_fi,  constraint_vec3);

    constraint_fi   = uniqueConstraintFi.  topRows(numConstraints);
    constraint_vec3 = uniqueConstraintVec3.topRows(numConstraints);

    clearSelection();
}

void clearConstraints() {
    constraint_fi.resize(0);
    constraint_vec3.resize(0, 3);
}

void loadConstraints() {
    clearConstraints();
    std::string filename = igl::file_dialog_open();
    if (!filename.empty()) {
        Eigen::MatrixXd mat = readMatrix(filename);
        constraint_fi   = mat.leftCols(1).cast<int>();
        constraint_vec3 = mat.rightCols(3);
    }
}

void saveConstraints() {
    std::string filename = igl::file_dialog_save();
    if (!filename.empty()) {
        Eigen::MatrixXd mat(constraint_fi.rows(), 4);
        mat.col(0)       = constraint_fi.cast<double>();
        mat.rightCols(3) = constraint_vec3;
        ofstream ofs(filename);
        ofs << mat << endl;
    }
}

bool callback_mouse_down(Viewer& viewer, int button, int modifier) {
    if (button == int(Viewer::MouseButton::Right))
        return false;

    if (selection_mode) {
        int fid = selector->strokeAdd(viewer.current_mouse_x, viewer.current_mouse_y);
        activelySelecting = fid >= 0;
        return activelySelecting;
    }

    return false;
}

bool callback_mouse_move(Viewer& viewer, int mouse_x, int mouse_y) {
    if (selection_mode && activelySelecting) {
        selector->strokeAdd(mouse_x, mouse_y);
        return true;
    }
    return false;
}

bool callback_mouse_up(Viewer& viewer, int button, int modifier) {
    if (activelySelecting) {
        selector->strokeFinish(selected_faces, selected_vec3, selection_stroke_points);
        activelySelecting = false;
        callback_key_down(viewer, '1', 0);
        return true;
    }

    return false;
};

void line_texture() {
    int size = 128;              // Texture size
    int w    = 7;                // Line width
    int pos  = size / 2 - w / 2; // Center the line
    texture_I.setConstant(size, size, 255);
    texture_I.block(0, pos, size, w).setZero();
    texture_I.block(pos, 0, w, size).setZero();
}

Eigen::MatrixXd readMatrix(const string &filename) {
    ifstream infile(filename);
    if (!infile.is_open())
        throw runtime_error("Failed to open " + filename);

    vector<double> data;
    size_t rows = 0, cols = 0;
    for (string line; getline(infile, line); ++rows) {
        stringstream ss(line);
        const size_t prevSize = data.size();
        copy(istream_iterator<double>(ss), istream_iterator<double>(),
             back_inserter(data));
        if (rows == 0) cols = data.size() - prevSize;
        if (cols != data.size() - prevSize) throw runtime_error("Unequal row sizes.");
    }

    Eigen::MatrixXd mat(rows, cols);
    for (int i = 0; i < int(rows); ++i)
        for (size_t j = 0; j < cols; ++j)
            mat(i, j) = data[i * cols + j];

    return mat;
}
