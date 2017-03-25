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

// For selecting faces
std::unique_ptr<Select> selector;
bool selection_mode = false;
bool activelySelecting = false;
Eigen::VectorXi selected_faces;
Eigen::MatrixXd selected_vec3(0, 3),
                selection_stroke_points(0, 3);

// Face vector constraints: face indices and prescribed vector constraints
Eigen::VectorXi constraint_fi;
Eigen::MatrixXd constraint_vec3(0, 3);

//scale for displaying vectors
double vScale;

//texture image (grayscale)
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

//function declarations (see below for implementation)
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
        // draw selection and constraints only
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;

        // Initialize face colors
        face_colors = Eigen::MatrixXd::Constant(F.rows(),3,.9);
        //color selected faces and constrained faces
        //first, color selection
        for (int i = 0; i < selected_faces.rows(); ++i)
            face_colors.row(selected_faces[i]) << 231./255, 99./255, 113./255.;

        //then, color constraints
        for (int i = 0; i<constraint_fi.size(); ++i)
            face_colors.row(constraint_fi(i)) << 69/255.,163/255.,232./255;
        viewer.data.set_colors(face_colors);

        //draw selection vectors
        Eigen::MatrixXd MF_s;
        igl::slice(MF, selected_faces, 1, MF_s);
        viewer.data.add_edges(
                MF_s,
                MF_s + vScale*selected_vec3,
                Eigen::RowVector3d(0,1,0));

        //draw constraint vectors
        igl::slice(MF, constraint_fi, 1, MF_s);
        viewer.data.add_edges(
                MF_s,
                MF_s + vScale * constraint_vec3,
                Eigen::RowVector3d(0,0,1));


        //draw the stroke of the selection
        int ns = selection_stroke_points.rows();
        if (ns) {
            viewer.data.add_points(selection_stroke_points,Eigen::RowVector3d(0.4,0.4,0.4));
            viewer.data.add_edges(selection_stroke_points.topRows(ns-1), selection_stroke_points.bottomRows(ns-1), Eigen::RowVector3d(0.7,0.7,.7));
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
        face_colors = Eigen::MatrixXd::Constant(F.rows(),3,.9);
        // Color the constrained faces
        for (int i = 0; i<constraint_fi.size(); ++i)
            face_colors.row(constraint_fi(i)) << 69/255.,163/255.,232./255;
        viewer.data.set_colors(face_colors);

        // Draw constraint vectors
        Eigen::MatrixXd MF_s;
        igl::slice(MF,constraint_fi, 1, MF_s);
        viewer.data.add_edges(MF_s, MF_s + vScale*constraint_vec3 , Eigen::RowVector3d(0,0,1));

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
        igl::slice(MF,constraint_fi, 1, MF_s);
        viewer.data.add_edges(MF_s, MF_s + vScale*constraint_vec3 , Eigen::RowVector3d(0,0,1));

        // Add your code for fitting and displaying the scalar function here
    }

    if (key == '4') {
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;

        // Add your code for computing hamonic parameterization here, store in UV

        viewer.data.set_uv(10*UV);
        viewer.core.show_texture = true;
    }

    if (key == '5') {
        viewer.data.clear();
        viewer.data.set_mesh(V, F);
        viewer.data.set_texture(texture_I, texture_I, texture_I);
        viewer.core.show_texture = false;

        // Add your code for computing LSCM parameterization here, store in UV

        viewer.data.set_uv(10*UV);
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
        cout << "Usage ex1_bin mesh.obj" << endl;
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
    viewer.callback_mouse_up = callback_mouse_up;

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
        v.ngui->addButton("Load Selection", [&]() {
            loadConstraints();
            callback_key_down(v, '1', 0);
        });
        v.ngui->addButton("Save Selection", [&]() {
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
    igl::vertex_triangle_adjacency(V,F,VF,VFi);

    // Initialize selector
    selector = std::unique_ptr<Select>(new Select(V, F, MF, FN, VF, viewer.core, 2));

    // Initialize scale for displaying vectors
    vScale = .5 * igl::avg_edge_length(V,F);

    // Initialize texture image
    line_texture();

    // Initialize texture coordinates with something
    UV.setZero(V.rows(),2);

    viewer.data.set_texture(texture_I, texture_I, texture_I);
    viewer.core.point_size = 10;

    viewer.launch();
}

void clearSelection() {
    selected_faces.resize(0);
    selected_vec3.resize(0,3);
    selection_stroke_points.resize(0, 3);
}

void applySelection() {
    // Add selected faces and associated vectors to the existing constraints,
    // overwriting on conflicts.

    // First, mark which faces are affected by the new constraints
    std::vector<bool> hasNewConstraint(F.rows());
    for (int i = 0; i < selected_faces.rows(); ++i) {
        assert(!hasNewConstraint.at(selected_faces[i]));
        hasNewConstraint.at(selected_faces[i]) = true;
    }

    // Count how many constraints we'll have in total
    size_t numConstraints = selected_faces.rows();
    for (int i = 0; i < constraint_fi.rows(); ++i)
        if (!hasNewConstraint.at(constraint_fi[i])) ++numConstraints;

    // Add the old, non-conflicting constraints to the selection
    size_t offset = selected_faces.rows();
    selected_faces.conservativeResize(numConstraints);
    selected_vec3 .conservativeResize(numConstraints, Eigen::NoChange);
    for (int i = 0; i < constraint_fi.rows(); ++i) {
        if (!hasNewConstraint.at(constraint_fi[i])) {
            selected_faces[offset]    = constraint_fi[i];
            selected_vec3.row(offset) = constraint_vec3.row(i);
            ++offset;
        }
    }

    constraint_fi = selected_faces;
    constraint_vec3 = selected_vec3;

    clearSelection();
}

void clearConstraints() {
    constraint_fi.resize(0, 1);
    constraint_vec3.resize(0,3);
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
        activelySelecting = fid>=0;
        return activelySelecting;
    }

    return false;
}

bool callback_mouse_move(Viewer& viewer, int mouse_x, int mouse_y) {
    if (selection_mode) {
        if (activelySelecting) {
            int fid = selector->strokeAdd(mouse_x, mouse_y);
            activelySelecting = fid>=0;
            return true;
        }
    }
    return false;
}

bool callback_mouse_up(Viewer& viewer, int button, int modifier)
{
    if (activelySelecting) {
        if (selection_mode) {
            selected_faces.resize(0,1);
            selected_vec3.resize(0,3);
            selection_stroke_points.resize(0, 3);
            selector->strokeFinish(selected_faces, selected_vec3, selection_stroke_points);
        }
        activelySelecting = false;
        callback_key_down(viewer, '1', 0);
        return true;
    }

    return false;
};

void line_texture() {
    unsigned size = 128;
    unsigned size2 = size / 2;
    unsigned lineWidth = 3;
    texture_I.setConstant(size, size, 255);
    for (unsigned i = 0; i < size; ++i)
        for (unsigned j = size2 - lineWidth; j <= size2 + lineWidth; ++j)
            texture_I(i, j) = 0;
    for (unsigned i = size2 - lineWidth; i <= size2 + lineWidth; ++i)
        for (unsigned j = 0; j < size; ++j)
            texture_I(i, j) = 0;
}

#define MAXBUFSIZE ((int) 1e6)
Eigen::MatrixXd readMatrix(const std::string &filename) {
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename);
    if (!infile.is_open())
        return Eigen::MatrixXd::Zero(0, 0);

    while (! infile.eof()) {
        string line;
        getline(infile, line);

        int temp_cols = 0;
        stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols * rows + temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
    }

    infile.close();

    rows--;

    // Populate matrix with numbers.
    Eigen::MatrixXd result(rows,cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result(i,j) = buff[cols * i + j];

    return result;
}
