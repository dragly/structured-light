#define GLM_ENABLE_EXPERIMENTAL 1
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Halide.h>
#include <fstream>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <halide_image_io.h>
#include <stdio.h>

using std::string;
using std::stringstream;
using std::vector;

using namespace Halide;
using namespace Halide::Tools;
using namespace Eigen;

using Vector4h = Matrix<Halide::Expr, 4, 1>;
using Matrix4h = Matrix<Halide::Expr, 4, 4>;
using Translation3h = Translation<Halide::Expr, 3>;

Matrix4h convert(const Matrix4f& m)
{
    Matrix4h result;
    result << m(0, 0), m(0, 1), m(0, 2), m(0, 3), m(1, 0), m(1, 1), m(1, 2), m(1, 3), m(2, 0), m(2, 1), m(2, 2), m(2, 3), m(3, 0), m(3, 1), m(3, 2), m(3, 3);
    return result;
}

Matrix4h convert(const glm::mat4& m)
{
    Matrix4h result;
    result << m[0][0], m[1][0], m[2][0], m[3][0], m[0][1], m[1][1], m[2][1], m[3][1], m[0][2], m[1][2], m[2][2], m[3][2], m[0][3], m[1][3], m[2][3], m[3][3];
    return result;
}

Var x, y, c, i, ii, xo, yo, xi, yi, img;

struct Point
{
    float x;
    float y;
    float z;
    float red;
    float green;
    float blue;
};

Buffer<uint8_t> loadImages(vector<string> filenames)
{
    if (filenames.empty()) {
        throw std::invalid_argument("List of filenames cannot be empty");
    }
    vector<Buffer<uint8_t>> images;
    Func assigner;
    assigner(x, y, c, i) = cast<uint8_t>(0);
    for (int i = 0; i < filenames.size(); i++) {
        const auto filename = filenames[i];
        std::cout << "Loading " << filename << std::endl;
        images.push_back(load_image(filename));
        assigner(x, y, c, i) = images[i](x, y, c);
    }
    Buffer<uint8_t> input(images[0].width(), images[0].height(), images[0].channels(), images.size());
    assigner.realize(input);
    return input;
}

Func binaryEncode(Buffer<uint8_t> input)
{
    Func maxValue;
    Func minValue;
    Func gray;
    Func pixelValue;
    const auto rdom = RDom(0, 7);
    minValue(x, y) = 1.0f;
    maxValue(x, y) = 0.0f;
    gray(x, y, i) = 0.333f * input(x, y, 0, i) / 255.0f + 0.333f * input(x, y, 1, i) / 255.0f + 0.333f * input(x, y, 2, i) / 255.0f;
    minValue(x, y) = min(minValue(x, y), gray(x, y, rdom));
    maxValue(x, y) = max(maxValue(x, y), gray(x, y, rdom));
    pixelValue(x, y, i) = cast<float>(gray(x, y, i) > (maxValue(x, y) + minValue(x, y)) / 2.0f);
    return pixelValue;
}

Func calculateProjectorX(Func pixelValue)
{
    Func pixelValueWeighted;
    pixelValueWeighted(x, y, i) = pixelValue(x, y, i) * pow(2.0f, 7 - i - 1);
    const auto rdom = RDom(0, 7);
    Func accumulated;
    accumulated(x, y) = 0.0f;
    accumulated(x, y) = accumulated(x, y) + pixelValueWeighted(x, y, rdom);
    Expr actualValueX = 2.0f * accumulated(x, y) / 127.0f - 1.0f;
    Func projectorNormalizedX;
    projectorNormalizedX(x, y) = select(
            accumulated(x, y) == 0.0f, -1000.f,
            select(
                accumulated(x, y) >= 1.0f, -1000.f,
                actualValueX
                )
            );
    return projectorNormalizedX;
}

void saveImages(Expr result, size_t width, size_t height, size_t imageCount, const string& basename)
{
    Target target = get_host_target();
    Func byteResult;
    byteResult(x, y, i) = cast<uint8_t>(clamp(result, 0.0f, 1.0f) * 255.0f);
    byteResult.compile_jit(target);
    Buffer<uint8_t> output(width, height, imageCount);
    byteResult.realize(output);
    for (int i = 0; i < imageCount; i++) {
        stringstream filename;
        filename << basename << i << ".png";
        const Buffer<uint8_t> image = output.sliced(2, i);
        save_image(image, filename.str());
    }
}

void saveImage(Expr result, size_t width, size_t height, const string& basename)
{
    Target target = get_host_target();
    Func byteResult;
    byteResult(x, y) = cast<uint8_t>(clamp(result, 0.0f, 1.0f) * 255.0f);
    byteResult.compile_jit(target);
    Buffer<uint8_t> output(width, height);
    byteResult.realize(output);
    stringstream filename;
    filename << basename << ".png";
    save_image(output, filename.str());
}

Func calculateColor(Buffer<uint8_t> input)
{
    Func minColor;
    Func maxColor;

    const auto rdom = RDom(0, 7);

    minColor(x, y, c) = 1.0f;
    minColor(x, y, c) = min(minColor(x, y, c), input(x, y, c, rdom) / 255.0f);

    maxColor(x, y, c) = 0.0f;
    maxColor(x, y, c) = max(maxColor(x, y, c), input(x, y, c, rdom) / 255.0f);

    Func color;
    color(x, y, c) = 0.5f * (minColor(x, y, c) + maxColor(x, y, c));
    return color;
}

Matrix4h createInverseTranslation()
{
    const auto offset = -0.2;
    const auto translation = Translation3f(Vector3f(offset, 0.0, 0.0));
    const auto inverseTranslationF = translation.inverse();
    const auto inverseTranslation = convert(Affine3f(inverseTranslationF).matrix());
    return inverseTranslation;
}

Matrix4h createInverseRotation()
{
    const auto angle = -10.0 / 180.0 * M_PI;

    const auto rotation = AngleAxisf(angle, Vector3f(0.0, 1.0, 0.0));
    const auto inverseRotationF = rotation.inverse();
    const auto inverseRotation = convert(Affine3f(inverseRotationF).matrix());
    return inverseRotation;
}

Matrix4h createInverseProjection()
{
    const float width = 600.0;
    const float height = 600;
    const float aspect = width / height;
    //const float fovx = 53.0;
    //const float fovy = fovx * (1.0 / aspect) * (M_PI / 180.0);
    const float fovy = 50.0 * M_PI / 180.0;
    const float near = 0.1;
    const float far = 100.0;
    const auto projection = glm::perspective(fovy, aspect, near, far);
    const auto inverseProjectionF = glm::inverse(projection);
    const Matrix4h inverseProjection = convert(inverseProjectionF);
    return inverseProjection;
}

template <typename F>
std::tuple<Vector4h, Vector4h> findCameraLine(F unprojectCamera, size_t width, size_t height)
{
    const Expr nx = 2.0f * cast<float>(x) / float(width) - 1.0f;
    const Expr ny = 2.0f * (1.0f - cast<float>(y) / float(height)) - 1.0f;
    const Vector4h normalCam1 { { nx }, { ny }, { 0.1f }, { 1.0f } };
    const Vector4h normalCam2 { { nx }, { ny }, { 0.7f }, { 1.0f } };
    const Vector4h pCam1 = unprojectCamera(normalCam1);
    const Vector4h pCam2 = unprojectCamera(normalCam2);

    const Vector4h lCam = pCam2 - pCam1;
    return { pCam1, lCam };
}

template <typename F>
Vector4h findProjectorLine(F unprojectProjector, Expr projectorX)
{
    const Vector4h normalPro {
        { projectorX }, { 0.0f }, { 0.7f }, { 1.0f }
    };
    const Vector4h lPro = unprojectProjector(normalPro);
    return lPro;
}

Vector4h intersect(Vector4h cameraPosition1, Vector4h cameraLine, Vector4h projectorLine)
{
    const auto p = Vector4h { { 0.0f }, { 0.0f }, { 0.0f }, { 1.0f } };
    const auto r = projectorLine - p;

    const auto q = cameraPosition1;
    const auto s = cameraLine;

    const auto cross2D = [](auto a, auto b) {
        return a(0, 0) * b(2, 0) - b(0, 0) * a(2, 0);
    };

    const auto pxr = cross2D(p, r);
    const auto qxr = cross2D(q, r);
    const auto qmpxr = qxr - pxr;
    const auto rxs = cross2D(r, s);
    const auto uCam = qmpxr / rxs;
    return q + s * uCam;
}

int main(int argc, char** argv)
{
    const vector<string> filenames {
        //"scene/scene_0.png",
        "scenes/scene_1.png",
        "scenes/scene_2.png",
        "scenes/scene_3.png",
        "scenes/scene_4.png",
        "scenes/scene_5.png",
        "scenes/scene_6.png",
        "scenes/scene_7.png"
    };

    std::cout << filenames[0] << std::endl;

    const auto input = loadImages(filenames);

    const auto encoded = binaryEncode(input);
    saveImages(encoded(x, y, i), input.width(), input.height(), input.extent(3), "pixel-value");

    const auto projectorX = calculateProjectorX(encoded);
    saveImage((projectorX(x, y) + 1.0f) / 2.0f, input.width(), input.height(), "projector-x");

    const auto inverseTranslation = createInverseTranslation();
    const auto inverseRotation = createInverseRotation();
    const auto inverseProjection = createInverseProjection();

    const auto unprojectCamera = [=](Vector4h nCam) {
        Vector4h pCam = inverseProjection * nCam;
        pCam = pCam / pCam(3, 0);

        Vector4h pPro = inverseTranslation * inverseRotation * pCam;
        return pPro;
    };

    const auto unprojectProjector = [=](Vector4h nPro) {
        Vector4h pPro = inverseProjection * nPro;
        pPro = pPro / pPro(3, 0);
        return pPro;
    };

    const auto [cameraPosition1, cameraLine] = findCameraLine(unprojectCamera, input.width(), input.height());
    const auto projectorLine = findProjectorLine(unprojectProjector, projectorX(x, y));

    const auto intersection = intersect(cameraPosition1, cameraLine, projectorLine);
    saveImage(-intersection(2, 0) / 3.0f, input.width(), input.height(), "intersection");

    const auto color = calculateColor(input);
    Func result;
    result(x, y, c) = 0.0f; // cast<uint8_t>(255.f);
    result(x, y, 0) = intersection(0, 0);
    result(x, y, 1) = intersection(1, 0);
    result(x, y, 2) = intersection(2, 0);
    result(x, y, 3) = color(x, y, 0);
    result(x, y, 4) = color(x, y, 1);
    result(x, y, 5) = color(x, y, 2);

    Target target = get_host_target();
    result.compile_jit(target);
    Buffer<float> output(input.width(), input.height(), 6);
    result.realize(output);
    output.copy_to_host();

    std::vector<Point> points;

    for (int j = 0; j < output.height(); j++) {
        for (int i = 0; i < output.width(); i++) {
            const auto x = output(i, j, 0);
            const auto y = output(i, j, 1);
            const auto z = output(i, j, 2);
            const auto red = output(i, j, 3);
            const auto green = output(i, j, 4);
            const auto blue = output(i, j, 5);

            if (x < -10.0f || x > 10.0f || y < -10.0f || y > 10.0f || z < -10.0f || z > 10.0f) {
                continue;
            }
            points.push_back({ x, y, z, red, green, blue });
        }
    }

    std::ofstream outFile;
    outFile.open("out.xyz");
    outFile << points.size() << "\n";
    outFile << "comment"
            << "\n";
    for (const auto& point : points) {
        outFile << point.x << " " << point.y << " " << point.z << " " << point.red
                << " " << point.green << " " << point.blue << "\n";
    }

    return 0;
}
