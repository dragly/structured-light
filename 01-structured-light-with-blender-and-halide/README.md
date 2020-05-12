# Structured light, part one #

Create patterns with

```bash
mkdir output
python scripts/create_patterns.py
```

Open `scene/structured-light.blend` in Blender and render animation.

Create a build directory and run cmake and make:

```bash
mkdir build
cd build
cmake ..
make -j8
```

Open the resulting `out.xyz` file in Ovito or any other XYZ file viewer.
