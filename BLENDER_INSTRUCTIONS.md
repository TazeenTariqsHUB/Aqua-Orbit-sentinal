# Blender Underwater Scene Script Instructions

## Overview
This Blender Python script creates a realistic underwater scene with a distant shark, reef grass, and professional lighting. It's designed to work with Blender 3.x and produces high-quality renders suitable for scientific visualization or artistic projects.

## Features
- **Procedural Shark**: Creates a detailed shark model with fins, body, and realistic materials
- **Reef Grass**: 22,000+ individual grass blades using particle systems
- **Volumetric Water**: Realistic underwater atmosphere with light scattering
- **Professional Lighting**: Sun and area lights for natural underwater illumination
- **Sensor Tags**: Optional tracking device on the shark
- **Camera Setup**: Pre-configured camera with depth of field
- **Material System**: Procedural materials for sand, grass, and shark

## Requirements
- **Blender 3.x** (3.0 or higher recommended)
- **Cycles Render Engine** (recommended) or Eevee
- **Python 3.7+** (included with Blender)

## Installation & Usage

### Method 1: Direct Script Execution
1. Download `blender_underwater_scene.py`
2. Open Blender
3. Switch to the **Scripting** workspace (top tabs)
4. Click **Open** and select the downloaded script
5. Click **Run Script** (▶️ button)
6. The scene will be automatically generated

### Method 2: Text Editor
1. In Blender, go to **Scripting** workspace
2. Click **New** in the text editor
3. Copy and paste the entire script content
4. Click **Run Script**

## Customization Options

### Shark Import
To use your own shark model instead of the procedural one:
```python
SHARK_PATH = "C:/path/to/your/shark.glb"  # or .obj, .fbx
USE_IMPORT = True
```

Supported formats:
- `.glb` / `.gltf` (recommended)
- `.obj`
- `.fbx`

### Render Settings
```python
RENDER_ENGINE = 'CYCLES'  # or 'EEVEE'
```

### Scene Parameters
- **Shark Position**: `location=(0,10,-0.4)`
- **Shark Scale**: `scale=0.9`
- **Camera Position**: `location=(0,-6,0.3)`
- **Focal Length**: `focal_length=90`

## Scene Components

### 1. Seabed
- Procedural sand material with noise displacement
- 40x40 unit plane with 60 subdivisions
- Realistic sand colors and roughness

### 2. Reef Grass
- 22,000 individual grass blades
- Procedural green material
- Particle system with child particles for density
- Natural bending and length variation

### 3. Water Volume
- 60x60x60 unit cube
- Volume scatter and absorption shaders
- Blue-tinted underwater atmosphere
- Realistic light scattering

### 4. Shark Model
- Procedural UV sphere with modifiers
- Subdivision surface for smoothness
- Taper deformation for realistic shape
- Dorsal fin, tail fins, and pectoral fins
- Counter-shading material (dark top, light belly)

### 5. Lighting Setup
- **Sun Light**: Directional light simulating sunlight
- **Area Light**: Soft fill light for caustics
- **Point Light**: Subtle shading near shark

### 6. Camera
- Pre-positioned for optimal composition
- Depth of field enabled (f/2.8)
- 90mm focal length for natural perspective

## Rendering Tips

### For Best Quality (Cycles):
- Increase samples: `bpy.context.scene.cycles.samples = 256`
- Enable denoising: `bpy.context.scene.cycles.use_denoising = True`
- Adjust max bounces: `bpy.context.scene.cycles.max_bounces = 12`

### For Faster Renders (Eevee):
- Use Eevee for real-time preview
- Enable bloom and screen space reflections
- Adjust TAA samples for quality vs speed

## Troubleshooting

### Common Issues:
1. **Script won't run**: Ensure you're in Blender 3.x
2. **Import fails**: Check file path and format support
3. **Slow rendering**: Reduce particle count or use Eevee
4. **Memory issues**: Lower subdivision levels

### Performance Optimization:
- Reduce grass particle count: `psettings.count = 10000`
- Lower subdivision levels: `sub.levels = 2`
- Use Eevee for faster preview renders

## Advanced Customization

### Adding More Marine Life:
```python
# Add fish schools
def create_fish_school():
    # Your fish creation code here
    pass
```

### Modifying Materials:
```python
# Custom shark material
mat = bpy.data.materials.new("CustomShark")
# Modify shader nodes as needed
```

### Animation:
```python
# Add swimming animation
def animate_shark():
    # Keyframe shark movement
    pass
```

## Output Formats
The script sets up the scene for rendering. To export:
- **Images**: Render → Render Image (F12)
- **Video**: Render → Render Animation
- **3D Models**: File → Export → Choose format

## Support
For issues or questions:
1. Check Blender console for error messages
2. Verify all requirements are met
3. Try reducing complexity (particle count, subdivisions)
4. Test with default settings first

## License
This script is provided as-is for educational and research purposes. Feel free to modify and adapt for your projects.

---

**Happy Rendering! 🦈🌊**
