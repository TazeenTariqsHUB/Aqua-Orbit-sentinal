# Blender Python script: Realistic underwater scene with distant shark and reef grass
# Run inside Blender's Scripting workspace (Blender 3.x recommended).
import bpy
import math
from mathutils import Vector, Euler

# -------------------------
# User settings
# -------------------------
SHARK_PATH = ""  # e.g. "C:/models/shark.glb" or leave empty to use procedurally generated shark
USE_IMPORT = bool(SHARK_PATH)
SCENE_NAME = "UnderwaterScene"
RENDER_ENGINE = 'CYCLES'  # 'CYCLES' or 'EEVEE'

# -------------------------
# Helpers
# -------------------------
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

def set_renderer(engine='CYCLES'):
    bpy.context.scene.render.engine = engine
    if engine == 'CYCLES':
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.cycles.use_adaptive_sampling = True
        bpy.context.scene.cycles.max_bounces = 8
    else:
        bpy.context.scene.eevee.taa_render_samples = 64

# -------------------------
# Create seabed / sand plane
# -------------------------
def create_seabed():
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0,0,0))
    sand = bpy.context.object
    sand.name = "Seabed"
    # Subdivide and displace for ripples
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=60)
    bpy.ops.object.mode_set(mode='OBJECT')
    disp_mod = sand.modifiers.new("Disp", type='DISPLACE')
    tex = bpy.data.textures.new("SandNoise", type='DISTORTED_NOISE')
    tex.noise_scale = 1.0
    disp_mod.texture = tex
    disp_mod.strength = 0.25
    sand.location.z = -2.0
    # Add sand material (procedural)
    mat = bpy.data.materials.new("SandMaterial")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    nodes = nt.nodes
    links = nt.links
    out = nodes.new("ShaderNodeOutputMaterial")
    princ = nodes.new("ShaderNodeBsdfPrincipled")
    tex_coord = nodes.new("ShaderNodeTexCoord")
    noise = nodes.new("ShaderNodeTexNoise")
    color_ramp = nodes.new("ShaderNodeValToRGB")
    noise.inputs["Scale"].default_value = 4.0
    color_ramp.color_ramp.elements[0].position = 0.25
    color_ramp.color_ramp.elements[0].color = (0.85, 0.8, 0.65, 1)
    color_ramp.color_ramp.elements[1].color = (0.95,0.92,0.85,1)
    links.new(tex_coord.outputs['Object'], noise.inputs['Vector'])
    links.new(noise.outputs['Fac'], color_ramp.inputs['Fac'])
    links.new(color_ramp.outputs['Color'], princ.inputs['Base Color'])
    princ.inputs['Roughness'].default_value = 0.9
    links.new(princ.outputs['BSDF'], out.inputs['Surface'])
    sand.data.materials.append(mat)
    return sand

# -------------------------
# Create grass blade (object used by particle system)
# -------------------------
def create_grass_blade():
    bpy.ops.mesh.primitive_plane_add(size=0.2, location=(0,0,0))
    blade = bpy.context.object
    blade.name = "GrassBlade"
    # thin and bend
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.transform.resize(value=(0.15,0.02,1))
    bpy.ops.transform.rotate(value=math.radians(20), orient_axis='Y')
    bpy.ops.object.mode_set(mode='OBJECT')
    # convert to single face tall blade - give some thickness using solidify
    mod = blade.modifiers.new("Solid", type='SOLIDIFY')
    mod.thickness = 0.01
    # give simple green material
    mat = bpy.data.materials.new("GrassMaterial")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    princ = nodes.new("ShaderNodeBsdfPrincipled")
    color = nodes.new("ShaderNodeRGB")
    color.outputs[0].default_value = (0.08, 0.45, 0.12, 1)
    princ.inputs['Roughness'].default_value = 0.8
    links.new(color.outputs[0], princ.inputs['Base Color'])
    links.new(princ.outputs['BSDF'], out.inputs['Surface'])
    blade.data.materials.append(mat)
    # rotate blade so Y axis points up for particle orientation
    blade.rotation_euler = Euler((math.radians(-90),0,0), 'XYZ')
    blade.scale = (1,1,1)
    return blade

# -------------------------
# Add grass particle system to seabed
# -------------------------
def add_grass_particles(ground, blade_obj):
    ps = ground.modifiers.new("Grass", type='PARTICLE_SYSTEM')
    psettings = ps.particle_system.settings
    psettings.count = 22000
    psettings.hair_length = 0.35
    psettings.use_advanced_hair = True
    psettings.render_type = 'OBJECT'
    psettings.instance_object = blade_obj
    psettings.child_type = 'SIMPLE'
    psettings.child_nbr = 3
    psettings.physics_type = 'NO'
    psettings.use_rotations = True
    psettings.rotation_mode = 'NOR'
    psettings.distribution = 'RAND'
    psettings.kink = 'BEND'
    psettings.kink_amplitude = 0.05
    psettings.length_random = 0.35

# -------------------------
# Volumetric water cube
# -------------------------
def create_water_volume():
    bpy.ops.mesh.primitive_cube_add(size=60, location=(0,0,8))
    water = bpy.context.object
    water.name = "WaterVolume"
    # make it non-renderable in other ways (we want just volume)
    water.display_type = 'WIRE'
    # volume material: volume scatter + absorption
    mat = bpy.data.materials.new("UnderwaterVolume")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    vol_scatter = nodes.new("ShaderNodeVolumeScatter")
    vol_abs = nodes.new("ShaderNodeVolumeAbsorption")
    add = nodes.new("ShaderNodeAddShader")
    vol_scatter.inputs['Density'].default_value = 0.02
    vol_abs.inputs['Density'].default_value = 0.03
    # slight blue tint
    vol_scatter.inputs['Color'].default_value = (0.02, 0.08, 0.18, 1)
    vol_abs.inputs['Color'].default_value = (0.01, 0.04, 0.08, 1)
    links.new(vol_scatter.outputs['Volume'], add.inputs[0])
    links.new(vol_abs.outputs['Volume'], add.inputs[1])
    links.new(add.outputs['Shader'], out.inputs['Volume'])
    water.data.materials.append(mat)
    # make the cube not cast shadows on top by setting visibility
    water.cycles.is_shadow_catcher = False
    return water

# -------------------------
# Add light sources (sun + soft fill from above)
# -------------------------
def create_lighting():
    # Sun (directional) simulating sunlight from above-water
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 20))
    sun = bpy.context.object
    sun.name = "Sun"
    sun.rotation_euler = Euler((math.radians(-75), 0, 0), 'XYZ')
    sun.data.energy = 3.0
    # Create an area light above to imitate caustic soft fill
    bpy.ops.object.light_add(type='AREA', location=(0, -5, 18))
    area = bpy.context.object
    area.data.size = 15
    area.data.energy = 300.0
    area.name = "TopFill"

# -------------------------
# Create a procedural shark placeholder (if no import)
# A simplified elongated body + fins
# -------------------------
def create_procedural_shark(location=(0,8,-0.5), scale=1.0, rotation=(0,0,0)):
    # body: stretched UV sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=location)
    body = bpy.context.object
    body.name = "SharkBody"
    body.scale = (1.6*scale, 4.0*scale, 0.9*scale)  # elongated
    # add subdivision for smoothness
    sub = body.modifiers.new("Subsurf", type='SUBSURF')
    sub.levels = 3
    sub.render_levels = 3
    # simple deform to taper nose
    def_mod = body.modifiers.new("Taper", type='SIMPLE_DEFORM')
    def_mod.deform_method = 'TAPER'
    def_mod.factor = -0.5
    def_mod.origin = None
    # dorsal fin (cone)
    bpy.ops.mesh.primitive_cone_add(radius1=0.3*scale, depth=0.8*scale, location=(location[0], location[1]-0.3*scale, location[2]+0.8*scale))
    dorsal = bpy.context.object
    dorsal.name = "DorsalFin"
    dorsal.rotation_euler = Euler((math.radians(0), math.radians(0), math.radians(90)), 'XYZ')
    # tail: two fin cones mirrored
    bpy.ops.mesh.primitive_cone_add(radius1=0.25*scale, depth=0.5*scale, location=(location[0], location[1]+3.6*scale, location[2]+0.05*scale))
    tail1 = bpy.context.object
    tail1.rotation_euler = Euler((math.radians(0), math.radians(0), math.radians(-30)), 'XYZ')
    bpy.ops.mesh.primitive_cone_add(radius1=0.25*scale, depth=0.5*scale, location=(location[0], location[1]+3.6*scale, location[2]-0.15*scale))
    tail2 = bpy.context.object
    tail2.rotation_euler = Euler((math.radians(0), math.radians(0), math.radians(30)), 'XYZ')
    # pectoral fins
    bpy.ops.mesh.primitive_plane_add(size=0.6*scale, location=(location[0]-0.9*scale, location[1]+0.3*scale, location[2]-0.3*scale))
    p1 = bpy.context.object
    p1.name = "Pectoral1"
    p1_scale = (0.6*scale,0.2*scale,0.2*scale)
    p1.scale = p1_scale
    p1.rotation_euler = Euler((math.radians(30), math.radians(0), math.radians(20)), 'XYZ')
    bpy.ops.mesh.primitive_plane_add(size=0.6*scale, location=(location[0]+0.9*scale, location[1]+0.3*scale, location[2]-0.3*scale))
    p2 = bpy.context.object
    p2.name = "Pectoral2"
    p2.scale = p1_scale
    p2.rotation_euler = Euler((math.radians(30), math.radians(0), math.radians(-20)), 'XYZ')

    # Parent all fins to body
    for o in [dorsal, tail1, tail2, p1, p2]:
        o.parent = body

    # Create shark material
    mat = bpy.data.materials.new("SharkMaterial")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    princ = nodes.new('ShaderNodeBsdfPrincipled')
    ramp = nodes.new('ShaderNodeValToRGB')
    geom = nodes.new('ShaderNodeNewGeometry')
    mix = nodes.new('ShaderNodeMixShader')
    glossy = nodes.new('ShaderNodeBsdfGlossy')
    princ.inputs['Specular'].default_value = 0.2
    princ.inputs['Roughness'].default_value = 0.4
    # color: darker gray on top, lighter belly
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.08,0.10,0.13,1)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.78,0.80,0.85,1)
    # use position/normal to blend - use geometry 'Normal' dot to simulate top vs belly
    sep = nodes.new('ShaderNodeSeparateXYZ')
    # simpler: use Geometry->Normal Z to drive factor
    # create node to access Normal Z via attribute
    # Node setup
    links.new(geom.outputs['Normal'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], princ.inputs['Base Color'])
    links.new(princ.outputs['BSDF'], out.inputs['Surface'])
    body.data.materials.append(mat)
    # rotate and position
    body.rotation_euler = Euler((0,0,0), 'XYZ')
    body.location = Vector(location)
    # slight smooth shading
    bpy.ops.object.shade_smooth()
    return body

# -------------------------
# Import shark if specified
# -------------------------
def import_shark(path, location=(0,8,-0.5), scale=1.0):
    lower = path.lower()
    imported = None
    try:
        if lower.endswith('.obj'):
            bpy.ops.import_scene.obj(filepath=path)
            imported = bpy.context.selected_objects[0]
        elif lower.endswith('.fbx'):
            bpy.ops.import_scene.fbx(filepath=path)
            imported = bpy.context.selected_objects[0]
        elif lower.endswith('.glb') or lower.endswith('.gltf'):
            bpy.ops.import_scene.gltf(filepath=path)
            imported = bpy.context.selected_objects[0]
        else:
            print("Unsupported format for import, using procedural shark.")
            return None
    except Exception as e:
        print("Import failed:", e)
        return None

    if imported:
        # move to location and scale
        imported.location = Vector(location)
        imported.scale = (scale, scale, scale)
        imported.rotation_euler = Euler((0,0,0),'XYZ')
        # smooth shading
        for ob in bpy.context.selected_objects:
            try:
                bpy.context.view_layer.objects.active = ob
                bpy.ops.object.shade_smooth()
            except:
                pass
    return imported

# -------------------------
# Add sensor/tags object above shark
# -------------------------
def add_sensor(location=(0,7.5,1.8)):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.06, depth=0.35, location=location)
    cyl = bpy.context.object
    cyl.name = "SensorTag"
    cyl.rotation_euler = Euler((math.radians(90),0,0),'XYZ')
    mat = bpy.data.materials.new("SensorMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    princ = nodes.new('ShaderNodeBsdfPrincipled')
    princ.inputs['Base Color'].default_value = (0.95,0.9,0.2,1)
    princ.inputs['Metallic'].default_value = 0.1
    princ.inputs['Roughness'].default_value = 0.3
    links = nt.links
    links.new(princ.outputs['BSDF'], out.inputs['Surface'])
    cyl.data.materials.append(mat)
    # small emission stripe for visibility
    bpy.ops.mesh.primitive_plane_add(size=0.12, location=(location[0], location[1]+0.02, location[2]))
    stripe = bpy.context.object
    stripe.rotation_euler = Euler((math.radians(90),0,0),'XYZ')
    em = bpy.data.materials.new("Stripe")
    em.use_nodes = True
    n = em.node_tree
    n.nodes.clear()
    out = n.nodes.new('ShaderNodeOutputMaterial')
    emis = n.nodes.new('ShaderNodeEmission')
    emis.inputs['Strength'].default_value = 4.0
    emis.inputs['Color'].default_value = (0.0, 0.8, 1.0, 1)
    n.links.new(emis.outputs['Emission'], out.inputs['Surface'])
    stripe.data.materials.append(em)
    stripe.parent = cyl
    return cyl

# -------------------------
# Camera setup
# -------------------------
def create_camera(location=(0,-8,0.5), look_at=(0,8,0), focal_length=85, dof_distance=8.0):
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.object
    cam.name = "MainCamera"
    cam.data.lens = focal_length
    # point camera to look_at using track-to
    direction = Vector(look_at) - Vector(location)
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    # depth of field
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = dof_distance
    cam.data.dof.aperture_fstop = 2.8
    bpy.context.scene.camera = cam
    return cam

# -------------------------
# Main assembly
# -------------------------
def build_scene():
    clear_scene()
    set_renderer(RENDER_ENGINE)
    seabed = create_seabed()
    blade = create_grass_blade()
    add_grass_particles(seabed, blade)
    water = create_water_volume()
    create_lighting()
    # import or procedural shark
    shark_obj = None
    if USE_IMPORT:
        shark_obj = import_shark(SHARK_PATH, location=(0,10,-0.3), scale=1.0)
    if not shark_obj:
        shark_obj = create_procedural_shark(location=(0,10,-0.4), scale=0.9)
    # position shark deeper and farther to appear in the distance
    shark_obj.location = Vector((0, 10, -0.4))
    shark_obj.scale = (0.9,0.9,0.9)
    # add small sensor above shark
    add_sensor(location=(0,9.6,1.2))
    # add camera
    create_camera(location=(0,-6,0.3), look_at=(0,9.5,0.0), focal_length=90, dof_distance=10.0)
    # small particulate: create many small sprites or volumetric noise using particle system
    # optional: add a few point lights close to shark for subtle shading
    bpy.ops.object.light_add(type='POINT', location=(0,8,1.5))
    point = bpy.context.object
    point.data.energy = 60.0
    point.data.shadow_soft_size = 0.6
    # render settings: film, denoising
    bpy.context.scene.render.film_transparent = False
    bpy.context.scene.view_layers["View Layer"].use_pass_diffuse_color = True
    if RENDER_ENGINE == 'CYCLES':
        bpy.context.scene.cycles.use_denoising = True

# -------------------------
# Run build
# -------------------------
build_scene()
print("Underwater scene created. If you have a high-quality shark file, set SHARK_PATH and re-run to import it.")
