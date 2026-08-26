"""
Master Wok Simulation: Slower Motion + Zero Bounciness + High-Frequency Physics + Fixed Joint Unparenting + Handle Attachment
"""

import asyncio
import numpy as np
import torch
import omni.usd
import omni.kit.commands
import traceback
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ROBOT_PRIM_PATH = "/World/crx10ial"  # Your FANUC robot prim path
EE_LINK_NAME = "link_6"                # End-effector link name
NUM_POPCORN = 30                       # Number of popcorn kernels
WOK_RADIUS = 0.18                      # Wok radius in meters
WOK_DEPTH = 0.13                       # Wok depth
WOK_OFFSET_X = 0.27                    # 27cm offset away from end-effector (18cm + 9cm)
WOK_OFFSET_Z = -0.10                   # 10cm offset down relative to end-effector
# ==============================================================================


# 0. Ensure Robot Arm Exists (Safe Check to Prevent Freezing)
def ensure_robot_exists(stage, prim_path=ROBOT_PRIM_PATH):
    if stage.GetPrimAtPath(prim_path).IsValid():
        return True

    print(f"[Robot Setup] Robot prim '{prim_path}' not found on stage!")
    print(f"[Robot Setup] Attempting non-blocking spawn from Isaac Sim assets...")
    try:
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.stage import add_reference_to_stage
        
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("[Warning] Isaac Sim Nucleus server is unreachable. Please manually load/import the FANUC robot into the stage at '/World/crx10ial'.")
            return False
            
        usd_path = assets_root_path + "/Isaac/Robots/Fanuc/CRX10IAL/crx10ial.usd"
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        print(f"[Robot Setup] Successfully added FANUC CRX-10iA/L reference to {prim_path}!")
        return True
    except Exception as e:
        print(f"[Warning] Auto-spawn skipped or failed ({e}). Please ensure '/World/crx10ial' exists on your stage.")
        return False


# 1a. Procedural 3D Solid Wok Dish Mesh (With Contact Padding & Thickness)
def create_concave_wok_mesh(stage, wok_path: str, radius: float = WOK_RADIUS, depth: float = WOK_DEPTH, thickness: float = 0.005):
    num_rings = 10
    num_segments = 24
    vertices = []
    face_vertex_counts = []
    face_vertex_indices = []

    # A. Generate Vertices (Inner & Outer Shells)
    # Inner Shell (0 to N-1)
    for i in range(num_rings + 1):
        v_angle = (i / num_rings) * (np.pi / 2.0)
        r = radius * np.sin(v_angle)
        z = depth * (1.0 - np.cos(v_angle))
        for j in range(num_segments):
            h_angle = (j / num_segments) * 2.0 * np.pi
            vertices.append(Gf.Vec3f(float(r * np.cos(h_angle)), float(r * np.sin(h_angle)), float(z)))

    # Outer Shell (N to 2N-1)
    for i in range(num_rings + 1):
        v_angle = (i / num_rings) * (np.pi / 2.0)
        r = (radius + thickness) * np.sin(v_angle)
        z = depth * (1.0 - np.cos(v_angle)) - thickness * np.cos(v_angle)
        for j in range(num_segments):
            h_angle = (j / num_segments) * 2.0 * np.pi
            vertices.append(Gf.Vec3f(float(r * np.cos(h_angle)), float(r * np.sin(h_angle)), float(z)))

    N = (num_rings + 1) * num_segments

    # B. Generate Faces
    # Inner Shell Faces
    for i in range(num_rings):
        for j in range(num_segments):
            next_j = (j + 1) % num_segments
            idx0 = i * num_segments + j
            idx1 = i * num_segments + next_j
            idx2 = (i + 1) * num_segments + next_j
            idx3 = (i + 1) * num_segments + j
            face_vertex_counts.append(4)
            face_vertex_indices.extend([idx0, idx1, idx2, idx3])

    # Outer Shell Faces (Reverse winding for outside-facing normals)
    for i in range(num_rings):
        for j in range(num_segments):
            next_j = (j + 1) % num_segments
            idx0 = N + i * num_segments + j
            idx1 = N + i * num_segments + next_j
            idx2 = N + (i + 1) * num_segments + next_j
            idx3 = N + (i + 1) * num_segments + j
            face_vertex_counts.append(4)
            face_vertex_indices.extend([idx0, idx3, idx2, idx1])

    # Rim Faces (Connect top edges)
    for j in range(num_segments):
        next_j = (j + 1) % num_segments
        idx0 = num_rings * num_segments + j
        idx1 = num_rings * num_segments + next_j
        idx2 = N + num_rings * num_segments + next_j
        idx3 = N + num_rings * num_segments + j
        face_vertex_counts.append(4)
        face_vertex_indices.extend([idx0, idx1, idx2, idx3])

    mesh = UsdGeom.Mesh.Define(stage, wok_path)
    mesh.GetPointsAttr().Set(vertices)
    mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
    mesh.GetDoubleSidedAttr().Set(False)
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.bilinear)

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)

    # PhysX Collision Offsets to prevent kernel penetration/tunneling through decomposition seams
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    physx_collision.CreateContactOffsetAttr().Set(0.02)  # 2cm contact sensing distance
    physx_collision.CreateRestOffsetAttr().Set(0.001)     # 1mm safety padding offset

    return mesh


# 1b. Create Wok Handle Cylinder (Connects End-Effector to Wok Outer Rim Only)
def create_wok_handle(stage, handle_path: str, offset_x: float = WOK_OFFSET_X, wok_radius: float = WOK_RADIUS, handle_radius: float = 0.015, z_offset: float = -WOK_OFFSET_Z):
    # Length spans from EE flange (-offset_x) to Wok Rim (-wok_radius)
    length = max(0.02, offset_x - wok_radius)
    
    cylinder = UsdGeom.Cylinder.Define(stage, handle_path)
    cylinder.CreateRadiusAttr().Set(handle_radius)
    cylinder.CreateHeightAttr().Set(length)
    cylinder.CreateAxisAttr().Set("X")  # Align cylinder along X-axis
    
    # Midpoint between EE Link (-offset_x) and Wok Rim (-wok_radius)
    midpoint_x = -(offset_x + wok_radius) / 2.0
    xform = UsdGeom.XformCommonAPI(cylinder.GetPrim())
    xform.SetTranslate(Gf.Vec3d(midpoint_x, 0.0, z_offset))
    
    cylinder.CreateDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.1, 0.1)])
    return cylinder


# 2. Attach Wok via FixedJoint & Spawn Popcorn Particles
def setup_wok_and_popcorn(robot_prim_path=ROBOT_PRIM_PATH, ee_link_name=EE_LINK_NAME, num_popcorn=NUM_POPCORN):
    stage = omni.usd.get_context().get_stage()

    # Ensure Robot Arm Exists
    if not ensure_robot_exists(stage, robot_prim_path):
        return False

    ee_path = f"{robot_prim_path}/{ee_link_name}"
    ee_prim = stage.GetPrimAtPath(ee_path)

    if not ee_prim.IsValid():
        print(f"[Error] End-effector link '{ee_path}' not found!")
        return False

    # A. Clean up all previous elements
    paths_to_delete = [
        "/World/PopcornParticles",            
        "/World/PopcornInstancer",
        "/World/Wok_Bowl",
        f"{ee_path}/WokFixedJoint",
        f"{ee_path}/Wok_Bowl/PopcornGroup",   
        f"{ee_path}/Wok_Bowl"                 
    ]
    for path in paths_to_delete:
        if stage.GetPrimAtPath(path).IsValid():
            omni.kit.commands.execute('DeletePrims', paths=[path])

    # B. Create Wok Bowl as a STANDALONE prim at /World/Wok_Bowl (Isolates material inheritance)
    wok_path = "/World/Wok_Bowl"
    create_concave_wok_mesh(stage, wok_path, radius=WOK_RADIUS, depth=WOK_DEPTH)
    wok_prim = stage.GetPrimAtPath(wok_path)

    # Create Wok Handle attaching the end-effector to the wok bowl
    handle_path = f"{wok_path}/Handle"
    create_wok_handle(stage, handle_path, offset_x=WOK_OFFSET_X, wok_radius=WOK_RADIUS, handle_radius=0.015)

    # Enable Rigid Body & Symmetric CCD on Wok
    UsdPhysics.RigidBodyAPI.Apply(wok_prim)
    physx_wok = PhysxSchema.PhysxRigidBodyAPI.Apply(wok_prim)
    physx_wok.CreateEnableCCDAttr().Set(True)

    # Enable CCD on Robot End-Effector Link as well
    physx_ee = PhysxSchema.PhysxRigidBodyAPI.Apply(ee_prim)
    physx_ee.CreateEnableCCDAttr().Set(True)

    # Position Wok relative to end-effector initial world transform
    ee_xform_matrix = omni.usd.get_world_transform_matrix(ee_prim)
    wok_offset_matrix = Gf.Matrix4d().SetTranslate(Gf.Vec3d(WOK_OFFSET_X, 0.0, WOK_OFFSET_Z))
    final_wok_matrix = wok_offset_matrix * ee_xform_matrix

    wok_xformable = UsdGeom.Xformable(wok_prim)
    wok_xformable.ClearXformOpOrder()
    wok_xformable.AddTransformOp().Set(final_wok_matrix)

    # Attach Wok to End-Effector using Physics FixedJoint with explicit local frames
    joint_path = f"{ee_path}/WokFixedJoint"
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    fixed_joint.CreateBody0Rel().SetTargets([ee_path])
    fixed_joint.CreateBody1Rel().SetTargets([wok_path])
    
    # Author joint frame offsets so PhysX maintains WOK_OFFSET_X and WOK_OFFSET_Z!
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(WOK_OFFSET_X), 0.0, float(WOK_OFFSET_Z)))
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # Apply Physics Material (Zero Bounciness, High Friction)
    material_path = "/World/PopcornMaterial"
    if not stage.GetPrimAtPath(material_path).IsValid():
        UsdShade.Material.Define(stage, material_path)
        physics_material = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(material_path))
        physics_material.CreateRestitutionAttr().Set(0.0)   # 0% bounciness
        physics_material.CreateStaticFrictionAttr().Set(0.8) # High static friction
        physics_material.CreateDynamicFrictionAttr().Set(0.6) # High dynamic friction
    popcorn_mat = UsdShade.Material(stage.GetPrimAtPath(material_path))

    # Bind physics material to wok specifically under "physics" purpose
    UsdShade.MaterialBindingAPI.Apply(wok_prim).Bind(popcorn_mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    # C. Spawn Popcorn in WORLD SPACE
    popcorn_root = "/World/PopcornParticles"
    stage.DefinePrim(popcorn_root, "Xform")
    
    kernel_radius = 0.008
    spacing = 0.04 # 4cm spacing
    grid_xy = int(np.ceil(np.sqrt(num_popcorn / 4.0)))
    if grid_xy < 5: grid_xy = 5

    for i in range(num_popcorn):
        p_path = f"{popcorn_root}/kernel_{i}"
        sphere = UsdGeom.Sphere.Define(stage, p_path)
        sphere.CreateRadiusAttr().Set(kernel_radius)
        
        # Color popcorn yellow
        sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.9, 0.1)])

        layer = i // (grid_xy * grid_xy)
        remainder = i % (grid_xy * grid_xy)
        ix = remainder % grid_xy
        iy = remainder // grid_xy
        
        offset_x = (ix - grid_xy/2.0) * spacing
        offset_y = (iy - grid_xy/2.0) * spacing
        offset_z = 0.05 + layer * spacing

        local_pos = Gf.Vec3d(WOK_OFFSET_X + offset_x, offset_y, WOK_OFFSET_Z + 0.05 + layer * spacing)
        world_pos = ee_xform_matrix.Transform(local_pos)

        sphere_xform = UsdGeom.XformCommonAPI(sphere.GetPrim())
        sphere_xform.SetTranslate(world_pos)

        # Rigid Body Physics & Mass (~0.5g)
        UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        UsdPhysics.MassAPI.Apply(sphere.GetPrim()).CreateMassAttr().Set(0.0005)
        
        # Enable CCD on every popcorn kernel
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(sphere.GetPrim())
        physx_rb.CreateEnableCCDAttr().Set(True)

        # Bind Physics Material
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(popcorn_mat)

    print(f"Spawned {num_popcorn} yellow popcorn kernels with symmetric CCD and contact padding!")
    return True


# 3. Minimum-Jerk S-Curve Position Generator
def compute_scurve_wok_position(
    sim_time: float, 
    freq: float = 1.15,                 
    amp_x: float = 0.116,               
    amp_z: float = 0.025,               
    amp_pitch: float = np.radians(16),  
    phase_lag_ratio: float = 0.15
) -> tuple:
    ramp_duration = 2.0
    if sim_time < ramp_duration:
        u = sim_time / ramp_duration
        envelope = 6 * u**5 - 15 * u**4 + 10 * u**3  # Smootherstep
    else:
        envelope = 1.0

    def scurve_periodic(t):
        phase = (t * freq) % 1.0
        if phase < 0.5:
            x = phase * 2.0
            return 6 * x**5 - 15 * x**4 + 10 * x**3
        else:
            x = (phase - 0.5) * 2.0
            return 1.0 - (6 * x**5 - 15 * x**4 + 10 * x**3)

    pos_x = envelope * amp_x * 2.0 * (scurve_periodic(sim_time) - 0.5)
    pos_z = envelope * amp_z * 2.0 * (scurve_periodic(sim_time) - 0.5)
    pos_pitch = envelope * amp_pitch * 2.0 * (scurve_periodic(sim_time - phase_lag_ratio / freq) - 0.5)

    return pos_x, pos_z, pos_pitch


def apply_aesthetics(stage, robot_prim_path):
    from pxr import Usd, UsdShade, Gf, Sdf
    
    def create_color_material(path, color_rgb, roughness=0.4):
        looks_scope = "/World/Looks"
        if not stage.GetPrimAtPath(looks_scope).IsValid():
            UsdGeom.Scope.Define(stage, looks_scope)
            
        if stage.GetPrimAtPath(path).IsValid():
            material = UsdShade.Material(stage.GetPrimAtPath(path))
            shader = UsdShade.Shader(stage.GetPrimAtPath(path + "/Shader"))
        else:
            material = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color_rgb)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
        
        return material

    orange_mat = create_color_material("/World/Looks/Orange", Gf.Vec3f(1.0, 0.35, 0.0), 0.3)
    grey_mat = create_color_material("/World/Looks/LightGrey", Gf.Vec3f(0.6, 0.6, 0.6), 0.8)
    black_mat = create_color_material("/World/Looks/WokBlack", Gf.Vec3f(0.05, 0.05, 0.05), 0.2)

    # Color Manipulator Orange
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if robot_prim.IsValid():
        mat_api = UsdShade.MaterialBindingAPI.Apply(robot_prim)
        mat_api.Bind(orange_mat, UsdShade.Tokens.strongerThanDescendants)
        
    # Color Wok Black
    wok_prim = stage.GetPrimAtPath("/World/Wok_Bowl")
    if wok_prim.IsValid():
        print(f"[DEBUG] Applying Black Material to standalone Wok at {wok_prim.GetPath()}")
        mat_api = UsdShade.MaterialBindingAPI.Apply(wok_prim)
        mat_api.Bind(black_mat, UsdShade.Tokens.strongerThanDescendants)
        UsdGeom.Mesh(wok_prim).CreateDisplayColorAttr().Set([Gf.Vec3f(0.05, 0.05, 0.05)])
        
    # Color Handle Black
    handle_prim = stage.GetPrimAtPath("/World/Wok_Bowl/Handle")
    if handle_prim.IsValid():
        mat_api = UsdShade.MaterialBindingAPI.Apply(handle_prim)
        mat_api.Bind(black_mat, UsdShade.Tokens.strongerThanDescendants)

    # Color Ground Plane Light Grey
    for prim in stage.Traverse():
        if "ground" in prim.GetPath().pathString.lower() and prim.IsA(UsdGeom.Gprim):
            mat_api = UsdShade.MaterialBindingAPI.Apply(prim)
            mat_api.Bind(grey_mat, UsdShade.Tokens.strongerThanDescendants)


# 4. Master Live Simulation Execution Loop
async def run_wok_simulation_master(prim_path: str = ROBOT_PRIM_PATH, total_seconds: float = 30.0):
    try:
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.types import ArticulationAction
        import omni.timeline
        import omni.kit.app

        stage = omni.usd.get_context().get_stage()

        # Step 1. Setup Scene Geometry & Popcorn (Includes Robot Auto-Spawn)
        if not setup_wok_and_popcorn(robot_prim_path=prim_path):
            return

        # Apply Aesthetics
        apply_aesthetics(stage, prim_path)

        # Step 2. Increase PhysX Sub-stepping to 120Hz for high-velocity stability
        for p in stage.Traverse():
            if p.IsA(UsdPhysics.Scene):
                physx_scene = PhysxSchema.PhysxSceneAPI.Apply(p)
                physx_scene.CreateTimeStepsPerSecondAttr().Set(120.0)
                print("[Physics] Set PhysX simulation rate to 120Hz!")
                break

        # Step 3. Start Physics Timeline
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.play()
            for _ in range(15):
                await omni.kit.app.get_app().next_update_async()

        # Step 4. Initialize Robot
        robot = Articulation(prim_path=prim_path, name="fanuc_robot")
        robot.initialize()
        controller = robot.get_articulation_controller()
        
        controller.set_gains(
            kps=np.array([10000000.0] * robot.num_dof),
            kds=np.array([100000.0] * robot.num_dof)
        )

        base_positions = robot.get_joint_positions()
        dt = 0.02
        step = 0
        max_steps = int(total_seconds / dt)

        print("=== [Live Motion] Wok Flipping Simulation Started ===")

        # Step 5. Active Motion Loop
        while step < max_steps:
            sim_time = step * dt
            
            pos_x, pos_z, pos_pitch = compute_scurve_wok_position(sim_time=sim_time)
            
            target_joint_positions = base_positions.copy()
            target_joint_positions[1] += pos_x       # Joint 2
            target_joint_positions[2] += pos_z       # Joint 3
            target_joint_positions[4] += pos_pitch   # Joint 5

            controller.apply_action(ArticulationAction(joint_positions=target_joint_positions))

            step += 1
            await omni.kit.app.get_app().next_update_async()

        print("=== [Live Motion] Wok Flipping Simulation Completed ===")

    except Exception as e:
        print("\n" + "="*50)
        print("CRITICAL ERROR IN WOK SIMULATION:")
        print("="*50)
        traceback.print_exc()
        print("="*50 + "\n")

# Run Master Simulation
asyncio.ensure_future(run_wok_simulation_master(ROBOT_PRIM_PATH))
